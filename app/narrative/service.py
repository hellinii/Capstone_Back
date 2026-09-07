"""app/narrative/service.py — LLM 서술 생성 오케스트레이션

fact_sheet 로부터 파생값·기준치·숫자 화이트리스트를 준비하고 LLM 을 호출한 뒤, grounding
검증(환각 방어)을 거쳐 응답을 조립한다. 키 없음/호출 실패/검증 실패/조립 오류 시 규칙 폴백으로
대체한다. 파생 계산은 derived, 환각 방어는 grounding 모듈에 위임한다.

상호작용
- 의존(import): openai(런타임 client), app.core.schemas, app.narrative.baselines/fallback/
  prompt/derived(compute_derived)/grounding(build_number_whitelist·verify_grounding·_collect_grounding_texts)
- 사용처: app.narrative.router(generate_narrative_endpoint), tests
"""
import json
import logging

from app.narrative.schemas import ConclusionOut, InterpretationOut, NarrativeMeta, NarrativeRequest, NarrativeResponse, RecommendationNarrativeOut, RecommendationOut
from app.narrative.baselines import build_benchmark_refs
from app.narrative.fallback import build_fallback_narrative
from app.narrative.prompt import build_system_prompt, build_user_prompt, build_response_schema
from app.narrative.derived import compute_derived
from app.narrative.grounding import (
    build_number_whitelist,
    find_verdict_contradictions,
    verify_grounding,
    _collect_grounding_texts,
)
from app.core import llm_budget
from app.core.concurrency import llm_slot, run_cpu_bound

logger = logging.getLogger(__name__)

_MODEL = "gpt-4.1-nano"


def _prepare_facts(fs, task_type_value: str):
    """파생값·기준치·숫자 화이트리스트 준비 — 순수 CPU 구간(스레드풀로 오프로드된다).

    build_number_whitelist 는 혼동행렬을 행x열 이중 루프로 돌며 셀마다 문자열을
    여러 개 만들므로 행렬 차원에 대해 제곱으로 커진다(G-04b 실측: 2500x2500 입력이
    이벤트 루프를 15.1 초 막았다).
    """
    derived = compute_derived(fs)
    benchmark_refs = build_benchmark_refs(task_type_value, fs.metrics)
    whitelist = build_number_whitelist(fs, benchmark_refs, derived)
    return derived, benchmark_refs, whitelist


def _verify_llm_output(data: dict, whitelist):
    """LLM 출력 전수 수집 + grounding 검증 — 순수 CPU 구간."""
    return verify_grounding(_collect_grounding_texts(data), whitelist)


async def generate_narrative(client, req: NarrativeRequest) -> NarrativeResponse:
    """
    LLM 서술 생성 진입점.
    - client is None(무키) / LLM 호출 실패 / grounding 위반 → 규칙 폴백.
    - verdict 는 항상 fact_sheet 값으로 강제(LLM echo 무시).
    """
    fs = req.fact_sheet
    derived, benchmark_refs, whitelist = await run_cpu_bound(
        _prepare_facts, fs, req.task_type.value
    )

    # 1. 무키 → 폴백
    if client is None:
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="no_key")

    # 1-1. 시간당 예산 소진 → 폴백(ISSUES.md G-03, 결정 9).
    #      **429 가 아니라 200 + 폴백이다** — 프론트는 비 200 을 폴백이 아니라 오류로
    #      처리해 7·8·9절이 '생성 예정' 안내로 인쇄된다(사용자에게는 서비스 고장으로 보인다).
    if not llm_budget.try_consume():
        logger.warning(
            "LLM 시간당 예산 소진(%d회) → 규칙 폴백으로 강등합니다.",
            llm_budget.MAX_LLM_CALLS_PER_HOUR,
        )
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="budget_exceeded")

    # 2. LLM 호출
    try:
        async with llm_slot():
            response = await client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": build_system_prompt(req.report_purpose.value)},
                    {"role": "user", "content": build_user_prompt(fs.model_dump(), benchmark_refs, derived)},
                ],
                response_format=build_response_schema(),
                temperature=0,
                seed=4213,
            )
        data = json.loads(response.choices[0].message.content)
    except Exception as exc:
        # 조용한 품질 강등 지점(G-06). 종전에는 아무 흔적도 남지 않아, 남용으로 인한
        # 키 소진과 일시적 장애를 사후에 구분할 수 없었다.
        logger.warning("LLM 서술 호출 실패 → 규칙 폴백(api_error): %r", exc)
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="api_error")

    # 3~5. grounding 검증 + 응답 조립. 스키마 불일치/누락으로 조립이 실패해도
    #      500 대신 규칙 폴백(assembly_error)으로 강등한다(D7[1]).
    try:
        interp = data.get("interpretation", {})
        concl = data.get("conclusion", {})
        rec_narr = data.get("recommendation_narrative", {})
        recs = data.get("recommendations", [])

        # LLM 이 생성한 모든 문자열 필드를 전수 수집해 검증(수동 나열 제거 — D3b).
        grounding = await run_cpu_bound(_verify_llm_output, data, whitelist)

        # 위반 시 report_purpose 무관하게 폴백(internal fail-open 제거 — D3c).
        if not grounding.passed:
            # 어떤 숫자가 걸렸는지까지 남긴다 — '환각이 있었다'만으로는 원인을 못 찾는다.
            logger.warning(
                "서술 grounding 위반 → 규칙 폴백(grounding_failed). 검사 %d건, 위반 토큰 %s",
                grounding.checked, grounding.violations,
            )
            fb = build_fallback_narrative(fs, benchmark_refs, derived, reason="grounding_failed")
            fb.meta.grounding = grounding  # 어떤 환각이 잡혔는지 추적성 보존
            return fb

        # 숫자가 없는 정성 서술은 위 검사를 **무조건 통과**한다(ISSUES.md G-05).
        # 강제된 판정과 직접 모순되는 주장만 따로 잡는다 — 판정이 FAIL 인 성적서에
        # "모든 항목이 충족되었다"가 실리면 독자는 정반대 결론을 읽는다.
        contradictions = find_verdict_contradictions(_collect_grounding_texts(data), fs.verdict)
        if contradictions:
            logger.warning(
                "서술이 판정(%s)과 모순 → 규칙 폴백(verdict_contradiction): %s",
                fs.verdict, contradictions,
            )
            return build_fallback_narrative(
                fs, benchmark_refs, derived, reason="verdict_contradiction"
            )

        # 정상 — verdict 는 서버값으로 강제
        return NarrativeResponse(
            interpretation=InterpretationOut(**interp),
            conclusion=ConclusionOut(
                verdict=fs.verdict,
                benchmark=concl.get("benchmark", ""),
                narrative=concl.get("narrative", ""),
                risks=concl.get("risks", ""),
            ),
            recommendation_narrative=RecommendationNarrativeOut(**rec_narr),
            recommendations=[RecommendationOut(**r) for r in recs][:5],
            meta=NarrativeMeta(source="llm", model=_MODEL, grounding=grounding),
        )
    except Exception:
        logger.exception("narrative 조립/검증 실패 → 규칙 폴백(assembly_error)")
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="assembly_error")
