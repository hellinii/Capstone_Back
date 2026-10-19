"""app/narrative/service.py — LLM 서술 생성 오케스트레이션

fact_sheet 로부터 파생값·기준치·숫자 화이트리스트를 준비하고 LLM 을 호출한 뒤, grounding
검증(환각 방어)을 거쳐 응답을 조립한다. 키 없음/호출 실패/검증 실패/조립 오류 시 규칙 폴백으로
대체한다. 파생 계산은 derived, 환각 방어는 grounding 모듈에 위임한다.

상호작용
- 의존(import): app.narrative.llm(Claude 호출 한 곳), app.core.schemas, app.narrative.baselines/fallback/
  prompt/derived(compute_derived)/grounding(build_number_whitelist·verify_grounding·_collect_grounding_texts)
- 사용처: app.narrative.router(generate_narrative_endpoint), tests
"""
import asyncio
import logging

from app.narrative.schemas import BenchmarkRefOut, ConclusionOut, InterpretationOut, NarrativeMeta, NarrativeRequest, NarrativeResponse, RecommendationNarrativeOut, RecommendationOut
from app.narrative.baselines import build_benchmark_refs
from app.narrative.fallback import build_fallback_narrative, problem_recommendation
from app.narrative.prompt import build_system_prompt, build_user_prompt, build_response_schema
from app.narrative.derived import compute_derived
from app.narrative.llm import call_narrative_llm
from app.narrative.grounding import (
    build_number_whitelist,
    find_verdict_contradictions,
    has_number_token,
    verify_grounding,
    _collect_grounding_texts,
)
from app.core import llm_budget
from app.core.concurrency import llm_slot, run_cpu_bound

logger = logging.getLogger(__name__)

# 옛 동기 경로(/api/generate-narrative)의 서버 마감. 슬롯 대기와 호출을 합쳐 이 시간을 넘으면 200 + 규칙 문장
# (사유: 슬롯을 얻지 못했으면 busy, 호출 중이었으면 timeout). 프론트는 160초에 요청을 끊으므로 그보다 짧아야
# 빈 서술 대신 규칙 문장이 간다. llm.CALL_TIMEOUT_SECONDS × (llm.MAX_RETRIES + 1) 이 이 값 안에 든다(테스트 고정).
NARRATIVE_DEADLINE_SECONDS = 120.0


def _prepare_facts(fs, task_type_value: str, client_inputs: dict | None = None):
    """파생값·기준치·숫자 화이트리스트 준비 — 순수 CPU 구간(스레드풀로 오프로드된다).

    build_number_whitelist 는 혼동행렬을 행x열 이중 루프로 돌며 셀마다 문자열을
    여러 개 만들므로 행렬 차원에 대해 제곱으로 커진다(G-04b 실측: 2500x2500 입력이
    이벤트 루프를 15.1 초 막았다).
    """
    derived = compute_derived(fs)
    benchmark_refs = build_benchmark_refs(task_type_value, fs.metrics)
    whitelist = build_number_whitelist(fs, benchmark_refs, derived, client_inputs)
    return derived, benchmark_refs, whitelist


def _verify_llm_output(data: dict, whitelist):
    """LLM 출력 전수 수집 + grounding 검증 — 순수 CPU 구간."""
    return verify_grounding(_collect_grounding_texts(data), whitelist)


class _ImpactNumber(Exception):
    """권고의 기대 효과에 숫자가 있다 — 전체 폴백(impact_number)으로 강등하라는 신호."""


def _align_recommendations(recs: list, problems: list) -> list[RecommendationOut]:
    """LLM 권고를 문제 목록에 맞춘다(서술 설계 D22). 규칙은 계약서 §3 그대로다.

    - problem_id 가 비었거나 목록에 없는 권고는 버린다(어느 문제에도 대응하지 않는 권고).
    - 남은 권고의 기대 효과에 숫자가 있으면 _ImpactNumber — 근거 없는 예측값이라 전체를 폴백한다.
    - priority 는 문제의 것으로 강제한다(LLM 값 무시). 정렬은 문제 번호 순서, 같은 문제 안은 LLM 순서.
    - 권고가 하나도 없는 문제는 그 종류의 폴백 템플릿으로 채운다 — 9절 표가 문제 목록과 맞아야 한다.
    - 개수로 자르지 않는다(종전 [:5] 는 문제가 많을 때 뒤쪽 문제의 권고를 말없이 지웠다).
    문제가 없으면 권고도 없다. 권고 항목이 dict 가 아니면 예외가 나서 호출부가 assembly_error 로 강등한다.
    """
    by_id: dict[str, list[dict]] = {p.id: [] for p in problems}
    for r in recs:
        pid = str(r.get("problem_id") or "").strip()
        if pid not in by_id:
            continue
        if has_number_token(r.get("expected_impact", "")):
            raise _ImpactNumber(pid)
        by_id[pid].append(r)

    aligned: list[RecommendationOut] = []
    for p in problems:
        matched = by_id[p.id]
        if not matched:
            aligned.append(problem_recommendation(p))
            continue
        for r in matched:
            aligned.append(RecommendationOut(**{**r, "problem_id": p.id, "priority": p.priority}))
    return aligned


async def generate_narrative(client, req: NarrativeRequest) -> NarrativeResponse:
    """
    LLM 서술 생성 진입점.
    - client is None(무키) / LLM 호출 실패 / grounding 위반 → 규칙 폴백.
    - verdict 는 항상 fact_sheet 값으로 강제(LLM echo 무시).
    - 벤치마크 비교값(benchmark_refs)은 서술 경로와 무관하게 응답에 함께 싣는다 — 8.1 에 참조
      범위를 숫자로 찍는다(재료 목록 9절).
    """
    fs = req.fact_sheet
    derived, benchmark_refs, whitelist = await run_cpu_bound(
        _prepare_facts, fs, req.task_type.value, req.client_inputs
    )
    resp = await _narrate(client, req, fs, derived, benchmark_refs, whitelist)
    resp.benchmark_refs = [BenchmarkRefOut(**ref) for ref in benchmark_refs]
    return resp


async def _narrate(client, req: NarrativeRequest, fs, derived, benchmark_refs, whitelist) -> NarrativeResponse:
    """서술 본문 — LLM 또는 규칙 폴백. 반환 경로가 여럿이라 공통 부착은 호출자가 한다."""

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

    # 2. LLM 호출 — 슬롯 대기까지 서버 마감 안에서.
    try:
        system = build_system_prompt(req.report_purpose.value, req.task_type.value, fs)
        user = build_user_prompt(fs.model_dump(), benchmark_refs, derived, req.client_inputs)
        schema = build_response_schema(fs.problems)
    except Exception:
        # 종전처럼 500 이 아니라 규칙 폴백 — 요청 조립 실패도 서술 실패의 하나다.
        logger.exception("서술 요청 조립 실패 → 규칙 폴백(assembly_error)")
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="assembly_error")
    slot = {"acquired": False}

    async def _call():
        async with llm_slot():
            slot["acquired"] = True
            return await call_narrative_llm(client, system=system, user=user, schema=schema)

    try:
        async with asyncio.timeout(NARRATIVE_DEADLINE_SECONDS):
            result = await _call()
    except TimeoutError:
        reason = "timeout" if slot["acquired"] else "busy"
        logger.warning("서술 서버 마감(%.0f초) 초과 → 규칙 폴백(%s)", NARRATIVE_DEADLINE_SECONDS, reason)
        return build_fallback_narrative(fs, benchmark_refs, derived, reason=reason)
    except Exception:
        # call_narrative_llm 은 SDK 오류를 failure 로 돌려준다. 그 밖의 예외(조용한 품질 강등 지점, G-06)도
        # 500 이 아니라 규칙 폴백으로.
        logger.exception("LLM 서술 호출 중 예기치 않은 오류 → 규칙 폴백(api_error)")
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="api_error")
    if result.failure is not None:
        # 거절(세 경우 모두)·잘림·모델 없음·시간 초과·API 오류·JSON 오류 — 사유는 그대로 싣고, 자세한 것은
        # llm 의 호출 기록 로그에 있다(거절 종류·응답 모델·토큰).
        logger.warning("LLM 서술 호출 실패 → 규칙 폴백(%s)", result.failure)
        return build_fallback_narrative(fs, benchmark_refs, derived, reason=result.failure)
    data = result.data
    model = result.record.response_model or result.record.requested_model

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

        # 권고를 문제 목록에 묶는다(D22). 기대 효과에 숫자가 있으면 전체 폴백 — 권고 하나만 바꿔 끼우면
        # 같은 LLM 이 쓴 9절 서술(recommendation_narrative)과 어긋날 수 있다.
        try:
            aligned = _align_recommendations(recs, fs.problems)
        except _ImpactNumber as exc:
            logger.warning("권고 기대 효과에 숫자 → 규칙 폴백(impact_number): 문제 %s", exc)
            return build_fallback_narrative(fs, benchmark_refs, derived, reason="impact_number")

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
            recommendations=aligned,
            meta=NarrativeMeta(source="llm", model=model, grounding=grounding),
        )
    except Exception:
        logger.exception("narrative 조립/검증 실패 → 규칙 폴백(assembly_error)")
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="assembly_error")
