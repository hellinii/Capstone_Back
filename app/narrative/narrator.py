"""app/narrative/narrator.py — LLM 서술 생성 오케스트레이션

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

from app.core.schemas import (
    NarrativeRequest,
    NarrativeResponse,
    InterpretationOut,
    ConclusionOut,
    RecommendationNarrativeOut,
    RecommendationOut,
    NarrativeMeta,
)
from app.narrative.baselines import build_benchmark_refs
from app.narrative.fallback import build_fallback_narrative
from app.narrative.prompt import build_system_prompt, build_user_prompt, build_response_schema
from app.narrative.derived import compute_derived
from app.narrative.grounding import build_number_whitelist, verify_grounding, _collect_grounding_texts

_MODEL = "gpt-4.1-nano"


async def generate_narrative(client, req: NarrativeRequest) -> NarrativeResponse:
    """
    LLM 서술 생성 진입점.
    - client is None(무키) / LLM 호출 실패 / grounding 위반 → 규칙 폴백.
    - verdict 는 항상 fact_sheet 값으로 강제(LLM echo 무시).
    """
    fs = req.fact_sheet
    derived = compute_derived(fs)
    benchmark_refs = build_benchmark_refs(req.task_type.value, fs.metrics)
    whitelist = build_number_whitelist(fs, benchmark_refs, derived)

    # 1. 무키 → 폴백
    if client is None:
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="no_key")

    # 2. LLM 호출
    try:
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
    except Exception:
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="api_error")

    # 3~5. grounding 검증 + 응답 조립. 스키마 불일치/누락으로 조립이 실패해도
    #      500 대신 규칙 폴백(assembly_error)으로 강등한다(D7[1]).
    try:
        interp = data.get("interpretation", {})
        concl = data.get("conclusion", {})
        rec_narr = data.get("recommendation_narrative", {})
        recs = data.get("recommendations", [])

        # LLM 이 생성한 모든 문자열 필드를 전수 수집해 검증(수동 나열 제거 — D3b).
        grounding = verify_grounding(_collect_grounding_texts(data), whitelist)

        # 위반 시 report_purpose 무관하게 폴백(internal fail-open 제거 — D3c).
        if not grounding.passed:
            fb = build_fallback_narrative(fs, benchmark_refs, derived, reason="grounding_failed")
            fb.meta.grounding = grounding  # 어떤 환각이 잡혔는지 추적성 보존
            return fb

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
        logging.exception("narrative 조립/검증 실패 → 규칙 폴백(assembly_error)")
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="assembly_error")
