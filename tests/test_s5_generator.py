"""test_s5_generator.py — 작업 경로 생성기(app.narrative.generator): 장별 호출 + 칸별 검문 사다리(서술 설계 D36).

장치마다 따로 시험한다(방어가 두 겹이면 한쪽 변이가 다른 쪽에 가려 살아남는다 — 라운드 작업 함정).
사실 시트는 계약 예시(tests/contract)를 쓴다: 이진 예시는 문제 3개(P1~P3)·FN 위험 선언(D24 7.1)·최종 불합격.
"""
import asyncio
import json
from pathlib import Path

import anthropic
import httpx2
import pytest

from app.narrative import fallback, generator, prompt
from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.schemas import NarrativeRequest
from tests.conftest import fake_beta_message, requested_fields, section_reply

CONTRACT = Path(__file__).parent / "contract"
_REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _contract(name: str) -> NarrativeRequest:
    return NarrativeRequest.model_validate(json.loads((CONTRACT / f"fact_sheet_{name}.json").read_text()))


def _binary(**update) -> NarrativeRequest:
    req = _contract("binary")
    if update:
        req = req.model_copy(update={"fact_sheet": req.fact_sheet.model_copy(update=update)})
    return req


# D24(7.1 — FN 43건, 99건의 43.4%)를 지키는 기본 7.1 글.
D24_OK = "의뢰자가 위험하다고 밝힌 FN 은 43건이다."


def _client(make_section_client, texts=None, **kw):
    return make_section_client({"section_7_1": D24_OK, **(texts or {})}, **kw)


def _fields(fake) -> list[list[str]]:
    return [requested_fields(c.kwargs) for c in fake.beta.messages.create.call_args_list]


def _template(req: NarrativeRequest, sid: str) -> dict:
    fs = req.fact_sheet
    task = req.task_type.value
    return fallback.section_template(sid, fs, build_benchmark_refs(task, fs.metrics), compute_derived(fs, task_type=task))


async def _gen(client, req, **kw):
    return await generator.generate_sections(client, req, **kw)


# ── 첫 생성: 장별 호출 ──────────────────────────────────────────────────────────

async def test_three_chapter_calls_when_there_are_problems(make_section_client):
    fake = _client(make_section_client)
    gen = await _gen(fake, _binary())
    assert sorted(map(tuple, _fields(fake))) == sorted([
        ("section_7_1", "section_7_2", "section_7_3"),
        ("section_8_1", "section_8_2", "section_8_3"),
        ("section_9_1", "section_9_2", "recommendations"),
    ])
    assert gen.section_sources == {"7.1": "llm", "7.2": "llm", "7.3": "fixed", "8.1": "llm", "8.2": "llm",
                                   "8.3": "llm", "9": "llm"}
    assert gen.response.meta.source == "llm"
    assert [r.problem_id for r in gen.response.recommendations] == ["P1", "P2", "P3"]
    # 우선순위는 서버가 문제의 것으로 정한다(AI 에게 받지 않는다).
    assert [r.priority for r in gen.response.recommendations] == ["HIGH", "MEDIUM", "MEDIUM"]
    assert gen.response.conclusion.verdict == "FAIL"


async def test_chapter_calls_run_concurrently(make_section_client):
    in_flight, peak = 0, 0

    def slow(kwargs):
        return None

    fake = _client(make_section_client)
    original = fake.beta.messages.create.side_effect

    async def counted(*a, **k):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        try:
            return await original(*a, **k)
        finally:
            in_flight -= 1

    fake.beta.messages.create.side_effect = counted
    await _gen(fake, _binary())
    assert peak == 3


async def test_no_problems_skips_chapter_9(make_section_client):
    fake = _client(make_section_client)
    req = _binary(problems=[])
    gen = await _gen(fake, req)
    assert len(_fields(fake)) == 2 and all("recommendations" not in f for f in _fields(fake))
    assert gen.section_sources["9"] == "fixed"
    assert gen.trace["chapter_9_skipped"] is True
    t = _template(req, "9")
    assert gen.response.recommendation_narrative.data_quality == t["recommendation_narrative.data_quality"]
    assert gen.response.recommendations == []


def test_schemas_follow_structured_output_limits():
    for ids in (("7.1", "7.2", "7.3"), ("8.1", "8.2", "8.3"), ("9",), ("7.1",)):
        schema = prompt.response_schema(ids, ["P1", "P2"])
        text = json.dumps(schema)
        for banned in ("maxItems", "minimum", "maximum", "minLength", "maxLength", "reason", "rationale", "trace", "thinking"):
            assert banned not in text, (ids, banned)
        assert schema["required"] == list(schema["properties"]) and schema["additionalProperties"] is False
    nine = prompt.response_schema(("9",), ["P1", "P2"])["properties"]["recommendations"]
    assert nine["minItems"] == 1 and nine["items"]["properties"]["problem_id"]["enum"] == ["P1", "P2"]
    assert "priority" not in nine["items"]["properties"]  # 서버가 문제의 우선순위로 정한다
    with pytest.raises(ValueError):
        prompt.response_schema(("9",), [])


# ── 검사 → 걸린 칸만 다시 → 또 걸리면 그 칸만 템플릿 ──────────────────────────────

async def test_regeneration_targets_only_the_failed_section(make_section_client):
    def reply(kwargs):
        fields = requested_fields(kwargs)
        if fields[0] == "section_8_1":  # 8장 첫 호출에만 위반
            return {"section_8_2": "통과율 12.345% 이다."}
        return None

    fake = _client(make_section_client, replies=[reply, reply, reply])
    gen = await _gen(fake, _binary())
    assert _fields(fake)[3:] == [["section_8_2"]]
    assert gen.section_sources["8.2"] == "llm"
    assert gen.trace["sections"]["8.2"]["outcome"] == "second"
    assert gen.response.conclusion.narrative == "종합 판정의 사유를 적은 서술이다."
    assert [c["purpose"] for c in gen.trace["calls"]][3:] == ["regenerate"]
    # 다시 만들기 지시문은 그 칸 하나만 쓰라고 한다.
    regen_user = fake.beta.messages.create.call_args_list[3].kwargs["messages"][0]["content"]
    assert "8.2 종합 총평 칸을 다시 쓴다" in regen_user and "■ 8.1" not in regen_user


async def test_second_failure_uses_that_sections_template_only(make_section_client):
    fake = _client(make_section_client, {"section_8_2": "통과율 12.345% 이다."})
    req = _binary()
    gen = await _gen(fake, req)
    assert gen.section_sources["8.2"] == "fallback"
    assert gen.response.conclusion.narrative == _template(req, "8.2")["conclusion.narrative"]
    assert gen.section_sources["8.1"] == gen.section_sources["8.3"] == "llm"
    assert gen.trace["sections"]["8.2"]["reason"] == "check:numbers"
    assert len(_fields(fake)) == 4  # 장 셋 + 8.2 다시 한 번(세 번째는 없다)


async def test_verdict_contradiction_triggers_the_ladder(make_section_client):
    fake = _client(make_section_client, {"section_8_2": "종합 판정은 합격이다."})
    gen = await _gen(fake, _binary())
    sec = gen.trace["sections"]["8.2"]
    assert gen.section_sources["8.2"] == "fallback" and sec["reason"] == "check:verdict_contradiction"
    assert [c["verdict_contradiction"] for c in sec["checks"]] == [True, True]


async def test_english_verdict_code_triggers_the_ladder(make_section_client):
    fake = _client(make_section_client, {"section_8_2": "본 모델의 판정은 FAIL이다."})
    gen = await _gen(fake, _binary())
    sec = gen.trace["sections"]["8.2"]
    assert sec["reason"] == "check:code_words" and sec["checks"][0]["code_words"] == ["FAIL"]


async def test_empty_section_triggers_the_ladder(make_section_client):
    fake = _client(make_section_client, {"section_7_2": "   "})
    gen = await _gen(fake, _binary())
    assert gen.trace["sections"]["7.2"]["reason"] == "check:empty"


async def test_missing_field_is_regenerated(make_section_client):
    bad = fake_beta_message({"section_7_1": D24_OK, "section_7_2": "분포 서술이다."})  # 7.3 빠짐(스키마 밖)

    def reply(kwargs):
        if requested_fields(kwargs)[0] == "section_8_1":
            return fake_beta_message({"section_8_1": "기준 비교 서술이다.", "section_8_3": "위험 서술이다."})
        return None

    fake = _client(make_section_client, replies=[reply, reply, reply])
    gen = await _gen(fake, _binary())
    assert ["section_8_2"] in _fields(fake)[3:]
    assert gen.section_sources["8.2"] == "llm"
    del bad


# ── 7.3 추가 관찰(D38) ─────────────────────────────────────────────────────────

async def test_additional_observation_with_a_cited_number_is_printed(make_section_client):
    text = "확신 오류가 31건으로 오분류의 상당 부분을 차지한다."
    fake = _client(make_section_client, {"section_7_3": text})
    gen = await _gen(fake, _binary())
    assert gen.section_sources["7.3"] == "llm"
    assert gen.response.interpretation.additional_observations == text
    assert gen.trace["additional_observations_raw"] == [text]


@pytest.mark.parametrize("text, reason", [
    ("확신 오류가 77건으로 많다.", "check:numbers"),     # 허용 목록 밖
    ("오분류의 성격이 특이하다.", "check:no_citation"),   # 숫자 인용 없음
])
async def test_failing_additional_observation_is_dropped_without_regenerating(make_section_client, text, reason):
    fake = _client(make_section_client, {"section_7_3": text})
    gen = await _gen(fake, _binary())
    assert gen.section_sources["7.3"] == "fixed"
    assert gen.response.interpretation.additional_observations == ""
    sec = gen.trace["sections"]["7.3"]
    assert (sec["outcome"], sec["http_calls"]) == ("dropped", 1) and sec["reason"].startswith(reason)
    assert len(_fields(fake)) == 3  # 7.3 은 다시 만들지 않는다
    assert gen.trace["additional_observations_raw"] == [text]  # 승격 자료로 원문은 내부 기록에


async def test_empty_additional_observation_is_fixed(make_section_client):
    gen = await _gen(_client(make_section_client), _binary())
    assert gen.section_sources["7.3"] == "fixed" and gen.response.interpretation.additional_observations == ""


# ── D24 선언 숫자 ───────────────────────────────────────────────────────────────

async def test_missing_declared_number_regenerates_then_keeps_the_ai_text(make_section_client):
    fake = make_section_client({"section_7_1": "오분류의 방향을 해석한 서술이다."})  # FN 숫자 없음
    gen = await _gen(fake, _binary())
    sec = gen.trace["sections"]["7.1"]
    assert gen.section_sources["7.1"] == "llm"  # 다시 만든 뒤 그대로 싣는다
    assert (sec["outcome"], sec["http_calls"], sec["d24_missing"]) == ("second", 2, True)
    assert [c["d24_missing"] for c in sec["checks"]] == [True, True]


@pytest.mark.parametrize("text", ["FN 은 43건이다.", "FN 이 전체 오류의 43.4%이다.", "비율은 43.43%이다."])
async def test_declared_number_in_any_allowed_form_passes(make_section_client, text):
    fake = make_section_client({"section_7_1": text})
    gen = await _gen(fake, _binary())
    assert gen.trace["sections"]["7.1"]["checks"][0]["d24_missing"] is False
    assert gen.trace["sections"]["7.1"]["http_calls"] == 1


async def test_numbers_and_d24_together_regenerate_once(make_section_client):
    fake = make_section_client({"section_7_1": "오분류는 12.345% 이다."})
    gen = await _gen(fake, _binary())
    assert _fields(fake).count(["section_7_1"]) == 1
    assert gen.section_sources["7.1"] == "fallback"


async def test_d24_is_off_when_declaration_is_not_special(make_section_client):
    req = _binary()
    facts = req.fact_sheet.facts.model_copy(update={"declared_risk_errors": None})
    req = req.model_copy(update={"fact_sheet": req.fact_sheet.model_copy(update={"facts": facts})})
    fake = make_section_client({"section_7_1": "오분류의 방향을 해석한 서술이다."})
    gen = await _gen(fake, req)
    assert gen.trace["sections"]["7.1"]["http_calls"] == 1 and not gen.trace["sections"]["7.1"]["d24_missing"]


# ── 9칸: 기대 효과 예측 수치·빠진 권고 ──────────────────────────────────────────

def _recs(impact: str, ids=("P1", "P2", "P3")):
    return [{"problem_id": i, "category": "점검", "action": "재평가한다.", "expected_impact": impact} for i in ids]


async def test_predicted_number_in_expected_impact_regenerates_section_9(make_section_client):
    fake = _client(make_section_client, {"recommendations": _recs("F1 이 0.05 향상된다.")})
    gen = await _gen(fake, _binary())
    sec = gen.trace["sections"]["9"]
    assert gen.section_sources["9"] == "fallback" and sec["reason"] == "check:impact_number"
    assert sec["checks"][0]["impact_numbers"] == ["0.05", "0.05", "0.05"]
    assert gen.section_sources["8.2"] == "llm"  # 9칸만 — 서술 전체를 버리지 않는다(사용자 결정)


@pytest.mark.parametrize("impact", ["1차 원인을 찾는다.", "3개월 내 재평가 근거를 얻는다.", "2회 재평가로 판정을 확인한다."])
async def test_non_predictive_numbers_in_expected_impact_are_allowed(make_section_client, impact):
    fake = _client(make_section_client, {"recommendations": _recs(impact)})
    gen = await _gen(fake, _binary())
    assert gen.trace["sections"]["9"]["checks"][0]["impact_numbers"] == []


async def test_recommendation_for_one_problem_only_regenerates_then_fills(make_section_client):
    fake = _client(make_section_client, {"recommendations": _recs("오류가 줄어든다.", ids=("P1",))})
    gen = await _gen(fake, _binary())
    sec = gen.trace["sections"]["9"]
    assert (sec["outcome"], sec["http_calls"]) == ("filled", 2)
    assert gen.trace["recommendations_from_template"] == ["P2", "P3"]
    assert [r.problem_id for r in gen.response.recommendations] == ["P1", "P2", "P3"]
    assert gen.section_sources["9"] == "fallback"


# ── 호출 실패 ──────────────────────────────────────────────────────────────────

def _refusal():
    return fake_beta_message(None, stop_reason="refusal", stop_details={"type": "refusal", "category": "frontier_llm"})


async def test_refused_chapter_uses_templates_and_is_not_called_again(make_section_client):
    def reply(kwargs):
        return _refusal() if requested_fields(kwargs)[0] == "section_8_1" else None

    fake = _client(make_section_client, replies=[reply, reply, reply])
    req = _binary()
    gen = await _gen(fake, req)
    assert len(_fields(fake)) == 3
    assert [gen.section_sources[s] for s in ("8.1", "8.2", "8.3")] == ["fallback"] * 3
    assert gen.section_sources["7.1"] == "llm" and gen.section_sources["9"] == "llm"
    assert gen.response.conclusion.risks == _template(req, "8.3")["conclusion.risks"]
    assert {c.get("refusal_category") for c in gen.trace["calls"] if c["sections"][0] == "8.1"} == {"frontier_llm"}


async def test_truncated_chapter_is_not_called_again(make_section_client):
    def reply(kwargs):
        if requested_fields(kwargs)[0] == "section_7_1":
            return fake_beta_message(None, stop_reason="max_tokens")
        return None

    fake = _client(make_section_client, replies=[reply, reply, reply])
    gen = await _gen(fake, _binary())
    assert len(_fields(fake)) == 3
    assert gen.trace["sections"]["7.1"]["reason"] == "max_tokens"
    assert gen.section_sources["7.3"] == "fixed"


@pytest.mark.parametrize("error", [
    anthropic.InternalServerError("down", response=httpx2.Response(500, request=_REQ), body=None),
    anthropic.APIConnectionError(request=_REQ),
])
async def test_call_error_retries_the_chapter_once(make_section_client, error):
    seen = {"n": 0}

    def reply(kwargs):
        if requested_fields(kwargs)[0] == "section_7_1":
            seen["n"] += 1
            if seen["n"] == 1:
                return error
        return None

    fake = _client(make_section_client, replies=[reply] * 4)
    gen = await _gen(fake, _binary())
    assert _fields(fake).count(["section_7_1", "section_7_2", "section_7_3"]) == 2
    assert gen.section_sources["7.1"] == "llm"
    assert gen.trace["sections"]["7.1"]["http_calls"] == 2
    assert [c["purpose"] for c in gen.trace["calls"]].count("retry") == 1


async def test_retried_chapter_is_not_regenerated_again(make_section_client):
    """첫 장 호출이 재시도됐으면 그 칸은 이미 HTTP 2회다 — 검사에 걸려도 다시 부르지 않고 템플릿."""
    seen = {"n": 0}

    def reply(kwargs):
        if requested_fields(kwargs)[0] == "section_7_1":
            seen["n"] += 1
            if seen["n"] == 1:
                return anthropic.APIConnectionError(request=_REQ)
            return {"section_7_2": "분포 비율은 12.345% 이다."}
        return None

    fake = _client(make_section_client, replies=[reply] * 4)
    gen = await _gen(fake, _binary())
    assert len(_fields(fake)) == 4
    assert gen.section_sources["7.2"] == "fallback" and gen.trace["sections"]["7.2"]["http_calls"] == 2


async def test_invalid_json_retries_once(make_section_client):
    def reply(kwargs):
        if requested_fields(kwargs)[0] == "section_8_1" and not reply.done:
            reply.done = True
            return fake_beta_message("not json")
        return None
    reply.done = False

    fake = _client(make_section_client, replies=[reply] * 4)
    gen = await _gen(fake, _binary())
    assert gen.section_sources["8.2"] == "llm" and gen.trace["sections"]["8.2"]["http_calls"] == 2


async def test_rate_limited_chapters_are_retried_one_at_a_time(make_section_client):
    limited = {"n": 0}
    in_flight, peak_retry = 0, 0

    def reply(kwargs):
        if limited["n"] < 3:  # 첫 장 호출 셋이 모두 429
            limited["n"] += 1
            return anthropic.RateLimitError("slow", response=httpx2.Response(429, request=_REQ), body=None)
        return None

    fake = _client(make_section_client, replies=[reply] * 6)
    original = fake.beta.messages.create.side_effect

    async def counted(*a, **k):
        nonlocal in_flight, peak_retry
        in_flight += 1
        if limited["n"] >= 3:
            peak_retry = max(peak_retry, in_flight)
        await asyncio.sleep(0.02)
        try:
            return await original(*a, **k)
        finally:
            in_flight -= 1

    fake.beta.messages.create.side_effect = counted
    gen = await _gen(fake, _binary())
    assert gen.trace["rate_limited"] == 3
    assert peak_retry == 1  # 차례로
    assert len(_fields(fake)) == 6 and gen.response.meta.source == "llm"


async def test_second_rate_limit_falls_back(make_section_client):
    err = anthropic.RateLimitError("slow", response=httpx2.Response(429, request=_REQ), body=None)
    fake = _client(make_section_client, replies=[err] * 6)
    gen = await _gen(fake, _binary())
    assert set(gen.section_sources.values()) <= {"fallback", "fixed"}
    assert gen.trace["rate_limited"] == 6 and len(_fields(fake)) == 6


# ── 최종본 재검사(따로 시험 — 칸 검사를 끄고 본다) ────────────────────────────────

async def test_final_recheck_catches_a_contradiction_the_section_check_let_through(make_section_client, monkeypatch):
    from app.narrative import checks

    real = checks.check_section
    monkeypatch.setattr(checks, "check_section", lambda *a, **k: checks.SectionCheck(section=a[0]))
    fake = _client(make_section_client, {"section_8_2": "종합 판정은 합격이다."})
    req = _binary()
    gen = await _gen(fake, req)
    assert gen.trace["final_recheck"] == ["8.2"]
    assert gen.section_sources["8.2"] == "fallback"
    assert gen.response.conclusion.narrative == _template(req, "8.2")["conclusion.narrative"]
    assert real is not checks.check_section


# ── 시간·슬롯 ──────────────────────────────────────────────────────────────────

async def test_no_time_left_means_no_call(make_section_client):
    fake = _client(make_section_client)
    gen = await _gen(fake, _binary(), deadline_seconds=generator.MIN_CALL_SECONDS - 1)
    fake.beta.messages.create.assert_not_called()
    assert gen.trace["sections"]["7.1"]["reason"] == "deadline"
    assert all(c["http"] is False for c in gen.trace["calls"])


async def test_call_timeout_is_a_retryable_failure(make_section_client):
    fake = _client(make_section_client, delay=0.3)
    gen = await _gen(fake, _binary(), call_timeout=0.05)
    assert set(gen.section_sources.values()) <= {"fallback", "fixed"}
    assert [c.get("failure") for c in gen.trace["calls"]].count("timeout") == 6  # 장 셋 × (첫 호출 + 재시도)
    assert all(c["http"] for c in gen.trace["calls"])


async def test_slot_wait_limit_is_busy(make_section_client):
    import contextlib

    from app.core import concurrency

    fake = _client(make_section_client)
    async with contextlib.AsyncExitStack() as stack:
        for _ in range(concurrency.MAX_CONCURRENT_NARRATIVE_CALLS):
            await stack.enter_async_context(concurrency.narrative_slot())
        gen = await asyncio.wait_for(_gen(fake, _binary(), slot_wait=0.05), 5)
    fake.beta.messages.create.assert_not_called()
    assert gen.trace["sections"]["8.1"]["reason"] == "busy"


# ── 길·키·강제 사유 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("client, reason", [(None, "no_key")])
async def test_without_a_client_every_section_is_a_template(client, reason):
    req = _binary()
    gen = await _gen(client, req)
    assert gen.section_sources == {"7.1": "fallback", "7.2": "fallback", "7.3": "fixed", "8.1": "fallback",
                                   "8.2": "fallback", "8.3": "fallback", "9": "fallback"}
    assert gen.response.meta.reason == reason and gen.trace["calls"] == []
    whole = fallback.build_fallback_narrative(
        req.fact_sheet, build_benchmark_refs("binary", req.fact_sheet.metrics),
        compute_derived(req.fact_sheet, task_type="binary"), reason=reason,
    )
    assert gen.response.model_dump(exclude={"meta"}) == whole.model_dump(exclude={"meta"})


async def test_forced_reason_skips_the_model(make_section_client):
    fake = _client(make_section_client)
    gen = await _gen(fake, _binary(), forced_reason="budget_exceeded")
    fake.beta.messages.create.assert_not_called()
    assert gen.trace["reason"] == "budget_exceeded"


async def test_prompt_over_the_limit_is_input_too_large(make_section_client, monkeypatch):
    monkeypatch.setattr(prompt, "MAX_PROMPT_CHARS", 100)
    fake = _client(make_section_client)
    gen = await _gen(fake, _binary())
    fake.beta.messages.create.assert_not_called()
    assert gen.trace["reason"] == "input_too_large"


async def test_effort_and_calls_are_recorded(make_section_client, monkeypatch):
    monkeypatch.setenv("NARRATIVE_JOB_EFFORT", "medium")
    fake = _client(make_section_client, usage=(2000, 300))
    gen = await _gen(fake, _binary())
    assert {c.kwargs["output_config"]["effort"] for c in fake.beta.messages.create.call_args_list} == {"medium"}
    assert gen.trace["effort"] == "medium"
    call = gen.trace["calls"][0]
    assert call["http"] and call["input_tokens"] == 2000 and call["cost_usd"] == pytest.approx(0.007)
    assert call["fallbacks_sent"] is True and call["response_model"] == "claude-sonnet-5-5"


@pytest.mark.parametrize("value, expected", [("", "high"), ("fast", "high"), (" Medium ", "medium"), ("low", "low")])
def test_job_effort_env(monkeypatch, value, expected):
    monkeypatch.setenv("NARRATIVE_JOB_EFFORT", value)
    assert generator.job_effort() == expected


async def test_on_stage_order(make_section_client):
    stages = []

    async def on_stage(name):
        stages.append(name)

    fake = _client(make_section_client, {"section_8_2": "통과율 12.345% 이다."})
    await _gen(fake, _binary(), on_stage=on_stage)
    assert stages[:3] == ["generating"] * 3
    # 다시 만드는 호출은 슬롯을 얻어도 '다시 만드는 중'이다(S5 검토 jobs-3 — 전에는 곧바로 generating 으로 덮였다).
    assert stages[3:] == ["checking", "regenerating", "regenerating", "checking"]


@pytest.mark.parametrize("name", ["binary", "multiclass", "multilabel", "legacy_main"])
async def test_every_contract_example_goes_through(make_section_client, name):
    req = _contract(name)
    fake = make_section_client()
    gen = await _gen(fake, req)
    assert gen.response.conclusion.verdict == req.fact_sheet.verdict
    assert set(gen.section_sources) == {"7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "9"}
    assert section_reply  # 가짜 응답 도우미 사용 확인



async def test_each_sequential_retry_refreshes_the_job_stage(make_section_client):
    """429 차례 호출이 길어져도 작업 행의 갱신 시각이 움직이게 — 차례 호출마다 regenerating 을 알린다."""
    err = anthropic.RateLimitError("slow", response=httpx2.Response(429, request=_REQ), body=None)
    stages = []

    async def on_stage(name):
        stages.append(name)

    fake = _client(make_section_client, replies=[err, err, err])
    await _gen(fake, _binary(), on_stage=on_stage)
    after_check = stages[stages.index("checking") + 1:]
    # 다시 부르기 시작 1 + 차례 호출 셋마다(대기 전 1 + 슬롯을 얻고 1)
    assert after_check.count("regenerating") == 7 and "generating" not in after_check


async def test_trimmed_counts_are_allowed_numbers(monkeypatch):
    """지시문에서 덜어 낸 표의 개수는 원본 사실 시트에 없는 수일 수 있다 — 허용 목록에 넣는다(바탕값과 겹치지 않는 7777 로)."""
    from app.narrative import fact_view

    def fake_step(view):
        fact_view._note(view, "가짜 표", 7777)
        return True

    monkeypatch.setattr(prompt, "MAX_PROMPT_CHARS", 10)  # 늘 넘친다 — 모든 단계가 돈다
    monkeypatch.setattr(fact_view, "TRIM_STEPS", (("fake", fake_step),))
    *_, whitelist, _conclusions, ctx, _exempt, _masks = generator._prepare(_binary())
    assert ctx.view["덜어 낸 표(일부만 실림)"]["가짜 표"] == 7777
    assert "7777" in whitelist



# ── S5 독립 검토(wf_a65c97fb-fec) 반영 ─────────────────────────────────────────

async def test_slot_wait_counts_against_the_deadline(make_section_client, monkeypatch):
    """ladder-1(P1): 슬롯을 기다린 시간만큼 마감을 넘기지 않는다 — 다 기다린 뒤 남은 시간이 모자라면 부르지 않는다."""
    import contextlib

    from app.core import concurrency

    monkeypatch.setattr(generator, "MIN_CALL_SECONDS", 0.05)
    fake = _client(make_section_client, delay=0.45)

    async def hog():
        async with contextlib.AsyncExitStack() as stack:
            for _ in range(concurrency.MAX_CONCURRENT_NARRATIVE_CALLS):
                await stack.enter_async_context(concurrency.narrative_slot())
            await asyncio.sleep(0.3)

    holder = asyncio.create_task(hog())
    await asyncio.sleep(0.01)
    gen = await _gen(fake, _binary(), deadline_seconds=0.5, slot_wait=0.4)
    await holder
    assert gen.trace["seconds"] <= 0.5 + 0.1
    timed = [c for c in gen.trace["calls"] if c.get("http")]
    assert all(c["slot_wait_seconds"] >= 0.25 for c in timed)


async def test_kept_first_draft_survives_a_failed_regeneration(make_section_client):
    """ladder-2: D24 만 빠진 첫 글은 다시 만들기가 거절돼도 그대로 싣는다(D24 빠짐은 템플릿 사유가 아니다)."""
    def reply(kwargs):
        if requested_fields(kwargs) == ["section_7_1"]:
            return fake_beta_message(None, stop_reason="refusal", stop_details={"type": "refusal", "category": "bio"})
        return None

    fake = make_section_client({"section_7_1": "오분류의 방향을 해석한 서술이다."}, replies=[reply] * 4)
    gen = await _gen(fake, _binary())
    sec = gen.trace["sections"]["7.1"]
    assert gen.section_sources["7.1"] == "llm"
    assert (sec["outcome"], sec["d24_missing"]) == ("kept_first", True)
    assert gen.response.interpretation.confusion_analysis == "오분류의 방향을 해석한 서술이다."


async def test_kept_first_draft_survives_a_regeneration_that_fails_checks(make_section_client):
    def reply(kwargs):
        if requested_fields(kwargs) == ["section_7_1"]:
            return {"section_7_1": "오분류는 12.345% 이다."}
        return None

    fake = make_section_client({"section_7_1": "오분류의 방향을 해석한 서술이다."}, replies=[reply] * 4)
    gen = await _gen(fake, _binary())
    assert gen.section_sources["7.1"] == "llm" and gen.trace["sections"]["7.1"]["outcome"] == "kept_first"


async def test_one_missing_recommendation_then_failed_regeneration_fills(make_section_client):
    only_p1 = _recs("오류가 줄어든다.", ids=("P1",))

    def reply(kwargs):
        if requested_fields(kwargs)[0] == "section_9_1" and len(fake.beta.messages.create.call_args_list) > 3:
            return anthropic.APIConnectionError(request=_REQ)
        return None

    fake = _client(make_section_client, {"recommendations": only_p1}, replies=[reply] * 4)
    gen = await _gen(fake, _binary())
    sec = gen.trace["sections"]["9"]
    assert sec["outcome"] == "filled" and gen.trace["recommendations_from_template"] == ["P2", "P3"]
    assert gen.response.recommendation_narrative.data_quality == "데이터 측면의 권고 서술이다."  # AI 글은 남는다


async def test_dropped_7_3_after_final_recheck_is_recorded_as_fixed(make_section_client, monkeypatch):
    """ladder-3: 최종 재검사로 7.3 이 빠지면 기록도 fixed 다(전에는 'first' 로 남았다)."""
    from app.narrative import checks

    monkeypatch.setattr(checks, "check_section", lambda *a, **k: checks.SectionCheck(section=a[0]))
    fake = _client(make_section_client, {"section_7_3": "종합 판정은 합격이다. FN 은 43건이다."})
    gen = await _gen(fake, _binary())
    sec = gen.trace["sections"]["7.3"]
    assert gen.trace["final_recheck"] == ["7.3"]
    assert (sec["source"], sec["outcome"], sec["reason"]) == ("fixed", "fixed", "final_verdict_contradiction")


async def test_rate_limited_call_costs_zero_not_unknown(make_section_client):
    """jobs-2: 429 는 서버가 거절한 것이라 비용 0 이다 — 작업 비용이 '모름'이 되지 않는다."""
    from app.narrative import jobs

    err = anthropic.RateLimitError("slow", response=httpx2.Response(429, request=_REQ), body=None)
    fake = _client(make_section_client, replies=[err])
    gen = await _gen(fake, _binary())
    limited = [c for c in gen.trace["calls"] if c.get("failure") == "rate_limited"]
    assert limited and limited[0]["cost_usd"] == 0.0
    assert jobs.total_cost(jobs.http_calls(gen.trace["calls"])) == pytest.approx(0.007 * 3)  # 성공 셋(장 둘 + 재시도), 429 는 0



async def test_regenerated_section_9_missing_only_a_recommendation_is_filled(make_section_client):
    """첫 9칸은 숫자 위반(보관할 글 없음), 다시 만든 글은 권고 하나만 빠짐 → 빠진 권고만 채운다(D22)."""
    bad = {"section_9_1": "오분류가 12.345% 줄어든다.", "recommendations": _recs("오류가 줄어든다.")}

    def reply(kwargs):
        if requested_fields(kwargs)[0] == "section_9_1":
            reply.n += 1
            return bad if reply.n == 1 else {"recommendations": _recs("오류가 줄어든다.", ids=("P1", "P2"))}
        return None
    reply.n = 0

    fake = _client(make_section_client, replies=[reply] * 4)
    gen = await _gen(fake, _binary())
    assert gen.trace["sections"]["9"]["outcome"] == "filled"
    assert gen.trace["recommendations_from_template"] == ["P3"]


async def test_slot_wait_is_capped_by_the_time_left(make_section_client, monkeypatch):
    """슬롯이 오래 막혀도 남은 시간 안에서만 기다린다 — 슬롯 대기 상한(5초)이 길어도 마감 0.3초 안에 끝난다."""
    import contextlib

    from app.core import concurrency

    monkeypatch.setattr(generator, "MIN_CALL_SECONDS", 0.05)
    fake = _client(make_section_client)
    release = asyncio.Event()

    async def hog():
        async with contextlib.AsyncExitStack() as stack:
            for _ in range(concurrency.MAX_CONCURRENT_NARRATIVE_CALLS):
                await stack.enter_async_context(concurrency.narrative_slot())
            await release.wait()

    holder = asyncio.create_task(hog())
    await asyncio.sleep(0.01)
    try:
        gen = await asyncio.wait_for(_gen(fake, _binary(), deadline_seconds=0.3, slot_wait=5.0), 3)
    finally:
        release.set()
        await holder
    assert gen.trace["seconds"] < 0.5
    assert gen.trace["sections"]["8.1"]["reason"] == "busy"
    fake.beta.messages.create.assert_not_called()
