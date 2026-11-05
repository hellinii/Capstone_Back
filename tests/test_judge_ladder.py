"""tests/test_judge_ladder.py — 발급 불가 사다리(app.judge.ladder)와 심판 전용 재생성 길(GenerationSession), 심판 라운드 J5a.

계획 13-1~13-3·정6. 채점은 가짜 judge(차례로 JudgeRun 을 돌려줌)로, 생성은 가짜 Anthropic 클라이언트로 돈다 — 실호출 없음.
사유 격리(13-3) 세 겹: ① 재생성 길은 칸 id 만 받는다(서명 시험) ② 지시문이 생성 단계 다시 만들기와 바이트까지 같다
③ 가짜 심판이 기록·인용에 심은 표식이 어느 Anthropic 호출 인자에도 없다.
"""
from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path

import anthropic
import httpx2
import pytest

from app.judge import alert, ladder
from app.judge.core import JudgeRun
from app.narrative import fallback, generator, prompt
from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.schemas import NarrativeRequest
from tests.conftest import requested_fields

CONTRACT = Path(__file__).parent / "contract"
_REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
MARKER = "@@심판표식-7f3a@@"
REGEN_TEXT = "다시 쓴 데이터 분석 서술이다."
NARRATIVE = ("7.1", "7.2", "8.1", "8.2", "8.3", "9")


def _req(name: str) -> NarrativeRequest:
    return NarrativeRequest.model_validate(json.loads((CONTRACT / f"fact_sheet_{name}.json").read_text()))


def run_(gate, flagged=(), failure=None, unfixable=(), *, issuance="same", repair=(), blocking=None):
    """기본은 모순으로 걸린 채점 — 발급 규칙이 판정과 같고(issuance=gate) 걸린 칸이 곧 막는 칸이다. 사실 위반만 난 채점은
    issuance="pass" 로(판정자 판정은 blocked 여도 발급은 막지 않는다), 숫자 위반 칸은 repair 로 준다."""
    iss = gate if issuance == "same" else issuance
    return JudgeRun(gate, failure, list(flagged), [dict(u) for u in unfixable], None, {"criteria_version": "1.0-dev"},
                    issuance=iss, blocking_slots=list(flagged if blocking is None else blocking) if iss == "blocked" else [],
                    repair_slots=list(repair))


class FakeJudge:
    """차례로 JudgeRun 을 돌려준다. 받은 생성 결과(칸 출처)를 기록한다."""

    def __init__(self, *runs):
        self.runs = list(runs)
        self.seen: list[dict] = []

    async def __call__(self, result, deadline):
        self.seen.append(dict(result.section_sources))
        if not self.runs:
            raise AssertionError("가짜 심판: 준비한 판정보다 많이 불렸다")
        return self.runs.pop(0)


async def _session(make_section_client, name="multiclass_ed4", texts=None, replies=None):
    client = make_section_client({"section_7_1": "의뢰자가 위험하다고 밝힌 FN 은 43건이다.", **(texts or {})}, replies=replies)
    session = generator.GenerationSession(client, _req(name))
    await session.run()
    return session, client


def _calls(client) -> list[dict]:
    return [c.kwargs for c in client.beta.messages.create.call_args_list]


def _purposes(session) -> list[str]:
    return [c["purpose"] for c in session.trace["calls"]]


# ── 사다리 단계 ──


async def test_pass_on_first_round_changes_nothing(make_section_client):
    session, client = await _session(make_section_client)
    before = dict(session.result.section_sources)
    out = await ladder.run_ladder(session, FakeJudge(run_("pass")), deadline=None)
    assert (out.gate, out.held_kind, out.retry_allowed) == ("passed", None, False)
    assert out.result.section_sources == before and len(out.record["rounds"]) == 1
    assert "judge_regenerate" not in _purposes(session)


async def test_flagged_slot_is_regenerated_once_then_passes(make_section_client):
    session, client = await _session(make_section_client)
    n_before = len(_calls(client))
    judge = FakeJudge(run_("blocked", ["7.2"]), run_("pass"))
    out = await ladder.run_ladder(session, judge, deadline=None)
    assert out.gate == "passed" and len(out.record["rounds"]) == 2
    new = _calls(client)[n_before:]
    assert [requested_fields(k) for k in new] == [["section_7_2"]]          # 걸린 칸 하나만
    assert _purposes(session)[-1] == "judge_regenerate"
    sec = session.trace["sections"]["7.2"]
    assert (sec["source"], sec["outcome"], sec["http_calls"]) == ("llm", "judge_regenerated", 2)
    assert out.record["regenerated"] == {"7.2": "regenerated"}
    assert [r["after"] for r in out.record["rounds"]] == [None, "regenerate"]


async def test_flagged_again_after_regeneration_becomes_template(make_section_client):
    session, client = await _session(make_section_client)
    judge = FakeJudge(run_("blocked", ["7.2"]), run_("blocked", ["7.2", "8.3"]), run_("pass"))
    out = await ladder.run_ladder(session, judge, deadline=None)
    assert out.gate == "passed"
    assert [r["after"] for r in out.record["rounds"]] == [None, "regenerate", "template_flagged"]
    src = out.result.section_sources
    assert (src["7.2"], src["8.3"], src["8.2"]) == ("fallback", "fallback", "llm")  # 2차에 새로 걸린 8.3 도 바로 규칙 문장
    assert _purposes(session).count("judge_regenerate") == 1                        # 재생성은 작업당 한 바퀴
    assert out.record["templated"] == ["7.2", "8.3"]


async def test_still_blocked_goes_to_all_templates_then_held(make_section_client):
    session, client = await _session(make_section_client)
    judge = FakeJudge(*(run_("blocked", ["8.2"]) for _ in range(4)))
    out = await ladder.run_ladder(session, judge, deadline=None)
    assert (out.gate, out.held_kind, out.retry_allowed) == ("held", "gate_failed", False)
    assert [r["after"] for r in out.record["rounds"]] == [None, "regenerate", "template_flagged", "template_all"]
    assert all(out.result.section_sources[s] == "fallback" for s in NARRATIVE)
    assert judge.runs == []  # 채점 4번


async def test_held_result_text_is_all_templates(make_section_client):
    session, _ = await _session(make_section_client)
    out = await ladder.run_ladder(session, FakeJudge(*(run_("blocked", ["8.2"]) for _ in range(4))), deadline=None)
    req = _req("multiclass_ed4")
    fs, task = req.fact_sheet, req.task_type.value
    want = fallback.section_template("8.2", fs, build_benchmark_refs(task, fs.metrics, edition=4),
                                     compute_derived(fs, task_type=task), task_type=task, client_inputs=req.client_inputs,
                                     conclusions=session.conclusions, edition=4)
    assert out.result.response.conclusion.narrative == want["conclusion.narrative"]


async def test_unfixable_contradiction_skips_to_all_templates(make_section_client):
    """걸린 칸이 없고(표·고정 문구끼리 모순) 다시 만들 칸이 없으면 재생성·걸린 칸 단을 건너뛰고 전부 규칙 문장 → 채점 2."""
    session, client = await _session(make_section_client)
    n_before = len(_calls(client))
    judge = FakeJudge(run_("blocked", [], unfixable=[{"type": "value_mismatch"}]), run_("blocked", []))
    out = await ladder.run_ladder(session, judge, deadline=None)
    assert [r["after"] for r in out.record["rounds"]] == [None, "template_all"]
    assert (out.gate, out.held_kind) == ("held", "gate_failed") and len(_calls(client)) == n_before


async def test_template_slots_are_not_regenerated_and_held_when_nothing_changes(make_section_client):
    """키 없이 만든 서술(전부 규칙 문장)은 다시 만들 것이 없다 — 같은 글을 다시 채점하지 않고 보류(채점 1번)."""
    session = generator.GenerationSession(None, _req("multiclass_ed4"))
    await session.run()
    judge = FakeJudge(run_("blocked", ["7.2"]))
    out = await ladder.run_ladder(session, judge, deadline=None)
    assert (out.gate, out.held_kind, len(out.record["rounds"])) == ("held", "gate_failed", 1)


async def test_7_3_is_dropped_not_regenerated(make_section_client):
    session, client = await _session(make_section_client, texts={"section_7_3": "클래스 2 의 재현율이 가장 낮다."})
    assert session.result.section_sources["7.3"] == "llm"
    n_before = len(_calls(client))
    out = await ladder.run_ladder(session, FakeJudge(run_("blocked", ["7.3"]), run_("pass")), deadline=None)
    assert out.record["regenerated"] == {"7.3": "dropped"} and len(_calls(client)) == n_before
    sec = session.trace["sections"]["7.3"]
    assert (sec["source"], sec["outcome"]) == ("fixed", "dropped")
    assert out.result.response.interpretation.additional_observations == ""


async def test_regeneration_failure_or_failed_check_is_template(make_section_client):
    def reply(kwargs):
        if requested_fields(kwargs) == ["section_7_2"]:
            return {"section_7_2": "정확도는 12.345% 이다."}  # 숫자 검문에 걸림
        if requested_fields(kwargs) == ["section_8_3"]:
            return anthropic.APIConnectionError(request=_REQ)
        return None

    session, _ = await _session(make_section_client, replies=[None] * 3 + [reply] * 2)
    out = await ladder.run_ladder(session, FakeJudge(run_("blocked", ["7.2", "8.3"]), run_("pass")), deadline=None)
    assert out.record["regenerated"] == {"7.2": "template", "8.3": "template"}
    secs = session.trace["sections"]
    assert secs["7.2"]["reason"].startswith("judge_regenerate:check:") and secs["8.3"]["reason"] == "judge_regenerate:api_error"


async def test_per_slot_ai_calls_are_at_most_three(make_section_client):
    """첫 장 호출 + 생성 단계 다시 만들기 + 심판 재생성 = 칸당 최대 3번."""
    def first_bad(kwargs):
        return {"section_7_2": "정확도는 12.345% 이다."} if "section_7_2" in requested_fields(kwargs) else None

    session, client = await _session(make_section_client, replies=[first_bad] * 3)
    assert session.trace["sections"]["7.2"]["http_calls"] == 2
    await ladder.run_ladder(session, FakeJudge(run_("blocked", ["7.2"]), run_("pass")), deadline=None)
    assert session.trace["sections"]["7.2"]["http_calls"] == 3
    assert max(s["http_calls"] for s in session.trace["sections"].values()) <= 3


# ── 둬 둔 첫 글을 되살리지 않는다(정6 ⑦) ──


async def test_kept_d24_draft_is_not_restored_when_judge_regeneration_fails(make_section_client):
    """판 3 이진 예시: 7.1 첫 글이 D24 만 빠짐 → 다시 만든 글이 숫자에 걸려 첫 글(kept_first)이 실림. 심판이 7.1 을 걸고
    재생성이 또 검문에 걸리면 그 첫 글이 아니라 규칙 문장이다."""
    def reply(kwargs):
        if requested_fields(kwargs) == ["section_7_1"]:
            return {"section_7_1": "오분류는 12.345% 이다."}
        return None

    client = make_section_client({"section_7_1": "오분류의 방향을 해석한 서술이다."}, replies=[reply] * 6)
    session = generator.GenerationSession(client, _req("binary"))
    await session.run()
    assert session.trace["sections"]["7.1"]["outcome"] == "kept_first"
    out = await ladder.run_ladder(session, FakeJudge(run_("blocked", ["7.1"]), run_("pass")), deadline=None)
    assert out.result.section_sources["7.1"] == "fallback"
    assert out.result.response.interpretation.confusion_analysis != "오분류의 방향을 해석한 서술이다."
    assert session.states["7.1"].kept is None


async def test_section_9_missing_recommendation_on_judge_regeneration_is_full_template(make_section_client):
    """생성 단계라면 빠진 권고만 채우지만(filled), 심판 재생성에서 권고가 빠지면 9칸 전체가 규칙 문장이다."""
    one = [{"problem_id": "P1", "category": "점검", "action": "점검한다.", "expected_impact": "근거가 분명해진다."}]

    def reply(kwargs):
        return {"recommendations": one} if requested_fields(kwargs)[0] == "section_9_1" else None

    session, _ = await _session(make_section_client, replies=[None] * 3 + [reply])
    assert len(session.req.fact_sheet.problems) > 1
    out = await ladder.run_ladder(session, FakeJudge(run_("blocked", ["9"]), run_("pass")), deadline=None)
    sec = session.trace["sections"]["9"]
    assert (sec["source"], sec["outcome"]) == ("fallback", "template")
    assert "missing_recommendation" in sec["reason"]
    assert out.record["regenerated"] == {"9": "template"}


async def test_judge_regenerated_slot_goes_through_final_verdict_recheck(make_section_client, monkeypatch):
    from app.narrative import checks

    session, _ = await _session(make_section_client)
    monkeypatch.setattr(checks, "check_section", lambda *a, **k: checks.SectionCheck(section=a[0]))
    monkeypatch.setattr(generator, "find_verdict_contradictions", lambda texts, verdict: ["모순"])
    await ladder.run_ladder(session, FakeJudge(run_("blocked", ["8.2"]), run_("pass")), deadline=None)
    assert session.trace["sections"]["8.2"]["reason"] == "judge_regenerate:final_verdict_contradiction"


# ── 시간(정6 ⑥) ──


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


async def test_short_time_skips_regeneration_to_templates(make_section_client):
    session, client = await _session(make_section_client)
    n_before = len(_calls(client))
    clock = Clock()
    deadline = clock.t + ladder.JUDGE_MIN_SECONDS + ladder.REGENERATE_MIN_SECONDS - 1
    out = await ladder.run_ladder(session, FakeJudge(run_("blocked", ["7.2"]), run_("pass")), deadline=deadline, clock=clock)
    assert out.record["regenerate_skipped"] == "deadline" and len(_calls(client)) == n_before
    assert [r["after"] for r in out.record["rounds"]] == [None, "template_flagged"]
    assert out.result.section_sources["7.2"] == "fallback" and out.gate == "passed"


async def test_no_time_to_judge_is_held_timeout(make_section_client):
    session, _ = await _session(make_section_client)
    clock = Clock()
    judge = FakeJudge(run_("pass"))
    out = await ladder.run_ladder(session, judge, deadline=clock.t + ladder.JUDGE_MIN_SECONDS - 1, clock=clock)
    assert (out.gate, out.held_kind, out.retry_allowed) == ("held", "timeout", True)  # 게이트에 걸린 적 없음
    assert judge.seen == [] and out.record["rounds"] == [{"round": 1, "after": None, "skipped": "deadline"}]


async def test_no_time_after_a_block_is_held_without_retry(make_section_client):
    session, _ = await _session(make_section_client)
    clock = Clock()
    deadline = clock.t + ladder.JUDGE_MIN_SECONDS + 10

    class Slow(FakeJudge):
        async def __call__(self, result, dl):
            clock.t += 100  # 채점 한 번이 100초
            return await super().__call__(result, dl)

    out = await ladder.run_ladder(session, Slow(run_("blocked", ["7.2"])), deadline=deadline, clock=clock)
    assert (out.gate, out.held_kind, out.retry_allowed) == ("held", "timeout", False)


# ── 채점 실패 → 보류 종류·다시 만들기(Q6) ──


@pytest.mark.parametrize("failure, kind, retry", [
    ("transient", "judge_error", True),
    ("timeout", "judge_error", True),       # 호출 하나의 제한 — 작업 시간 초과가 아님
    ("deadline", "timeout", True),
    ("api_error", "judge_error", False),    # 4xx — 다시 해도 같음
    ("refusal", "judge_error", False),
    ("bad_output", "judge_error", False),
    ("uncited", "judge_error", False),
    ("no_claims", "judge_error", False),
    ("safety", "judge_error", False),
    ("error", "judge_error", False),
    ("no_material", "no_material", False),
    ("no_client", "no_key", False),
])
async def test_judge_failure_is_held_with_kind_and_retry(make_section_client, failure, kind, retry):
    session, _ = await _session(make_section_client)
    out = await ladder.run_ladder(session, FakeJudge(run_(None, failure=failure)), deadline=None)
    assert (out.gate, out.held_kind, out.retry_allowed, out.final) == ("held", kind, retry, None)


async def test_transient_failure_after_a_block_gets_no_retry(make_section_client):
    session, _ = await _session(make_section_client)
    out = await ladder.run_ladder(session, FakeJudge(run_("blocked", ["7.2"]), run_(None, failure="transient")), deadline=None)
    assert (out.held_kind, out.retry_allowed) == ("judge_error", False)


@pytest.mark.parametrize("kind, failure, blocked, want", [
    ("budget", None, False, True), ("budget", None, True, True),
    ("no_key", None, False, False), ("job_limit", None, False, False), ("no_material", None, False, False),
    ("gate_failed", None, True, False), ("timeout", None, False, True), ("timeout", None, True, False),
])
def test_retry_allowed_table(kind, failure, blocked, want):
    assert ladder.retry_allowed(kind, failure=failure, ever_blocked=blocked) is want


def test_held_kinds_match_the_customer_contract():
    contract = json.loads((CONTRACT / "judge_bundle_ed4.json").read_text(encoding="utf-8"))
    assert set(ladder.HELD_KINDS) == set(contract["held_kinds"])


async def test_odd_gate_value_is_never_passed(make_section_client):
    session, _ = await _session(make_section_client)
    out = await ladder.run_ladder(session, FakeJudge(run_("PASS")), deadline=None)
    assert out.gate == "held" and out.final is None


# ── 사유 격리(계획 13-3) ──


def test_regeneration_path_takes_slot_ids_only():
    params = list(inspect.signature(generator.GenerationSession.judge_regenerate).parameters)
    assert params == ["self", "sids", "seconds"]
    assert list(inspect.signature(generator.GenerationSession.to_template).parameters) == ["self", "sids", "reason"]


async def test_judge_regeneration_prompt_is_byte_identical_to_regenerate(make_section_client):
    session, client = await _session(make_section_client)
    n_before = len(_calls(client))
    await ladder.run_ladder(session, FakeJudge(run_("blocked", ["8.2"]), run_("pass")), deadline=None)
    sent = _calls(client)[n_before]
    want_user = prompt.build_user_message(session.ctx, ("8.2",), regenerate=True)
    users = [m for m in sent["messages"] if m["role"] == "user"]
    text = users[0]["content"] if isinstance(users[0]["content"], str) else "".join(b.get("text", "") for b in users[0]["content"])
    assert text == want_user
    assert sent["output_config"]["format"]["schema"] == prompt.response_schema(("8.2",), session.problem_ids)


async def test_judge_marker_never_reaches_the_generator(make_section_client):
    session, client = await _session(make_section_client)
    n_before = len(_calls(client))
    marked = JudgeRun("blocked", None, ["7.2", "8.2"], [{"type": "value_mismatch", "pieces": [MARKER], "quote": MARKER}],
                      {"faithfulness": {"client_input_assertive": 0, "note": MARKER}}, {"reason": MARKER, "quote": MARKER},
                      issuance="blocked", blocking_slots=["7.2", "8.2"])
    judge = FakeJudge(marked, JudgeRun("blocked", None, ["9"], [], None, {"reason": MARKER}, issuance="blocked",
                                       blocking_slots=["9"]), run_("pass"))
    out = await ladder.run_ladder(session, judge, deadline=None)
    new = _calls(client)[n_before:]
    assert new, "재생성 호출이 하나도 없으면 이 시험은 아무것도 보지 않는다"
    assert all(MARKER not in json.dumps(k, ensure_ascii=False, default=str) for k in new)
    assert MARKER not in json.dumps(out.result.response.model_dump(), ensure_ascii=False)
    assert MARKER not in json.dumps(session.trace, ensure_ascii=False, default=str)  # 생성 기록에도 없다(심판 기록은 따로)


# ── 내부 기록(16-1) ──


async def test_record_is_written_into_the_given_dict_and_summarized(make_section_client):
    session, _ = await _session(make_section_client)
    rec: dict = {}
    calls = [{"cost_usd": 0.01}, {"cost_usd": 0.02}]
    r1 = JudgeRun("blocked", None, ["7.2"], [], {"faithfulness": {"client_input_assertive": 2}},
                  {"criteria_version": "1.0-dev", "criteria_fingerprint": "abc", "material_check_version": "m1",
                   "extract": {"calls": calls}}, issuance="blocked", blocking_slots=["7.2"])
    r2 = JudgeRun("pass", None, [], [], {"faithfulness": {"client_input_assertive": 1}}, {"extract": {"calls": calls}}, issuance="pass")
    out = await ladder.run_ladder(session, FakeJudge(r1, r2), deadline=None, record=rec)
    assert out.record is rec and [r["round"] for r in rec["rounds"]] == [1, 2]
    s = ladder.summary(rec)
    assert s == {"gate": "passed", "held_kind": None, "failure": None, "retry_allowed": False, "rounds": 2,
                 "regenerated": ["7.2"], "templated": [], "number_templated": [], "cost_usd": 0.06, "seconds": s["seconds"],
                 "criteria_version": "1.0-dev", "criteria_fingerprint": "abc", "material_check_version": "m1",
                 "client_input_assertive": 1}
    json.dumps(rec, ensure_ascii=False)  # 직렬화된다


async def test_summary_cost_is_none_when_any_call_cost_is_unknown(make_section_client):
    session, _ = await _session(make_section_client)
    r = JudgeRun("pass", None, [], [], None, {"extract": {"calls": [{"cost_usd": None}]}}, issuance="pass")
    out = await ladder.run_ladder(session, FakeJudge(r), deadline=None)
    assert ladder.summary(out.record)["cost_usd"] is None


async def test_rounds_survive_an_outer_cancel(make_section_client):
    """바깥 제한이 채점 도중 끊어도 앞 회차 기록이 남는다(dict 에 바로 씀)."""
    import asyncio

    session, _ = await _session(make_section_client)
    rec: dict = {}

    class Hang(FakeJudge):
        async def __call__(self, result, dl):
            if len(self.seen) == 1:
                await asyncio.sleep(10)
            return await super().__call__(result, dl)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.5):
            await ladder.run_ladder(session, Hang(run_("blocked", ["7.2"]), run_("pass")), deadline=None, record=rec)
    assert [r["gate"] for r in rec["rounds"]] == ["blocked"] and rec["regenerated"] == {"7.2": "regenerated"}


# ── 운영자 알림(13-4·정7) ──


async def test_alert_logs_once_without_webhook(monkeypatch, caplog):
    monkeypatch.delenv(alert.WEBHOOK_ENV, raising=False)
    a = alert.JobAlert("job-1")
    with caplog.at_level(logging.WARNING, logger="app.judge.alert"):
        assert await a.send("held", "gate_failed", text="x", nested={"no": 1}) is True
        assert await a.send("held", "gate_failed") is False
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(alert.LOG_PREFIX)]
    assert len(lines) == 1
    payload = json.loads(lines[0][len(alert.LOG_PREFIX) + 1:])
    assert payload == {"job_id": "job-1", "kind": "held", "reason": "gate_failed", "text": "x"}  # dict 값은 싣지 않는다


async def test_hanging_webhook_does_not_hold_the_job(monkeypatch):
    import asyncio
    import threading
    import time

    monkeypatch.setenv(alert.WEBHOOK_ENV, "https://hooks.example.test/x")
    monkeypatch.setattr(alert, "WEBHOOK_TIMEOUT_SECONDS", 0.1)
    release = threading.Event()
    posted: list[bytes] = []

    def hang(url, body):
        posted.append(body)
        release.wait(5)
        return 200

    monkeypatch.setattr(alert, "_post", hang)
    started = time.perf_counter()
    try:
        assert await alert.JobAlert("job-2").send("held", "judge_error") is True
    finally:
        release.set()
    assert time.perf_counter() - started < 1.5 and len(posted) == 1
    other = asyncio.create_task(asyncio.sleep(0))
    await other  # 다른 작업은 그대로 돈다


async def test_webhook_error_is_swallowed(monkeypatch):
    monkeypatch.setenv(alert.WEBHOOK_ENV, "https://hooks.example.test/x")

    def boom(url, body):
        raise OSError("down")

    monkeypatch.setattr(alert, "_post", boom)
    assert await alert.JobAlert("job-3").send("failed", "timeout") is True


@pytest.mark.parametrize("value, want", [("", None), ("ftp://x", None), (" https://h.test/a ", "https://h.test/a")])
def test_webhook_url_env(monkeypatch, value, want):
    monkeypatch.setenv(alert.WEBHOOK_ENV, value)
    assert alert.webhook_url() == want


# ── 발급 규칙(2026-11-05 평가 단계 뒤 민서 결정): 사실 위반만 난 채점은 사다리를 타지 않는다 ──


async def test_fact_only_block_passes_at_once_and_templates_only_number_slots(make_section_client):
    session, client = await _session(make_section_client)
    before, n_before = dict(session.result.section_sources), len(_calls(client))
    assert before["7.2"] == "llm" and before["8.2"] == "llm"
    judge = FakeJudge(run_("blocked", ["7.2", "8.2"], issuance="pass", repair=["7.2"]))
    out = await ladder.run_ladder(session, judge, deadline=None)
    assert (out.gate, out.held_kind, len(out.record["rounds"]), len(judge.seen)) == ("passed", None, 1, 1)  # 다시 채점하지 않는다
    assert (out.record["number_templated"], out.record["templated"], out.record["regenerated"]) == (["7.2"], [], {})
    after = out.result.section_sources
    assert after["7.2"] == "fallback" and {k: v for k, v in after.items() if k != "7.2"} == {
        k: v for k, v in before.items() if k != "7.2"}
    assert session.trace["sections"]["7.2"]["reason"] == "judge_number"
    assert len(_calls(client)) == n_before  # 생성 호출 없음
    assert out.record["rounds"][0]["gate"] == "blocked" and out.record["rounds"][0]["issuance"] == "pass"
    assert ladder.summary(out.record)["number_templated"] == ["7.2"]


async def test_fact_only_block_without_number_slots_changes_nothing(make_section_client):
    session, _ = await _session(make_section_client)
    before = dict(session.result.section_sources)
    out = await ladder.run_ladder(session, FakeJudge(run_("blocked", ["8.2"], issuance="pass")), deadline=None)
    assert out.gate == "passed" and out.result.section_sources == before and out.record["number_templated"] == []
    assert out.record["ever_blocked"] is False


async def test_ever_blocked_follows_the_issuance_rule_not_the_judge_gate(make_section_client):
    # 발급 규칙이 막은 적이 있으면 그 뒤 일시적 실패 보류에 '서술 다시 만들기'를 주지 않는다(판정자 판정이 pass 여도)
    session, _ = await _session(make_section_client)
    judge = FakeJudge(run_("pass", ["8.2"], issuance="blocked"), run_(None, failure="transient"))
    out = await ladder.run_ladder(session, judge, deadline=None)
    assert (out.gate, out.held_kind, out.retry_allowed) == ("held", "judge_error", False)
    assert out.record["ever_blocked"] is True


async def test_ladder_regenerates_only_the_contradiction_slot_not_fact_violation_slots(make_section_client):
    # 판정자 게이트는 7.2(사실 위반)·8.2(모순)를 걸었지만 발급을 막는 칸은 8.2 뿐 — 사다리는 8.2 만 다시 만든다
    session, client = await _session(make_section_client)
    n_before = len(_calls(client))
    judge = FakeJudge(run_("blocked", ["7.2", "8.2"], blocking=["8.2"]), run_("pass"))
    out = await ladder.run_ladder(session, judge, deadline=None)
    assert out.gate == "passed" and len(out.record["rounds"]) == 2
    assert [requested_fields(k) for k in _calls(client)[n_before:]] == [["section_8_2"]]
