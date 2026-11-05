"""tests/test_judge_jobs.py — 판 4 서술 작업의 심판 갈래(app.narrative.jobs.run_job_ed4·create_job), 심판 라운드 J5b.

계획 14-1~14-3·정3. 생성은 가짜 Anthropic, 채점은 가짜 OpenAI(진짜 judge_once) 또는 judge_once 바꿔 끼우기 — 실호출 없음.
닫힌 쪽 실패 행렬: 실패 종류마다 done + 보류(종류) + 알림 1번, 그리고 결과·내부 기록 어디에도 gate passed 가 없음을 단정한다.
감시 시험은 호출 수를 세는 가짜 심판으로 "판 3 은 0번, 판 4 는 1번 이상"을 함께 본다(클라이언트가 None 이라 저절로 통과 방지).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from app.core import concurrency, llm_budget
from app.judge import alert, bundle, consistency, coverage, extract, faithfulness, ladder, readability
from app.judge.core import JudgeRun
from app.main import app
from app.narrative import generator, jobs
from app.narrative.schemas import JudgeBundle, NarrativeJobResult
from tests import judge_fakes as F
from tests.test_judge_core import _empty_answers, router
from tests.test_narrative_jobs import _row, _stages, factory  # noqa: F401 — 픽스처

DEV = Path(__file__).parent / "data" / "judge_dev"
CONTRACT = Path(__file__).parent / "contract"
MARKER = "@@심판표식-j5b@@"


def _ed4(name="dev_multiclass") -> dict:
    return json.loads((DEV / f"{name}.request.json").read_text(encoding="utf-8"))


def _ed3(name="dev_multiclass") -> dict:
    raw = _ed4(name)
    raw["report_edition"] = 3
    raw.pop("judge_material", None)
    return raw


def _pass_answers():
    answers = _empty_answers(None)

    def claims(kw):
        text = kw["input"][0]["content"] if isinstance(kw["input"], list) else kw["input"]
        for block in text.split("\n\n"):  # 숫자가 든 첫 서술 칸 조각의 숫자 낱말 하나를 주장으로
            head, _, body = block.partition("\n")
            tok = next((t for t in body.split() if any(ch.isdigit() for ch in t)), None)
            if head.startswith("[slot:") and tok:
                pid = head[1:].split(" | ")[0]
                return {"claims": [{"kind": "number", "quote": tok, "piece_id": pid, "field": "", "field_b": "",
                                    "number_text": "", "polarity": ""}]}
        return {"claims": []}  # 서술에 숫자가 없으면 주장 0개(no_claims 안전장치는 숫자가 있을 때만)

    answers["claims"] = claims
    answers["faithfulness"] = {"claims": [{"claim_id": "C1", "resolution": "no_data", "expr": "", "expr_b": "",
                                           "relation": "", "name_field": "", "named": "", "metric_id": "",
                                           "inference_basis": ""}]}
    return answers


class Alerts:
    def __init__(self, monkeypatch):
        self.sent: list[tuple] = []
        outer = self

        async def send(self_, kind, reason, **extra):
            if self_.sent:
                return False
            self_.sent = True
            outer.sent.append((self_.job_id, kind, reason))
            return True

        monkeypatch.setattr(alert.JobAlert, "send", send)


@pytest.fixture
def alerts(monkeypatch):
    return Alerts(monkeypatch)


async def _create(factory, raw, *, client=None, judge_client=None):
    tasks = BackgroundTasks()
    created = await jobs.create_job(raw, client=client, session_factory=factory, background_tasks=tasks,
                                    judge_client=judge_client)
    for task in tasks.tasks:
        await task()
    return created


def _internal(factory, job_id) -> dict:
    return json.loads(_row(factory, job_id).internal_json or "{}")


async def _read(factory, job_id) -> dict:
    return (await jobs.read_job(job_id, session_factory=factory)).model_dump(mode="json")


def _no_passed_anywhere(factory, job_id):
    row = _row(factory, job_id)
    assert '"passed"' not in (row.result_json or "") and '"passed"' not in (row.internal_json or "")


def _fake_judge(monkeypatch, *runs, sleep=0.0):
    seen = []

    async def fake(client, req, result, sources, refs, *, deadline=None, clock=None):
        seen.append(dict(sources))
        if sleep:
            await asyncio.sleep(sleep)
        item = runs[min(len(seen), len(runs)) - 1]
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(jobs, "judge_once", fake)
    return seen


CARD = {"faithfulness": {"checked": 3, "matched": 3, "violations": 0, "inference_ok": 0, "inference_violations": 0,
                         "unverifiable": 0, "target_unclear": 0, "client_input_assertive": 0},
        "consistency": {"checked": 5, "contradictions": 0, "by_type": {}},
        "coverage": {"applicable": 1, "described": 1, "not_applicable": 17,
                     "items": [{"id": "분석-1", "described": True, "place": "5.2"}]},
        "readability": {"rules": 2, "violations": 0, "llm_flags": 0}}


# ── 감시: 판 3 은 심판 0번, 판 4 는 1번 이상(진짜 judge_once + 세는 가짜 OpenAI) ──


async def test_edition4_passes_through_the_real_judge_and_edition3_never_calls_it(factory, make_section_client, alerts):
    oa = F.FakeOpenAI([router(_pass_answers())] * 40)
    an = make_section_client()
    c3 = await _create(factory, _ed3(), client=an, judge_client=oa)
    assert oa.calls == []
    r3 = await _read(factory, c3.job_id)
    assert "judge" not in r3["result"] and "judge" not in _internal(factory, c3.job_id)
    assert "report_edition" not in _internal(factory, c3.job_id)  # 판 3 내부 기록은 전과 같은 키

    c4 = await _create(factory, _ed4(), client=an, judge_client=oa)
    assert len(oa.calls) >= 5
    r4 = await _read(factory, c4.job_id)
    j = r4["result"]["judge"]
    assert (j["gate"], j["held_kind"], j["retry_allowed"]) == ("passed", None, False)
    assert list(j) == ["gate", "held_kind", "retry_allowed", "criteria_version", "model", "axes", "counts"]
    # 응답에는 싣는 축만 — 뺀 축(필수 내용·표기)은 'null' 도 아니고 키째 없다(JudgeCounts 직렬화)
    assert list(j["counts"]) == list(bundle.PUBLISHED_AXES) == ["faithfulness", "consistency"]
    assert j["axes"] == ["faithfulness", "consistency"]
    internal = _internal(factory, c4.job_id)
    assert internal["judge"]["gate"] == "passed" and internal["judge"]["summary"]["rounds"] == 1
    assert "judging" in _stages(factory, c4.job_id)
    assert "judging" not in _stages(factory, c3.job_id)
    assert alerts.sent == []


# ── 만들 때 정해지는 보류(정3) ──


async def test_no_judge_key_holds_without_any_ai_call(factory, make_section_client, alerts):
    an = make_section_client()
    c = await _create(factory, _ed4(), client=an, judge_client=None)
    assert an.beta.messages.create.call_count == 0
    j = (await _read(factory, c.job_id))["result"]["judge"]
    assert (j["gate"], j["held_kind"], j["retry_allowed"], j["counts"]) == ("held", "no_key", False, None)
    internal = _internal(factory, c.job_id)
    assert (internal["budget"], internal["judge_budget"], internal["reason"]) == ("not_needed", "no_key", "judge_no_key")
    assert alerts.sent == [(c.job_id, "held", "no_key")]
    _no_passed_anywhere(factory, c.job_id)


async def test_judge_wallet_exhausted_holds_with_retry_and_spares_the_narrative_wallet(factory, make_section_client, alerts, monkeypatch):
    monkeypatch.setattr(llm_budget.judge, "limit", 0)
    an = make_section_client()
    before = llm_budget.narrative.remaining()
    c = await _create(factory, _ed4(), client=an, judge_client=F.FakeOpenAI([]))
    assert an.beta.messages.create.call_count == 0 and llm_budget.narrative.remaining() == before
    j = (await _read(factory, c.job_id))["result"]["judge"]
    assert (j["held_kind"], j["retry_allowed"]) == ("budget", True)
    assert alerts.sent == [(c.job_id, "held", "budget")]


async def test_judge_wallet_is_reserved_only_for_edition4(factory, make_section_client, monkeypatch):
    _fake_judge(monkeypatch, JudgeRun("pass", None, [], [], CARD, {}))
    before = llm_budget.judge.remaining()
    await _create(factory, _ed3(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    assert llm_budget.judge.remaining() == before
    await _create(factory, _ed4(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    assert llm_budget.judge.remaining() == before - 1


async def test_job_limit_edition4_result_is_held(factory, make_section_client, alerts, monkeypatch):
    monkeypatch.setattr(llm_budget.narrative_jobs, "limit", 0)
    c = await _create(factory, _ed4(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    assert c.job_id is None and c.result.judge.gate == "held" and c.result.judge.held_kind == "job_limit"
    assert alerts.sent == [(None, "held", "job_limit")]
    c3 = await _create(factory, _ed3(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    assert c3.result.judge is None and "judge" not in c3.result.model_dump(mode="json")


# ── 닫힌 쪽 실패 행렬 ──


class Unserializable:
    pass


@pytest.mark.parametrize("run, kind, retry", [
    (JudgeRun(None, "transient"), "judge_error", True),
    (JudgeRun(None, "timeout"), "judge_error", True),
    (JudgeRun(None, "api_error"), "judge_error", False),
    (JudgeRun(None, "refusal"), "judge_error", False),
    (JudgeRun(None, "bad_output"), "judge_error", False),
    (JudgeRun(None, "safety"), "judge_error", False),
    (JudgeRun(None, "no_material"), "no_material", False),
    (JudgeRun(None, "deadline"), "timeout", True),
    (RuntimeError("심판 안에서 터짐"), "judge_error", False),
    (JudgeRun("pass", None, [], [], CARD, {"weird": Unserializable()}), "judge_error", False),  # 직렬화 안 되는 기록
    (JudgeRun("blocked", None, [], [{"type": "value_mismatch"}], CARD, {}), "gate_failed", False),
])
async def test_failure_matrix_is_done_and_held(factory, make_section_client, alerts, monkeypatch, run, kind, retry):
    _fake_judge(monkeypatch, run)
    c = await _create(factory, _ed4(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    view = await _read(factory, c.job_id)
    assert view["status"] == "done" and view["result"] is not None
    j = view["result"]["judge"]
    assert (j["gate"], j["held_kind"], j["retry_allowed"], j["counts"]) == ("held", kind, retry, None)
    assert _internal(factory, c.job_id)["judge"]["gate"] == "held"
    assert alerts.sent == [(c.job_id, "held", kind)]
    _no_passed_anywhere(factory, c.job_id)


async def test_transient_http_errors_are_tried_twice_then_held_with_retry(factory, make_section_client, alerts):
    oa = F.FakeOpenAI([F.server_error()] * 40)
    c = await _create(factory, _ed4(), client=make_section_client(), judge_client=oa)
    j = (await _read(factory, c.job_id))["result"]["judge"]
    assert (j["held_kind"], j["retry_allowed"]) == ("judge_error", True)
    rnd = _internal(factory, c.job_id)["judge"]["rounds"][0]["record"]
    attempts = [call["attempts"] for m in ("extract", "coverage", "readability") for call in (rnd.get(m) or {}).get("calls", [])]
    assert attempts and all(a == 2 for a in attempts)  # 같은 호출 한 번 더(SDK 재시도 0)
    assert len(oa.calls) == sum(attempts)


async def test_unknown_gate_value_is_held(factory, make_section_client, alerts, monkeypatch):
    _fake_judge(monkeypatch, JudgeRun("PASS", None, [], [], CARD, {}))
    c = await _create(factory, _ed4(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    assert (await _read(factory, c.job_id))["result"]["judge"]["gate"] == "held"
    _no_passed_anywhere(factory, c.job_id)


# ── 시간 ──


async def test_outer_timeout_after_the_narrative_is_held_timeout(factory, make_section_client, alerts, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DEADLINE_SECONDS", 0.5)
    monkeypatch.setattr(ladder, "JUDGE_MIN_SECONDS", 0.0)
    _fake_judge(monkeypatch, JudgeRun("pass", None, [], [], CARD, {}), sleep=5)
    c = await _create(factory, _ed4(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    view = await _read(factory, c.job_id)
    j = view["result"]["judge"]
    assert view["status"] == "done" and (j["held_kind"], j["retry_allowed"]) == ("timeout", True)
    assert alerts.sent == [(c.job_id, "held", "timeout")]


async def test_outer_timeout_before_the_narrative_is_failed(factory, make_section_client, alerts, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DEADLINE_SECONDS", 0.3)
    c = await _create(factory, _ed4(), client=make_section_client(delay=5), judge_client=F.FakeOpenAI([]))
    view = await _read(factory, c.job_id)
    assert (view["status"], view["failure"], view["result"]) == ("failed", "timeout", None)
    assert alerts.sent == [(c.job_id, "failed", "timeout")]


async def test_edition4_generation_gets_its_own_cap_and_edition3_does_not(factory, make_section_client, monkeypatch):
    _fake_judge(monkeypatch, JudgeRun("pass", None, [], [], CARD, {}))
    seen = []
    real = generator.GenerationSession.__init__

    def spy(self, *a, **kw):
        seen.append(kw.get("deadline_seconds"))
        real(self, *a, **kw)

    monkeypatch.setattr(generator.GenerationSession, "__init__", spy)
    await _create(factory, _ed3(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    await _create(factory, _ed4(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    assert seen == [jobs.JOB_GENERATION_SECONDS, jobs.ED4_GENERATION_SECONDS]


async def test_judging_refreshes_the_stage_while_it_runs(factory, make_section_client, monkeypatch):
    monkeypatch.setattr(jobs, "JUDGE_HEARTBEAT_SECONDS", 0.05)
    _fake_judge(monkeypatch, JudgeRun("pass", None, [], [], CARD, {}), sleep=0.3)
    c = await _create(factory, _ed4(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    assert _stages(factory, c.job_id).count("judging") >= 4


async def test_busy_judge_slot_is_held_with_retry(factory, make_section_client, alerts, monkeypatch):
    monkeypatch.setattr(jobs, "JUDGE_SLOT_WAIT_SECONDS", 0.1)
    seen = _fake_judge(monkeypatch, JudgeRun("pass", None, [], [], CARD, {}))
    release = asyncio.Event()

    async def hog():
        async with contextlib.AsyncExitStack() as stack:
            for _ in range(concurrency.MAX_CONCURRENT_JUDGE_RUNS):
                await stack.enter_async_context(concurrency.judge_slot())
            await release.wait()

    holder = asyncio.create_task(hog())
    await asyncio.sleep(0.01)
    try:
        c = await _create(factory, _ed4(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    finally:
        release.set()
        await holder
    j = (await _read(factory, c.job_id))["result"]["judge"]
    assert (j["held_kind"], j["retry_allowed"], seen) == ("judge_error", True, [])


def test_time_invariants():
    """심판 호출 제한·슬롯 대기·단계 갱신 간격이 유실 판정 기준보다 짧고, 판 4 생성이 상한을 다 써도 첫 채점 시간이 남는다."""
    timeouts = [m.CALL_TIMEOUT_SECONDS for m in (extract, faithfulness, consistency, coverage, readability)]
    assert max(timeouts) < jobs.LOST_IDLE_SECONDS
    assert jobs.JUDGE_SLOT_WAIT_SECONDS < jobs.LOST_IDLE_SECONDS
    assert jobs.JUDGE_HEARTBEAT_SECONDS < jobs.LOST_IDLE_SECONDS
    assert jobs.JOB_GENERATION_SECONDS - jobs.ED4_GENERATION_SECONDS >= ladder.JUDGE_MIN_SECONDS


# ── 고객 응답에 심판 내부가 새지 않는다 ──


async def test_customer_view_has_no_judge_internals(factory, make_section_client, monkeypatch):
    card = {**CARD, "faithfulness": {**CARD["faithfulness"], "note": MARKER}}
    _fake_judge(monkeypatch, JudgeRun("pass", None, [], [{"quote": MARKER}], card, {"reason": MARKER}))
    c = await _create(factory, _ed4(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    view = await _read(factory, c.job_id)
    assert MARKER not in json.dumps(view, ensure_ascii=False)
    assert MARKER in (_row(factory, c.job_id).internal_json or "")  # 내부 기록에는 남는다(양성 대조)
    assert set(view["result"]) == {"interpretation", "conclusion", "recommendation_narrative", "recommendations",
                                   "section_sources", "section_models", "judge"}


# ── 알림: 판 3 생성 실패·유실 ──


async def test_edition3_failure_alerts_once(factory, make_section_client, alerts, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DEADLINE_SECONDS", 0.3)
    c = await _create(factory, _ed3(), client=make_section_client(delay=5))
    assert alerts.sent == [(c.job_id, "failed", "timeout")]


async def test_lost_judgement_alerts_once(factory, alerts):
    from datetime import timedelta

    from app.narrative.models import _utcnow

    old = _utcnow() - timedelta(hours=1)
    jobs._db_insert(factory, "job-lost", [], {}, old)
    await _read(factory, "job-lost")
    await _read(factory, "job-lost")
    assert alerts.sent == [("job-lost", "lost", "queued")]


# ── 묶음 모양 ──


def test_contract_examples_validate_and_bundle_matches_the_passed_example_shape():
    contract = json.loads((CONTRACT / "judge_bundle_ed4.json").read_text(encoding="utf-8"))
    for ex in contract["examples"].values():
        JudgeBundle.model_validate(ex)
    built = bundle.passed(CARD)
    example = contract["examples"]["passed"]
    assert list(built) == list(example)
    # 싣는 축(bundle.PUBLISHED_AXES)만 — 예시는 네 축을 다 싣던 때의 모양이고, 축을 빼면 그 축만 빠진다(계약 axes)
    assert list(built["counts"]) == list(bundle.PUBLISHED_AXES) == ["faithfulness", "consistency"]
    assert {a: set(v) for a, v in built["counts"].items()} == {a: set(example["counts"][a]) for a in built["counts"]}
    assert built["axes"] == ["faithfulness", "consistency"]
    JudgeBundle.model_validate(built)  # 뺀 축이 없는 counts 도 계약 모양
    assert list(bundle.held("budget", retry_allowed=True)) == list(contract["examples"]["held"])


def test_edition3_result_dump_has_no_judge_key():
    res = NarrativeJobResult.model_validate(json.loads(json.dumps({
        "interpretation": {"confusion_analysis": "", "distribution_analysis": ""},
        "conclusion": {"verdict": "PASS", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "section_sources": {}, "section_models": {},
    })))
    assert "judge" not in res.model_dump(mode="json") and "judge" not in json.loads(res.model_dump_json())


def test_route_passes_the_openai_client_as_judge_client(factory, monkeypatch):
    seen = {}

    async def fake_create(payload, **kw):
        seen.update(kw)
        raise RuntimeError("멈춤")

    monkeypatch.setattr(jobs, "create_job", fake_create)
    marker = object()
    app.state.openai_client = marker
    app.dependency_overrides[jobs.get_session_factory] = lambda: factory
    try:
        with pytest.raises(RuntimeError):
            TestClient(app).post("/api/narrative/jobs", json=_ed4())
    finally:
        app.dependency_overrides.pop(jobs.get_session_factory, None)
    assert seen["judge_client"] is marker


def test_job_deadline_moves_with_the_front_texts():
    """백엔드 마감 10분은 프론트 포기 시간(useReportData giveUpMs 12분)·'제한 시간(10분)' 글(ReportStateNotices, 판 3 보관본
    정답에도 있음)과 함께 움직인다 — 바꾸려면 세 곳을 같이 고친다(계획 14 "10분 안의 숫자", J5b 끝 확정: 그대로 600)."""
    assert jobs.JOB_DEADLINE_SECONDS == 600.0
    assert (jobs.ED4_GENERATION_SECONDS, ladder.JUDGE_MIN_SECONDS, ladder.REGENERATE_MIN_SECONDS) == (240.0, 160.0, 60.0)


async def test_pilot_timeline_reaches_the_rule_sentence_rung():
    """실측(생성 60초·채점 150초·재생성 45초)으로 늘 걸리는 경우: 채점 1 → 재생성 → 채점 2 → 걸린 칸 규칙 문장 → 채점 3 → 시간 보류.
    (JUDGE_ROUND/실측/J5b_시간.md 표의 한 줄 — 숫자를 바꾸면 이 시험과 표를 함께 고친다.)"""

    class Clock:
        t = 60.0

        def __call__(self):
            return self.t

    clock = Clock()

    class Sess:
        result = object()
        states = dict.fromkeys(("7.1", "7.2", "8.1", "8.2", "8.3", "9"))

        def __init__(self):
            self.ai = set(self.states)

        def has_ai_text(self, s):
            return s in self.ai

        async def judge_regenerate(self, sids, *, seconds):
            clock.t += 45
            return dict.fromkeys(sids, "regenerated")

        def to_template(self, sids, reason):
            changed = [s for s in sids if s in self.ai]
            self.ai -= set(changed)
            return changed

    async def judge(res, dl):
        clock.t += 150
        return JudgeRun("blocked", None, ["8.2"], [], None, {})

    out = await ladder.run_ladder(Sess(), judge, deadline=jobs.JOB_GENERATION_SECONDS, clock=clock)
    assert [r.get("after") for r in out.record["rounds"] if "gate" in r] == [None, "regenerate", "template_flagged"]
    assert (out.held_kind, out.retry_allowed, clock.t) == ("timeout", False, 555)


# ── J10 검토 ⑥ 검증 기록 잠금을 만들 때도 본다(P1-8 가) — 두 겹을 각각 ──


def _lock_calls(monkeypatch, answers):
    """criteria_lock.problem 을 부를 때마다 answers 를 차례로(끝나면 마지막 값) — 한 겹만 잠기게."""
    from app.judge import criteria_lock

    seen = []

    def problem(*a, **k):
        seen.append(1)
        return answers[min(len(seen), len(answers)) - 1]

    monkeypatch.setattr(criteria_lock, "problem", problem)
    return seen


async def test_lock_at_create_spends_no_ai_and_no_judge_wallet(factory, make_section_client, alerts, monkeypatch):
    """만들 때 겹만 잠김(심판 갈래 겹은 통과라고 답함) — 서술 AI·심판·두 지갑 모두 쓰지 않고 규칙 문장 + 보류(judge_error)."""
    calls = _lock_calls(monkeypatch, ["unvalidated", None])
    seen = _fake_judge(monkeypatch, JudgeRun("pass", None, [], [], CARD, {}))
    an = make_section_client()
    judge_before, narr_before = llm_budget.judge.remaining(), llm_budget.narrative.remaining()
    c = await _create(factory, _ed4(), client=an, judge_client=F.FakeOpenAI([]))
    assert calls == [1]  # 만들 때 한 번만 — 갈래 겹은 부르지 않았다(locked 길)
    assert an.beta.messages.create.call_count == 0 and seen == []
    assert (llm_budget.judge.remaining(), llm_budget.narrative.remaining()) == (judge_before, narr_before)
    j = (await _read(factory, c.job_id))["result"]["judge"]
    assert (j["gate"], j["held_kind"], j["retry_allowed"], j["counts"]) == ("held", "judge_error", False, None)
    internal = _internal(factory, c.job_id)
    assert (internal["budget"], internal["judge_budget"], internal["reason"]) == ("not_needed", "locked", "judge_locked")
    rec = internal["judge"]
    assert (rec["gate"], rec["held_kind"], rec["failure"], rec["criteria_lock"]) == ("held", "judge_error", "criteria_lock",
                                                                                     "unvalidated")
    assert alerts.sent == [(c.job_id, "held", "judge_error")]
    _no_passed_anywhere(factory, c.job_id)


async def test_lock_in_judge_branch_alone_still_holds(factory, make_section_client, alerts, monkeypatch):
    """만들 때 겹은 통과(None), 심판 갈래 겹만 잠김 — 그래도 심판을 부르지 않고 보류. 두 겹이 서로를 가리지 않는다."""
    calls = _lock_calls(monkeypatch, [None, "fingerprint_mismatch"])
    seen = _fake_judge(monkeypatch, JudgeRun("pass", None, [], [], CARD, {}))
    an = make_section_client()
    c = await _create(factory, _ed4(), client=an, judge_client=F.FakeOpenAI([]))
    assert len(calls) == 2 and seen == []
    assert an.beta.messages.create.call_count > 0  # 만들 때 겹을 지났으니 서술 AI 는 불렀다
    rec = _internal(factory, c.job_id)["judge"]
    assert (rec["gate"], rec["held_kind"], rec["criteria_lock"]) == ("held", "judge_error", "fingerprint_mismatch")
    _no_passed_anywhere(factory, c.job_id)


async def test_locked_result_has_the_same_shape_as_no_key(factory, make_section_client, alerts, monkeypatch):
    """고객이 받는 결과 — 잠긴 작업과 심판 키 없는 작업은 같은 규칙 문장 서술, 같은 묶음 꼴(보류 종류만 다름)."""
    no_key = await _create(factory, _ed4(), client=make_section_client(), judge_client=None)
    _lock_calls(monkeypatch, ["unvalidated"])
    locked = await _create(factory, _ed4(), client=make_section_client(), judge_client=F.FakeOpenAI([]))
    a, b = (await _read(factory, no_key.job_id))["result"], (await _read(factory, locked.job_id))["result"]
    ja, jb = a.pop("judge"), b.pop("judge")
    assert a == b  # 서술 본문·칸별 출처·칸별 모델이 같다(둘 다 규칙 문장)
    assert set(a["section_sources"].values()) <= {"fallback", "fixed"}
    assert list(ja) == list(jb) and {k: v for k, v in ja.items() if k != "held_kind"} == \
        {k: v for k, v in jb.items() if k != "held_kind"}
    assert (ja["held_kind"], jb["held_kind"]) == ("no_key", "judge_error")
