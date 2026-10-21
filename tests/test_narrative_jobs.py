"""test_narrative_jobs.py — 서술 작업 번호(job) 방식 (서술 라운드 S3a, app.narrative.jobs).

DB 는 테스트마다 인메모리 SQLite(StaticPool)다. 경로 테스트는 세션 팩토리 의존성을 그것으로 바꿔 끼우고,
run_job 은 직접 await 해 단계·기록을 본다. TestClient 는 BackgroundTasks 를 응답 전에 끝까지 돌리므로
POST 직후 GET 은 이미 끝 상태다.
"""
import asyncio
import contextlib
import json
import os
import subprocess
import sys
from datetime import timedelta

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core import concurrency, llm_budget
from app.core.database import Base, configure_sqlite
from app.core.schemas import TaskType
from app.main import app
from app.narrative import jobs, schemas
from app.narrative.models import NarrativeJob, _utcnow
from app.narrative.schemas import NarrativeRequest
from app.narrative.legacy import generate_narrative
from tests.conftest import fake_beta_message
from tests.test_narrative_router import _sample_request
from tests.test_narrator import _no_special_fact_sheet

_CLEAN = {
    "interpretation": {"confusion_analysis": "정확도 0.94로 양호.", "distribution_analysis": ""},
    "conclusion": {"verdict": "PASS", "benchmark": "", "narrative": "", "risks": ""},
    "recommendation_narrative": {"data_quality": "", "model_ops": ""},
}
_SECRET = "고객비밀입력-7731"


@pytest.fixture
def factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    configure_sqlite(engine)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, autocommit=False)
    engine.dispose()


@pytest.fixture
def api(factory):
    app.dependency_overrides[jobs.get_session_factory] = lambda: factory
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(jobs.get_session_factory, None)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("NARRATIVE_MODEL", "NARRATIVE_EFFORT", "NARRATIVE_RESULT_TTL_HOURS", "NARRATIVE_JOB_RETENTION_DAYS"):
        monkeypatch.delenv(key, raising=False)


def _request(**kw) -> NarrativeRequest:
    return NarrativeRequest(
        task_type=TaskType.binary, report_purpose="external", fact_sheet=_no_special_fact_sheet(), **kw
    )


def _body(req: NarrativeRequest | None = None) -> dict:
    return json.loads((req or _sample_request()).model_dump_json())


def _row(factory, job_id) -> NarrativeJob:
    with factory() as db:
        job = db.get(NarrativeJob, job_id)
        db.expunge(job)
        return job


def _internal(factory, job_id) -> dict:
    return json.loads(_row(factory, job_id).internal_json)


async def _create(factory, *, client=None, req=None, raw=None) -> tuple[str, BackgroundTasks]:
    """create_job 을 부르고 (번호, 미뤄 둔 작업)을 돌려준다 — 미뤄 둔 작업은 테스트가 직접 돌린다."""
    tasks = BackgroundTasks()
    created = await jobs.create_job(
        raw if raw is not None else _body(req or _request()),
        client=client, session_factory=factory, background_tasks=tasks,
    )
    return created.job_id, tasks


async def _run_pending(tasks: BackgroundTasks) -> None:
    for task in tasks.tasks:
        await task.func(*task.args, **task.kwargs)


def _stages(factory, job_id) -> list[str]:
    return [s["stage"] for s in _internal(factory, job_id)["stages"]]


# ── 순수 함수: 유실·보관 판정 ───────────────────────────────────────────────────

def test_lost_after_idle_strictly_longer_than_limit():
    t0 = _utcnow()
    limit = timedelta(seconds=jobs.LOST_IDLE_SECONDS)
    assert not jobs.is_lost(created_at=t0, updated_at=t0, now=t0 + limit)
    assert jobs.is_lost(created_at=t0, updated_at=t0, now=t0 + limit + timedelta(seconds=1))


def test_lost_after_total_deadline_even_if_recently_updated():
    t0 = _utcnow()
    limit = timedelta(seconds=jobs.JOB_DEADLINE_SECONDS + jobs.LOST_GRACE_SECONDS)
    fresh = t0 + limit  # 방금 갱신됨
    assert not jobs.is_lost(created_at=t0, updated_at=fresh, now=t0 + limit)
    assert jobs.is_lost(created_at=t0, updated_at=fresh, now=t0 + limit + timedelta(seconds=1))


def test_lost_limits_cover_the_longest_quiet_stretch():
    """가장 긴 무변화 구간(호출 1회, 슬롯 대기)보다 유실 판정이 길어야 정상 작업을 유실로 끊지 않는다."""
    assert jobs.LOST_IDLE_SECONDS > jobs.JOB_CALL_TIMEOUT_SECONDS
    assert jobs.LOST_IDLE_SECONDS > jobs.JOB_SLOT_WAIT_SECONDS
    assert jobs.JOB_SLOT_WAIT_SECONDS + jobs.JOB_CALL_TIMEOUT_SECONDS < jobs.JOB_DEADLINE_SECONDS
    assert jobs.JOB_DEADLINE_SECONDS == 600.0  # 사용자 결정: 품질 우선, 10분까지


def test_expired_strictly_after_ttl(monkeypatch):
    t0 = _utcnow()
    assert jobs.result_ttl() == timedelta(hours=24)
    assert not jobs.is_expired(finished_at=t0, now=t0 + timedelta(hours=24))
    assert jobs.is_expired(finished_at=t0, now=t0 + timedelta(hours=24, seconds=1))
    assert not jobs.is_expired(finished_at=None, now=t0 + timedelta(days=999))
    monkeypatch.setenv("NARRATIVE_RESULT_TTL_HOURS", "2")
    assert jobs.is_expired(finished_at=t0, now=t0 + timedelta(hours=2, seconds=1))


@pytest.mark.parametrize("value", ["", "abc", "0", "-3"])
def test_retention_env_falls_back_to_default(monkeypatch, value):
    monkeypatch.setenv("NARRATIVE_JOB_RETENTION_DAYS", value)
    monkeypatch.setenv("NARRATIVE_RESULT_TTL_HOURS", value)
    assert jobs.record_retention() == timedelta(days=90)
    assert jobs.result_ttl() == timedelta(hours=24)


def test_section_sources_are_the_seven_sections_with_7_3_fixed():
    # 소절 7칸(S4b). 7.3(추가 관찰)은 AI 가 쓰기 전(S5 이전)에는 늘 고정 문구 15 '해당 없음'이다.
    assert jobs.section_sources("llm") == {
        "7.1": "llm", "7.2": "llm", "7.3": "fixed", "8.1": "llm", "8.2": "llm", "8.3": "llm", "9": "llm",
    }
    assert jobs.section_sources("fallback") == {
        "7.1": "fallback", "7.2": "fallback", "7.3": "fixed",
        "8.1": "fallback", "8.2": "fallback", "8.3": "fallback", "9": "fallback",
    }
    assert jobs.section_sources("llm", template_recommendations=True)["9"] == "fallback"
    assert list(jobs.section_sources("llm")) == ["7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "9"]


def test_total_cost_refuses_to_guess():
    assert jobs.total_cost([]) == 0
    assert jobs.total_cost([{"cost_usd": 0.01}, {"cost_usd": 0.02}]) == pytest.approx(0.03)
    assert jobs.total_cost([{"cost_usd": 0.01}, {"cost_usd": None}]) is None


def test_iso_utc():
    from datetime import datetime

    assert jobs.iso_utc(datetime(2026, 10, 19, 1, 2, 3, 999999)) == "2026-10-19T01:02:03Z"


# ── 입력 상한 → 줄여서 규칙 문장으로 ────────────────────────────────────────────

def test_parse_request_valid_is_untouched():
    req, trimmed = jobs.parse_request(_body())
    assert trimmed == []
    assert req == _sample_request()


def _over(path_value: tuple[tuple, object]) -> dict:
    body = _body()
    (path, value) = path_value
    target = body
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return body


N = schemas.MAX_CLASS_ENTRIES + 1


@pytest.mark.parametrize("path, value, changed, check", [
    (("fact_sheet", "per_class"), [{"label": str(i)} for i in range(N)], "fact_sheet.per_class",
     lambda r: r.fact_sheet.per_class == []),
    (("fact_sheet", "confusion_counts"), [{"label": str(i), "tp": 1, "fp": 0, "fn": 0, "tn": 1} for i in range(N)],
     "fact_sheet.confusion_counts", lambda r: r.fact_sheet.confusion_counts == []),
    (("fact_sheet", "confusion"), {"labels": [str(i) for i in range(N)], "matrix": [[0]]}, "fact_sheet.confusion",
     lambda r: r.fact_sheet.confusion is None),
    (("fact_sheet", "confusion"), {"labels": ["0", "1"], "matrix": [[0] * N, [0] * N]}, "fact_sheet.confusion",
     lambda r: r.fact_sheet.confusion is None),
    (("fact_sheet", "distribution"), {"class_distribution": {str(i): 1 for i in range(N)}, "imbalance_ratio": 1.0},
     "fact_sheet.distribution", lambda r: r.fact_sheet.distribution is None),
    (("fact_sheet", "metrics"),
     [{"metric_id": f"M{i}", "display_name": "x", "value": 0.5, "status": "pass"} for i in range(schemas.MAX_METRIC_FACTS + 1)],
     "fact_sheet.metrics", lambda r: len(r.fact_sheet.metrics) == schemas.MAX_METRIC_FACTS),
    (("fact_sheet", "ci_method_note"), "가" * (schemas.MAX_CI_METHOD_NOTE_CHARS + 5), "fact_sheet.ci_method_note",
     lambda r: r.fact_sheet.ci_method_note == "가" * schemas.MAX_CI_METHOD_NOTE_CHARS),
    (("client_inputs",), {"memo": "x" * (schemas.MAX_CLIENT_INPUTS_CHARS + 1)}, "client_inputs",
     lambda r: r.client_inputs is None),
    (("fact_sheet", "derived_prediction"),
     {"method": "threshold_per_label", "threshold": {str(i): 0.5 for i in range(N)}},
     "fact_sheet.derived_prediction.threshold", lambda r: r.fact_sheet.derived_prediction.threshold is None),
])
def test_parse_request_trims_limit_errors(path, value, changed, check):
    req, trimmed = jobs.parse_request(_over((path, value)))
    assert trimmed == [changed]
    assert check(req)


def _problem(i, **kw):
    return {"id": f"P{i}", "kind": "core_fail", "priority": "HIGH", "title": f"문제 {i}", **kw}


def test_parse_request_trims_problem_text_evidence_and_count():
    body = _body()
    body["fact_sheet"]["problems"] = [_problem(i) for i in range(1, schemas.MAX_PROBLEMS + 3)]
    body["fact_sheet"]["problems"][0]["title"] = "가" * (schemas.MAX_PROBLEM_TEXT_CHARS + 1)
    body["fact_sheet"]["problems"][1]["evidence"] = {"x": "y" * schemas.MAX_PROBLEM_EVIDENCE_CHARS}
    req, trimmed = jobs.parse_request(body)
    assert trimmed == ["fact_sheet.problems", "fact_sheet.problems.0.title", "fact_sheet.problems.1.evidence"]
    assert [p.id for p in req.fact_sheet.problems] == [f"P{i}" for i in range(1, schemas.MAX_PROBLEMS + 1)]
    assert req.fact_sheet.problems[0].title == "가" * schemas.MAX_PROBLEM_TEXT_CHARS
    assert req.fact_sheet.problems[1].evidence == {}


def test_parse_request_keeps_422_when_any_error_is_not_a_limit():
    body = _over((("fact_sheet", "per_class"), [{"label": str(i)} for i in range(N)]))
    del body["fact_sheet"]["verdict"]  # 상한이 아닌 오류가 섞임
    with pytest.raises(ValidationError):
        jobs.parse_request(body)
    dup = _body()
    dup["fact_sheet"]["problems"] = [_problem(1), _problem(1)]  # 번호 겹침은 ValueError 지만 상한이 아니다
    with pytest.raises(ValidationError):
        jobs.parse_request(dup)


@pytest.mark.parametrize("path, value", [
    (("fact_sheet", "confusion"), {"labels": ["0", "1"], "matrix": [["x", 1], [0, 1]]}),  # 칸 안의 형식 오류
    (("fact_sheet", "per_class"), [{"label": "0", "support": "많음"}]),
    (("fact_sheet", "metrics"), [{"metric_id": "M1", "display_name": "x", "value": "높음", "status": "pass"}]),
])
def test_parse_request_does_not_clear_fields_for_non_limit_errors(path, value):
    """클래스 단위 칸이라도 상한이 아닌 형식 오류는 비우지 않고 422 다 — 조립 버그를 규칙 문장으로 숨기지 않는다."""
    with pytest.raises(ValidationError):
        jobs.parse_request(_over((path, value)))


def test_parse_request_does_not_mutate_the_input():
    body = _over((("fact_sheet", "per_class"), [{"label": str(i)} for i in range(N)]))
    jobs.parse_request(body)
    assert len(body["fact_sheet"]["per_class"]) == N


def test_limit_validators_raise_limit_exceeded_but_old_route_still_422():
    """상한 검사가 LimitExceeded(=ValueError) 라서 옛 경로는 종전처럼 422 다."""
    assert issubclass(schemas.LimitExceeded, ValueError)
    body = _over((("fact_sheet", "distribution"), {"class_distribution": {str(i): 1 for i in range(N)}}))
    assert TestClient(app).post("/api/generate-narrative", json=body).status_code == 422


# ── run_job: 단계와 끝 상태 ──────────────────────────────────────────────────────

async def test_no_key_job_finishes_with_rule_sentences(factory):
    job_id, tasks = await _create(factory)
    assert _row(factory, job_id).stage == "queued"
    await _run_pending(tasks)
    view = await jobs.read_job(job_id, session_factory=factory)
    assert (view.status, view.stage, view.failure) == ("done", "done", None)
    assert view.result.section_sources == jobs.section_sources("fallback")  # 7.3 만 고정 문구
    assert _stages(factory, job_id) == ["queued", "done"]
    assert _row(factory, job_id).finished_at is not None
    internal = _internal(factory, job_id)
    assert (internal["source"], internal["reason"], internal["budget"]) == ("fallback", "no_key", "no_key")
    assert internal["http_attempts"] == 0 and internal["calls"] == []


async def test_llm_job_goes_through_all_stages_and_records_internally(factory, make_fake_anthropic_client):
    fake = make_fake_anthropic_client(_CLEAN, usage=(1000, 500))
    job_id, tasks = await _create(factory, client=fake, req=_request(client_inputs={"memo": _SECRET}))
    await _run_pending(tasks)
    assert _stages(factory, job_id) == ["queued", "generating", "checking", "done"]
    view = await jobs.read_job(job_id, session_factory=factory)
    assert view.status == "done"
    assert view.result.section_sources == jobs.section_sources("llm")
    assert view.result.interpretation.confusion_analysis == "정확도 0.94로 양호."
    internal = _internal(factory, job_id)
    assert (internal["source"], internal["reason"], internal["budget"]) == ("llm", None, "reserved")
    assert internal["http_attempts"] == 1
    call = internal["calls"][0]
    assert call["response_model"] == "claude-sonnet-5-5"
    assert (call["input_tokens"], call["output_tokens"]) == (1000, 500)
    assert internal["cost_usd"] == pytest.approx(0.007)
    assert internal["slot_wait_seconds"] is not None and internal["prompt_chars"] > 0
    assert internal["failure"] is None
    assert internal["seconds_total"] > 0 and internal["task_type"] == "binary"
    assert internal["recommendations_from_template"] == []
    # 입력은 서버에 남기지 않는다
    row = _row(factory, job_id)
    assert _SECRET not in row.internal_json and _SECRET not in (row.result_json or "")


async def test_job_client_has_no_retries_and_its_own_timeout(factory, make_fake_anthropic_client):
    fake = make_fake_anthropic_client(_CLEAN)
    _, tasks = await _create(factory, client=fake)
    await _run_pending(tasks)
    fake.with_options.assert_called_once_with(timeout=jobs.JOB_CALL_TIMEOUT_SECONDS, max_retries=0)
    assert fake.beta.messages.create.call_count == 1


async def test_stage_is_generating_while_the_call_is_in_flight(factory, make_fake_anthropic_client):
    fake = make_fake_anthropic_client(_CLEAN)
    entered, gate = asyncio.Event(), asyncio.Event()
    original = fake.beta.messages.create.side_effect

    async def gated(*args, **kwargs):
        entered.set()
        await gate.wait()
        return await original(*args, **kwargs)

    fake.beta.messages.create.side_effect = gated
    job_id, tasks = await _create(factory, client=fake)
    # 만든 지 좀 된 작업처럼 — 단계가 바뀌면 updated_at 이 지금으로 와야 유실 판정이 다시 센다.
    earlier = _utcnow() - timedelta(seconds=200)
    _age(factory, job_id, created_at=earlier, updated_at=earlier)
    running = asyncio.create_task(_run_pending(tasks))
    await asyncio.wait_for(entered.wait(), 5)
    view = await jobs.read_job(job_id, session_factory=factory)
    assert (view.status, view.stage, view.result, view.failure) == ("running", "generating", None, None)
    assert _row(factory, job_id).updated_at > _utcnow() - timedelta(seconds=30)
    gate.set()
    await running
    assert (await jobs.read_job(job_id, session_factory=factory)).stage == "done"


async def test_refusal_finishes_with_rule_sentences_and_records_the_category(factory, make_fake_anthropic_client):
    fake = make_fake_anthropic_client(
        None, stop_reason="refusal", model="claude-sonnet-5-5",
        stop_details={"type": "refusal", "category": "bio"},
    )
    job_id, tasks = await _create(factory, client=fake)
    await _run_pending(tasks)
    view = await jobs.read_job(job_id, session_factory=factory)
    assert view.status == "done" and view.result.section_sources == jobs.section_sources("fallback")
    internal = _internal(factory, job_id)
    assert internal["reason"] == "refusal"
    assert internal["calls"][0]["refusal_category"] == "bio"
    assert _stages(factory, job_id) == ["queued", "generating", "checking", "done"]
    # 고객 응답에는 사유가 없다
    assert "refusal" not in json.dumps(view.model_dump(mode="json"))


async def test_grounding_violation_is_internal_only(factory, make_fake_anthropic_client):
    bad = {**_CLEAN, "interpretation": {"confusion_analysis": "정확도 0.12345 이다.", "distribution_analysis": ""}}
    fake = make_fake_anthropic_client(bad)
    job_id, tasks = await _create(factory, client=fake)
    await _run_pending(tasks)
    view = await jobs.read_job(job_id, session_factory=factory)
    assert view.result.section_sources == jobs.section_sources("fallback")
    internal = _internal(factory, job_id)
    assert internal["reason"] == "grounding_failed"
    assert "0.12345" in internal["grounding"]["violations"]
    assert "0.12345" not in json.dumps(view.model_dump(mode="json"))


async def test_whole_job_deadline_fails_as_timeout(factory, monkeypatch, make_fake_anthropic_client):
    monkeypatch.setattr(jobs, "JOB_DEADLINE_SECONDS", 0.05)
    fake = make_fake_anthropic_client(_CLEAN, delay=2.0)
    job_id, tasks = await _create(factory, client=fake)
    await _run_pending(tasks)
    view = await jobs.read_job(job_id, session_factory=factory)
    assert (view.status, view.stage, view.failure, view.result) == ("failed", "failed", "timeout", None)
    assert len(view.benchmark_refs) > 0  # 8.1 표는 서술 실패와 무관하게 남는다
    internal = _internal(factory, job_id)
    assert internal["failure"] == "timeout"
    # 호출 도중에 끊겨 호출 기록은 없지만 요청은 나갔다 — 시도 1, 비용은 모름(0 이 아니다)
    assert (internal["calls"], internal["http_attempts"], internal["cost_usd"]) == ([], 1, None)


async def test_job_path_is_not_bound_by_the_old_route_deadline(factory, monkeypatch, make_fake_anthropic_client):
    """옛 경로의 서버 마감(120초)은 작업 경로에 걸리지 않는다 — 작업은 전체 10분 제한만 받는다."""
    from app.narrative import legacy as service

    monkeypatch.setattr(service, "NARRATIVE_DEADLINE_SECONDS", 0.05)
    fake = make_fake_anthropic_client(_CLEAN, delay=0.3)
    job_id, tasks = await _create(factory, client=fake)
    await _run_pending(tasks)
    assert _internal(factory, job_id)["source"] == "llm"


async def test_unexpected_error_fails_as_error(factory, monkeypatch):
    async def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(jobs, "generate_narrative", boom)
    job_id, tasks = await _create(factory)
    await _run_pending(tasks)  # 밖으로 던지지 않는다
    view = await jobs.read_job(job_id, session_factory=factory)
    assert (view.status, view.failure) == ("failed", "error")


async def test_slot_wait_limit_finishes_as_busy_without_calling(factory, monkeypatch, make_fake_anthropic_client):
    monkeypatch.setattr(jobs, "JOB_SLOT_WAIT_SECONDS", 0.05)
    fake = make_fake_anthropic_client(_CLEAN)
    job_id, tasks = await _create(factory, client=fake)
    async with contextlib.AsyncExitStack() as stack:
        for _ in range(concurrency.MAX_CONCURRENT_NARRATIVE_CALLS):
            await stack.enter_async_context(concurrency.narrative_slot())
        # 대기 상한이 배선돼 있지 않으면 칸이 비지 않아 영원히 기다린다 — 테스트가 멈추지 않게 바깥 제한을 둔다.
        await asyncio.wait_for(_run_pending(tasks), 5)
    fake.beta.messages.create.assert_not_called()
    assert _internal(factory, job_id)["reason"] == "busy"
    assert (await jobs.read_job(job_id, session_factory=factory)).status == "done"


async def test_mapping_slots_do_not_block_narrative(factory, make_fake_anthropic_client):
    """매핑 칸이 다 차도 서술은 제 칸으로 돈다(칸 분리)."""
    fake = make_fake_anthropic_client(_CLEAN)
    job_id, tasks = await _create(factory, client=fake)
    async with contextlib.AsyncExitStack() as stack:
        for _ in range(concurrency.MAX_CONCURRENT_MAPPING_CALLS):
            await stack.enter_async_context(concurrency.mapping_slot())
        await asyncio.wait_for(_run_pending(tasks), 5)
    assert _internal(factory, job_id)["source"] == "llm"


# ── 지갑: 보고서 단위 예약 ──────────────────────────────────────────────────────

async def test_job_reserves_one_report_at_creation_and_not_again(factory, make_fake_anthropic_client):
    fake = make_fake_anthropic_client(_CLEAN)
    before = llm_budget.narrative.remaining()
    _, tasks = await _create(factory, client=fake)
    assert llm_budget.narrative.remaining() == before - 1
    await _run_pending(tasks)
    assert llm_budget.narrative.remaining() == before - 1  # run_job 은 다시 쓰지 않는다


async def test_spent_wallet_finishes_with_rule_sentences_without_calling(factory, make_fake_anthropic_client):
    while llm_budget.narrative.try_consume():
        pass
    fake = make_fake_anthropic_client(_CLEAN)
    job_id, tasks = await _create(factory, client=fake)
    await _run_pending(tasks)
    fake.beta.messages.create.assert_not_called()
    internal = _internal(factory, job_id)
    assert (internal["budget"], internal["reason"]) == ("exceeded", "budget_exceeded")
    assert (await jobs.read_job(job_id, session_factory=factory)).status == "done"


async def test_no_key_and_trimmed_jobs_do_not_reserve(factory, make_fake_anthropic_client):
    before = llm_budget.narrative.remaining()
    await _create(factory)  # 키 없음
    big = _over((("fact_sheet", "per_class"), [{"label": str(i)} for i in range(N)]))
    fake = make_fake_anthropic_client(_CLEAN)
    job_id, tasks = await _create(factory, client=fake, raw=big)
    assert llm_budget.narrative.remaining() == before
    await _run_pending(tasks)
    fake.beta.messages.create.assert_not_called()
    internal = _internal(factory, job_id)
    assert (internal["reason"], internal["trimmed"], internal["budget"]) == (
        "input_too_large", ["fact_sheet.per_class"], "not_needed")


async def test_old_route_still_consumes_its_own_report(make_fake_anthropic_client):
    fake = make_fake_anthropic_client(_CLEAN)
    before = llm_budget.narrative.remaining()
    await generate_narrative(fake, _request())
    assert llm_budget.narrative.remaining() == before - 1


async def test_job_row_limit_completes_immediately_without_a_row(factory, make_fake_anthropic_client):
    """상한을 넘으면 429 가 아니라 행 없이 규칙 문장 결과를 바로 준다 — 한 사람이 상한을 다 써도 남은 사용자는
    완성본과 8.1 표를 받는다(S3a 검토)."""
    while llm_budget.narrative_jobs.try_consume():
        pass
    fake = make_fake_anthropic_client(_CLEAN)
    before = llm_budget.narrative.remaining()
    tasks = BackgroundTasks()
    created = await jobs.create_job(_body(_request()), client=fake, session_factory=factory, background_tasks=tasks)
    assert (created.job_id, created.status, created.stage) == (None, "done", "done")
    assert created.result.section_sources == jobs.section_sources("fallback")
    assert [r.model_dump() for r in created.benchmark_refs] == [
        r.model_dump() for r in (await generate_narrative(None, _request())).benchmark_refs]
    assert tasks.tasks == [] and llm_budget.narrative.remaining() == before
    fake.beta.messages.create.assert_not_called()
    with factory() as db:
        assert db.query(NarrativeJob).count() == 0


def test_job_creation_wallet_limit_is_pinned():
    assert llm_budget.MAX_NARRATIVE_JOBS_PER_HOUR == 200
    assert llm_budget.narrative_jobs.limit == 200


# ── 유실·늦은 완료·보관 ─────────────────────────────────────────────────────────

def _age(factory, job_id, **fields):
    with factory() as db:
        job = db.get(NarrativeJob, job_id)
        for k, v in fields.items():
            setattr(job, k, v)
        db.commit()


async def test_stalled_job_is_judged_lost_on_read_and_stays_failed(factory, make_fake_anthropic_client):
    fake = make_fake_anthropic_client(_CLEAN)
    job_id, tasks = await _create(factory, client=fake)
    old = _utcnow() - timedelta(seconds=jobs.LOST_IDLE_SECONDS + 5)
    _age(factory, job_id, created_at=old, updated_at=old)
    view = await jobs.read_job(job_id, session_factory=factory)
    assert (view.status, view.stage, view.failure) == ("failed", "failed", "lost")
    internal = _internal(factory, job_id)
    assert internal["failure"] == "lost" and internal["last_stage"] == "queued"
    assert internal["lost_judged_at"].endswith("Z")
    # 늦게 끝난 작업이 실패를 완료로 뒤집지 않는다
    await _run_pending(tasks)
    again = await jobs.read_job(job_id, session_factory=factory)
    assert (again.status, again.stage, again.failure, again.result) == ("failed", "failed", "lost", None)


async def test_running_job_within_limits_is_not_lost(factory):
    job_id, _ = await _create(factory)
    recent = _utcnow() - timedelta(seconds=jobs.LOST_IDLE_SECONDS - 30)
    _age(factory, job_id, created_at=recent, updated_at=recent)
    view = await jobs.read_job(job_id, session_factory=factory)
    assert (view.status, view.failure) == ("running", None)


async def test_finished_result_expires_but_internal_record_stays(factory):
    job_id, tasks = await _create(factory)
    await _run_pending(tasks)
    _age(factory, job_id, finished_at=_utcnow() - jobs.result_ttl() - timedelta(seconds=5))
    view = await jobs.read_job(job_id, session_factory=factory)
    assert (view.status, view.stage, view.failure, view.result, view.benchmark_refs) == (
        "failed", "failed", "expired", None, [])
    row = _row(factory, job_id)
    assert row.result_json is None and row.status == "done"
    assert json.loads(row.internal_json)["reason"] == "no_key"


async def test_new_job_sweeps_expired_results_and_old_records(factory):
    expired_id, t1 = await _create(factory)
    await _run_pending(t1)
    old_id, t2 = await _create(factory)
    await _run_pending(t2)
    _age(factory, expired_id, finished_at=_utcnow() - jobs.result_ttl() - timedelta(minutes=1))
    _age(factory, old_id, created_at=_utcnow() - jobs.record_retention() - timedelta(minutes=1))
    await _create(factory)
    with factory() as db:
        assert db.get(NarrativeJob, old_id) is None
        swept = db.get(NarrativeJob, expired_id)
        assert swept.result_json is None and swept.internal_json != "{}"


# ── 경로(HTTP) ─────────────────────────────────────────────────────────────────

def test_post_then_get_round_trip_without_key(api):
    created = api.post("/api/narrative/jobs", json=_body())
    assert created.status_code == 200, created.text
    body = created.json()
    assert set(body) == {"job_id", "status", "stage", "benchmark_refs", "result"}
    assert (body["status"], body["stage"], body["result"]) == ("running", "queued", None)
    got = api.get(f"/api/narrative/jobs/{body['job_id']}")
    assert got.status_code == 200
    view = got.json()
    assert set(view) == {"job_id", "status", "stage", "failure", "benchmark_refs", "result", "created_at", "updated_at"}
    assert (view["status"], view["stage"], view["failure"]) == ("done", "done", None)
    assert set(view["result"]) == {
        "interpretation", "conclusion", "recommendation_narrative", "recommendations", "section_sources"}
    assert view["result"]["section_sources"] == jobs.section_sources("fallback")
    assert view["benchmark_refs"] == body["benchmark_refs"]
    assert view["created_at"].endswith("Z") and view["updated_at"].endswith("Z")


def test_job_benchmark_refs_match_the_old_route(api):
    old = TestClient(app).post("/api/generate-narrative", json=_body()).json()
    new = api.post("/api/narrative/jobs", json=_body()).json()
    assert new["benchmark_refs"] == old["benchmark_refs"] and new["benchmark_refs"]


def test_job_narrative_matches_the_old_route_fallback(api):
    """S3a 의 생성은 S1 한 번 호출을 감쌀 뿐이다 — 같은 요청의 규칙 문장이 옛 경로와 같다."""
    old = TestClient(app).post("/api/generate-narrative", json=_body()).json()
    job_id = api.post("/api/narrative/jobs", json=_body()).json()["job_id"]
    result = api.get(f"/api/narrative/jobs/{job_id}").json()["result"]
    for key in ("interpretation", "conclusion", "recommendation_narrative", "recommendations"):
        assert result[key] == old[key], key


def test_router_passes_the_anthropic_client(api, make_fake_anthropic_client):
    fake = make_fake_anthropic_client(_CLEAN)
    app.state.anthropic_client = fake
    job_id = api.post("/api/narrative/jobs", json=_body()).json()["job_id"]
    result = api.get(f"/api/narrative/jobs/{job_id}").json()["result"]
    assert result["section_sources"] == jobs.section_sources("llm")
    assert fake.beta.messages.create.call_count == 1


def test_oversized_input_completes_with_rule_sentences_not_422(api):
    big = _over((("fact_sheet", "confusion"), {"labels": [str(i) for i in range(N)], "matrix": [[0]]}))
    created = api.post("/api/narrative/jobs", json=big)
    assert created.status_code == 200, created.text
    view = api.get(f"/api/narrative/jobs/{created.json()['job_id']}").json()
    assert view["status"] == "done" and view["result"]["section_sources"] == jobs.section_sources("fallback")


def test_malformed_request_is_still_422(api):
    body = _body()
    del body["fact_sheet"]["verdict"]
    before = llm_budget.narrative_jobs.remaining()
    r = api.post("/api/narrative/jobs", json=body)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "fact_sheet", "verdict"]
    assert llm_budget.narrative_jobs.remaining() == before  # 잘못된 요청은 행 지갑을 쓰지 않는다
    assert api.post("/api/narrative/jobs", content=b"not json", headers={"content-type": "application/json"}).status_code == 422


def test_row_limit_over_http_is_200_with_result(api):
    while llm_budget.narrative_jobs.try_consume():
        pass
    r = api.post("/api/narrative/jobs", json=_body())
    assert r.status_code == 200
    body = r.json()
    assert (body["job_id"], body["status"]) == (None, "done")
    assert body["result"]["section_sources"] == jobs.section_sources("fallback") and body["benchmark_refs"]


def test_unknown_job_is_404(api):
    assert api.get("/api/narrative/jobs/00000000-0000-0000-0000-000000000000").status_code == 404
    assert api.get("/api/narrative/jobs/" + "x" * 65).status_code == 422


def test_job_routes_are_documented_with_the_request_model():
    spec = app.openapi()
    body = spec["paths"]["/api/narrative/jobs"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert body["$ref"] == "#/components/schemas/NarrativeRequest"


# ── DB 표 등록(init_db) ─────────────────────────────────────────────────────────

def test_init_db_creates_the_jobs_table(tmp_path):
    """init_db 의 import 목록에 모델이 있어야 운영 DB 에 표가 생긴다(다른 테스트가 모델을 import 해 둔 것과
    무관하게 보려고 새 인터프리터에서 돌린다)."""
    db = tmp_path / "boot.db"
    code = "from app.core import database; database.init_db()"
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db}"}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=root, capture_output=True)
    engine = create_engine(f"sqlite:///{db}")
    try:
        cols = {c["name"] for c in inspect(engine).get_columns("narrative_jobs")}
    finally:
        engine.dispose()
    assert cols == {"id", "status", "stage", "result_json", "internal_json", "created_at", "updated_at", "finished_at"}


def test_fake_with_options_rejects_args_the_sdk_does_not_take(make_fake_anthropic_client):
    fake = make_fake_anthropic_client(_CLEAN)
    with pytest.raises(TypeError):
        fake.with_options(retries=0)
    assert fake_beta_message(_CLEAN).model == "claude-sonnet-5-5"


# ── S3a 검토 반영 ───────────────────────────────────────────────────────────────

def _problems_request(**kw) -> NarrativeRequest:
    req = _request(**kw)
    fs = req.fact_sheet.model_copy(update={"problems": [
        schemas.ProblemFact(**_problem(1, recommend_section="9.1")),
        schemas.ProblemFact(**_problem(2, kind="weakest_class", priority="MEDIUM", recommend_section="9.2")),
    ]})
    return req.model_copy(update={"fact_sheet": fs})


async def test_template_filled_recommendations_mark_section_9_as_fallback(factory, make_fake_anthropic_client):
    """AI 가 문제 하나에만 권고를 쓰면 다른 문제의 권고 줄은 규칙 템플릿이다 — 9칸을 AI 로 적지 않는다."""
    content = {**_CLEAN, "recommendations": [
        {"problem_id": "P1", "priority": "HIGH", "category": "데이터", "action": "라벨을 다시 확인한다", "expected_impact": "오류 감소"}]}
    fake = make_fake_anthropic_client(content)
    job_id, tasks = await _create(factory, client=fake, req=_problems_request())
    await _run_pending(tasks)
    view = await jobs.read_job(job_id, session_factory=factory)
    assert view.result.section_sources == {**jobs.section_sources("llm"), "9": "fallback"}
    assert [r.problem_id for r in view.result.recommendations] == ["P1", "P2"]
    assert _internal(factory, job_id)["recommendations_from_template"] == ["P2"]


def test_job_recommendations_match_the_old_route_with_problems(api):
    body = _body(_problems_request())
    old = TestClient(app).post("/api/generate-narrative", json=body).json()
    job_id = api.post("/api/narrative/jobs", json=body).json()["job_id"]
    result = api.get(f"/api/narrative/jobs/{job_id}").json()["result"]
    assert result["recommendations"] == old["recommendations"] and len(result["recommendations"]) == 2


async def test_job_passes_no_inner_deadline(factory, monkeypatch):
    """작업 경로는 _narrate 에 마감 None(작업 전체 제한만)을 넘긴다 — 옛 경로 120초가 끼어들지 않게."""
    seen = {}
    original = jobs.generate_narrative

    async def spy(client, req, **kw):
        seen.update(kw)
        return await original(client, req, **kw)

    monkeypatch.setattr(jobs, "generate_narrative", spy)
    _, tasks = await _create(factory)
    await _run_pending(tasks)
    assert seen["deadline"] is None and seen["slot_wait"] == jobs.JOB_SLOT_WAIT_SECONDS


async def test_stage_write_failure_does_not_stop_the_job(factory, monkeypatch, make_fake_anthropic_client):
    def broken(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(jobs, "_db_set_stage", broken)
    fake = make_fake_anthropic_client(_CLEAN)
    job_id, tasks = await _create(factory, client=fake)
    await _run_pending(tasks)
    view = await jobs.read_job(job_id, session_factory=factory)
    assert view.status == "done" and _internal(factory, job_id)["source"] == "llm"
    assert fake.beta.messages.create.call_count == 1


async def test_finish_write_failure_is_swallowed(factory, monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(jobs, "_db_finish", broken)
    job_id, tasks = await _create(factory)
    await _run_pending(tasks)  # 밖으로 던지지 않는다
    assert _row(factory, job_id).status == "running"  # 조회 때 유실로 판정될 것이다


async def test_insert_failure_refunds_both_wallets(factory, monkeypatch, make_fake_anthropic_client):
    def broken(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(jobs, "_db_insert", broken)
    before = (llm_budget.narrative.remaining(), llm_budget.narrative_jobs.remaining())
    with pytest.raises(RuntimeError):
        await _create(factory, client=make_fake_anthropic_client(_CLEAN))
    assert (llm_budget.narrative.remaining(), llm_budget.narrative_jobs.remaining()) == before


@pytest.mark.parametrize("value", ["1e300", "inf", "nan", "1000000"])
def test_huge_retention_values_fall_back_instead_of_overflowing(monkeypatch, value):
    monkeypatch.setenv("NARRATIVE_JOB_RETENTION_DAYS", value)
    monkeypatch.setenv("NARRATIVE_RESULT_TTL_HOURS", value)
    assert jobs.record_retention() == timedelta(days=90)
    assert jobs.result_ttl() == timedelta(hours=24)
    _utcnow() - jobs.record_retention()  # 넘치지 않는다


def test_env_values_are_wired_into_the_modules(tmp_path):
    """칸 수·시간당 상한 환경변수가 실제로 모듈 값에 닿고, 오타는 기본값이 된다(새 인터프리터에서 import)."""
    code = (
        "from app.core import concurrency, llm_budget;"
        "print(concurrency.MAX_CONCURRENT_MAPPING_CALLS, concurrency.MAX_CONCURRENT_NARRATIVE_CALLS,"
        " llm_budget.MAX_NARRATIVE_JOBS_PER_HOUR, llm_budget.MAX_NARRATIVE_REPORTS_PER_HOUR, llm_budget.MAX_LLM_CALLS_PER_HOUR)"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "MAPPING_LLM_SLOTS": "3", "NARRATIVE_LLM_SLOTS": "7", "NARRATIVE_JOBS_PER_HOUR": "200.0",
           "NARRATIVE_REPORTS_PER_HOUR": "0", "LLM_CALLS_PER_HOUR": "45"}
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=root, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["3", "7", "200", "30", "45"]


# ── 서술용 파생값(facts, S4b) — 상한·형식 오류는 422 막다른 길이 아니라 줄이거나 버린다 ──────────────

def _with_facts(facts: dict) -> dict:
    body = _body()
    body["fact_sheet"]["facts"] = facts
    return body


def _low(n: int) -> list[dict]:
    return [{"label": f"c{i}", "support": 1, "share": 0.001, "share_text": "0.001"} for i in range(n)]


def test_facts_class_list_over_limit_is_emptied():
    req, trimmed = jobs.parse_request(_with_facts({"low_support_classes": _low(N)}))
    assert trimmed == ["fact_sheet.facts.low_support_classes"]
    assert req.fact_sheet.facts is not None and req.fact_sheet.facts.low_support_classes == []


def test_facts_bias_rows_over_limit_are_emptied_but_unit_stays():
    rows = [{"label": f"c{i}", "predicted": 1, "actual": 0, "difference": 1} for i in range(N)]
    req, trimmed = jobs.parse_request(_with_facts({"prediction_bias": {"unit": "per_class", "rows": rows}}))
    assert trimmed == ["fact_sheet.facts.prediction_bias.rows"]
    assert req.fact_sheet.facts.prediction_bias.unit == "per_class"
    assert req.fact_sheet.facts.prediction_bias.rows == []


def test_facts_total_over_limit_empties_lists_first_then_drops_facts():
    # 목록만 길면 목록을 비우고 facts 는 남긴다(판정 근거 등은 그대로).
    long_label = "가" * 200
    lists = {
        "low_support_classes": [{**r, "label": long_label + r["label"]} for r in _low(200)],
        "verdict_detail": {"verdict": "PASS", "judged_count": 1, "passed_count": 1},
    }
    req, trimmed = jobs.parse_request(_with_facts(lists))
    assert trimmed == ["fact_sheet.facts(lists)"]
    assert req.fact_sheet.facts.low_support_classes == []
    assert req.fact_sheet.facts.verdict_detail.verdict == "PASS"
    # 목록 밖 칸(생략 사유 64개 × 300자 + 이름)만으로 넘으면 facts 를 버린다.
    skipped = [{"metric_id": f"M{i}", "display_name": "지" * 200, "reason": "사" * 300} for i in range(64)]
    req, trimmed = jobs.parse_request(_with_facts({"ci_skipped": skipped}))
    assert trimmed == ["fact_sheet.facts"]
    assert req.fact_sheet.facts is None


def test_facts_format_error_drops_facts_without_forcing_rule_sentences():
    req, trimmed = jobs.parse_request(_with_facts({"declarations": {"risk_direction": "maybe"}}))
    assert trimmed == [jobs.FACTS_DROPPED]
    assert req.fact_sheet.facts is None
    assert req.fact_sheet == _sample_request().fact_sheet  # 나머지는 그대로


def test_facts_format_error_and_limit_error_together():
    body = _with_facts({"declarations": {"usage_mode": "sometimes"}})
    body["fact_sheet"]["per_class"] = [{"label": str(i)} for i in range(N)]
    req, trimmed = jobs.parse_request(body)
    assert trimmed[0] == jobs.FACTS_DROPPED and "fact_sheet.per_class" in trimmed
    assert req.fact_sheet.facts is None and req.fact_sheet.per_class == []


def test_non_facts_format_error_is_still_422():
    body = _with_facts({"declarations": {"risk_direction": "maybe"}})
    body["fact_sheet"]["verdict"] = None  # facts 밖의 형식 오류
    with pytest.raises(ValidationError):
        jobs.parse_request(body)


async def test_job_with_invalid_facts_still_calls_the_model(factory, make_fake_anthropic_client):
    fake = make_fake_anthropic_client(_CLEAN)
    job_id, tasks = await _create(factory, client=fake, raw=_with_facts({"declarations": {"risk_direction": "x"}}))
    await _run_pending(tasks)
    fake.beta.messages.create.assert_called_once()
    internal = _internal(factory, job_id)
    assert (internal["budget"], internal["trimmed"]) == ("reserved", [jobs.FACTS_DROPPED])
    assert internal["reason"] is None


async def test_job_with_facts_over_limit_finishes_with_rule_sentences(factory, make_fake_anthropic_client):
    fake = make_fake_anthropic_client(_CLEAN)
    job_id, tasks = await _create(factory, client=fake, raw=_with_facts({"low_support_classes": _low(N)}))
    await _run_pending(tasks)
    fake.beta.messages.create.assert_not_called()
    internal = _internal(factory, job_id)
    assert (internal["reason"], internal["trimmed"]) == ("input_too_large", ["fact_sheet.facts.low_support_classes"])


def test_legacy_route_rejects_facts_over_limit():
    with pytest.raises(ValidationError):
        schemas.NarrativeRequest.model_validate(_with_facts({"low_support_classes": _low(N)}))
