"""test_postgres_roundtrip.py — 서술 작업 표(narrative_jobs)를 진짜 Postgres 에 붙인 한 바퀴(서술 라운드 세션 A, 계획 S9).

운영 발급 DB 는 Neon(Postgres)이고 평소 테스트는 인메모리 SQLite 다. SQLite 와 다른 곳 — 동시 쓰기(장 셋이 동시에 단계를
쓴다), tz 없는 시각 칸, 한국어 JSON 글 — 을 Postgres 에서 한 번 돌린다.

`NARRATIVE_PG_URL`(예: postgresql://postgres:…@127.0.0.1:55433/narrative)이 있을 때만 돈다 — CI·평소에는 건너뛴다.
로컬에서 돌리는 법은 Capstone_Back/docs/ARCHITECTURE.md §10.7(기록 둘과 작업 표)의 "Postgres 왕복".
"""
import json
import os
import uuid
from datetime import timedelta

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.narrative import jobs
from app.narrative.models import NarrativeJob, _utcnow
from app.narrative.schemas import NarrativeRequest

PG_URL = os.getenv("NARRATIVE_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="NARRATIVE_PG_URL 이 없으면 Postgres 왕복을 건너뛴다")

CONTRACT = os.path.join(os.path.dirname(__file__), "contract")
# 이진 계약 예시는 문제 3개라 장 셋(7·8·9)을 동시에 부른다. 7.1 은 FN 위험 선언(D24) 숫자 43 을 담는다.
D24_OK = "의뢰자가 위험하다고 밝힌 FN 은 43건이다."


def _binary() -> NarrativeRequest:
    with open(os.path.join(CONTRACT, "fact_sheet_binary.json")) as f:
        return NarrativeRequest.model_validate(json.load(f))


@pytest.fixture
def pg_factory():
    engine = create_engine(PG_URL, pool_pre_ping=True)
    NarrativeJob.__table__.drop(engine, checkfirst=True)
    NarrativeJob.__table__.create(engine)
    yield sessionmaker(bind=engine, autoflush=False, autocommit=False)
    NarrativeJob.__table__.drop(engine, checkfirst=True)
    engine.dispose()


async def _run(tasks: BackgroundTasks) -> None:
    for task in tasks.tasks:
        await task.func(*task.args, **task.kwargs)


async def test_job_roundtrip_with_concurrent_chapter_stage_writes(pg_factory, make_section_client):
    """작업 생성 → 장 셋 동시 호출(각 호출이 '생성 중' 단계를 동시에 쓴다) → 검사 → 완료 → 조회."""
    fake = make_section_client({"section_7_1": D24_OK}, delay=0.3)
    tasks = BackgroundTasks()
    created = await jobs.create_job(
        json.loads(_binary().model_dump_json()), client=fake, session_factory=pg_factory, background_tasks=tasks,
    )
    queued = await jobs.read_job(created.job_id, session_factory=pg_factory)
    assert (queued.status, queued.stage) == ("running", "queued")
    assert [r.metric for r in created.benchmark_refs]  # 8.1 표는 생성 응답에 바로

    await _run(tasks)

    view = await jobs.read_job(created.job_id, session_factory=pg_factory)
    assert (view.status, view.stage, view.failure) == ("done", "done", None)
    assert view.result.section_sources == {
        "7.1": "llm", "7.2": "llm", "7.3": "fixed", "8.1": "llm", "8.2": "llm", "8.3": "llm", "9": "llm",
    }
    assert view.result.interpretation.confusion_analysis == D24_OK  # 한국어 글이 그대로 돌아온다
    assert fake.beta.messages.create.await_count == 3

    with pg_factory() as db:
        row = db.get(NarrativeJob, created.job_id)
        internal = json.loads(row.internal_json)
        assert row.created_at.tzinfo is None and row.finished_at is not None
        assert row.created_at <= row.updated_at <= row.finished_at
        # 시각은 tz 없는 UTC — 지금(UTC)과 몇 초 안이다(서버 시간대로 밀리지 않는다).
        assert abs(_utcnow() - row.finished_at) < timedelta(minutes=1)
    stages = [s["stage"] for s in internal["stages"]]
    assert stages.count("generating") == 3 and stages[-1] == "done" and "checking" in stages
    assert internal["http_attempts"] == 3 and internal["failure"] is None


def test_route_roundtrip_without_key(pg_factory):
    """POST /api/narrative/jobs → GET 완료(키 없음 → 칸마다 규칙 문장). CI 부팅 검사와 같은 왕복을 Postgres 로."""
    app.dependency_overrides[jobs.get_session_factory] = lambda: pg_factory
    try:
        api = TestClient(app)
        created = api.post("/api/narrative/jobs", json=json.loads(_binary().model_dump_json()))
        assert created.status_code == 200, created.text
        job_id = created.json()["job_id"]
        got = api.get(f"/api/narrative/jobs/{job_id}").json()
        assert (got["status"], got["stage"]) == ("done", "done")
        assert set(got["result"]["section_sources"].values()) <= {"fallback", "fixed"}
        assert "의뢰자가 더 위험하다고 밝힌 거짓 음성(FN, 놓침)은 43건" in got["result"]["interpretation"]["confusion_analysis"]
        assert api.get(f"/api/narrative/jobs/{uuid.uuid4()}").status_code == 404
    finally:
        app.dependency_overrides.pop(jobs.get_session_factory, None)


async def test_lost_and_expired_are_judged_from_postgres_timestamps(pg_factory):
    """유실(마지막 갱신 뒤 오래 멈춤)·만료(결과 글을 지움) 판정이 Postgres 시각 칸으로도 맞다."""
    now = _utcnow()
    lost_id, done_id = str(uuid.uuid4()), str(uuid.uuid4())
    with pg_factory() as db:
        db.add(NarrativeJob(id=lost_id, status="running", stage="generating", internal_json="{}",
                            created_at=now - timedelta(minutes=20), updated_at=now - timedelta(minutes=20)))
        db.add(NarrativeJob(id=done_id, status="done", stage="done", result_json=None, internal_json="{}",
                            created_at=now - timedelta(days=2), updated_at=now - timedelta(days=2),
                            finished_at=now - timedelta(days=2)))
        db.commit()
    lost = await jobs.read_job(lost_id, session_factory=pg_factory)
    assert (lost.status, lost.failure) == ("failed", "lost")
    expired = await jobs.read_job(done_id, session_factory=pg_factory)
    assert (expired.status, expired.failure) == ("failed", "expired")
    with pg_factory() as db:
        assert db.execute(text("select count(*) from narrative_jobs")).scalar() == 2
