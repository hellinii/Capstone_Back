"""tests/test_judge_issuance.py — 판 4 성적서 서버 발급 검사(app.issuance.judge_gate), 심판 라운드 J5b(계획 14-6·14-9a·사-6).

한 DB(임시 파일 SQLite)를 발급 API(get_db)와 서술 작업(jobs.get_session_factory) 두 의존성에 같이 꽂는다 — 따로 꽂으면 발급 쪽이
작업을 못 찾아 "작업 없음"으로 저절로 거절되어 시험에 이빨이 없다. 같은 장치로 "통과 → 발급" 양성 대조를 함께 본다.
작업은 실제 경로(POST /api/narrative/jobs)로 만든다 — 심판은 conftest judge_passes(통과) 또는 키 없음(보류).
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, configure_sqlite, get_db
from app.issuance import judge_gate
from app.issuance.bootstrap import DEFAULT_ORGANIZATION
from app.issuance.models import Organization
from app.main import app
from app.narrative import jobs
from app.narrative.models import NarrativeJob, _utcnow

DEV = Path(__file__).parent / "data" / "judge_dev"


@pytest.fixture
def shared(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'shared.db'}", connect_args={"check_same_thread": False})
    configure_sqlite(engine)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        db.add(Organization(**DEFAULT_ORGANIZATION))
        db.commit()

    def db_dep():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = db_dep
    app.dependency_overrides[jobs.get_session_factory] = lambda: factory
    try:
        yield TestClient(app), factory
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(jobs.get_session_factory, None)
        engine.dispose()


def _ed4() -> dict:
    return json.loads((DEV / "dev_binary.request.json").read_text(encoding="utf-8"))


def _job(client, raw) -> str:
    created = client.post("/api/narrative/jobs", json=raw)
    assert created.status_code == 200, created.text
    job_id = created.json()["job_id"]
    assert client.get(f"/api/narrative/jobs/{job_id}").json()["status"] == "done"
    return job_id


def _issue(client, *, run_id="run-1", content=None, job_id=None):
    body = {"run_id": run_id, "model_name": "m", "content": content}
    if job_id is not None:
        body["narrative_job_id"] = job_id
    return client.post("/api/reports/issue", json=body)


def _code(resp) -> str:
    assert resp.status_code == 409, resp.text
    return resp.json()["detail"].rsplit("(", 1)[1].rstrip(")")


ED4 = {"contentSchema": 4, "title": "판 4 성적서"}


# ── 판 판정(프론트 규칙과 같게, 닫힌 쪽) ──


@pytest.mark.parametrize("content, checked", [
    (None, False), ({"smoke": True}, False), ({"contentSchema": None}, False),
    ({"contentSchema": 1}, False), ({"contentSchema": 2}, False), ({"contentSchema": 3}, False),
    ({"contentSchema": 4}, True), ({"contentSchema": "4"}, True), ({"contentSchema": 4.0}, True),
    ({"contentSchema": 3.0}, True), ({"contentSchema": "3"}, True), ({"contentSchema": 5}, True),
    ({"contentSchema": 0}, True), ({"contentSchema": True}, True), ({"contentSchema": [4]}, True),
])
def test_edition_rule_is_closed(content, checked):
    assert judge_gate.content_is_edition4(content) is checked


# ── 첫 발급 ──


def test_smoke_and_edition3_content_are_not_checked(shared):
    client, _ = shared
    assert _issue(client, run_id="ci-smoke", content={"smoke": True}).status_code == 200  # CI 부팅 스모크 그대로
    assert _issue(client, run_id="ed3", content={"contentSchema": 3}).status_code == 200
    assert _issue(client, run_id="meta").status_code == 200


def test_passed_edition4_job_issues_and_every_other_case_is_rejected(shared, judge_passes):
    client, factory = shared
    passed = _job(client, _ed4())
    assert judge_passes  # 심판 자리가 실제로 불렸다(양성 대조가 진짜 통과 기록을 본다)
    ed3_raw = {**_ed4(), "report_edition": 3}
    ed3_raw.pop("judge_material")
    ed3_job = _job(client, ed3_raw)

    assert _code(_issue(client, run_id="a", content=ED4)) == "judge_job_missing"
    assert _code(_issue(client, run_id="b", content=ED4, job_id="  ")) == "judge_job_missing"
    assert _code(_issue(client, run_id="c", content=ED4, job_id="no-such-job")) == "judge_job_not_found"
    assert _code(_issue(client, run_id="d", content=ED4, job_id=ed3_job)) == "judge_job_not_edition4"
    ok = _issue(client, run_id="e", content=ED4, job_id=passed)
    assert ok.status_code == 200 and ok.json()["report_no"]


def test_held_job_is_rejected(shared):
    client, _ = shared
    app.state.openai_client = None  # 심판 키 없음 → 보류(no_key)
    held = _job(client, _ed4())
    assert client.get(f"/api/narrative/jobs/{held}").json()["result"]["judge"]["gate"] == "held"
    assert _code(_issue(client, content=ED4, job_id=held)) == "judge_not_passed"


def test_running_and_failed_jobs_are_rejected(shared):
    client, factory = shared
    jobs._db_insert(factory, "job-run", [], {"report_edition": 4}, _utcnow())
    assert _code(_issue(client, run_id="r", content=ED4, job_id="job-run")) == "judge_not_passed"
    jobs._db_finish(factory, "job-run", status="failed", narrative=None,
                    internal={"judge": {"gate": "passed"}}, now=_utcnow())  # 실패인데 기록만 passed — 상태가 먼저
    assert _code(_issue(client, run_id="f", content=ED4, job_id="job-run")) == "judge_not_passed"


def test_expired_job_record_is_rejected(shared, judge_passes):
    client, factory = shared
    passed = _job(client, _ed4())
    with factory() as db:
        job = db.get(NarrativeJob, passed)
        job.created_at = _utcnow() - jobs.record_retention() - timedelta(minutes=1)
        db.commit()
    assert _code(_issue(client, content=ED4, job_id=passed)) == "judge_record_expired"


def test_result_text_expiry_does_not_lose_the_gate(shared, judge_passes, monkeypatch):
    """결과 글은 24시간 뒤 지워져도 gate 는 내부 기록에 남는다(계획 사-4) — 발급은 된다."""
    client, factory = shared
    passed = _job(client, _ed4())
    with factory() as db:
        db.get(NarrativeJob, passed).result_json = None
        db.commit()
    assert _issue(client, content=ED4, job_id=passed).status_code == 200


# ── 정정 발급(14-9a) ──


def test_reissue_has_the_same_check(shared, judge_passes):
    client, factory = shared
    passed = _job(client, _ed4())
    issued = _issue(client, run_id="run-r", content=ED4, job_id=passed).json()
    no = issued["report_no"]

    def reissue(job_id=None, content=ED4):
        body = {"run_id": "run-r", "note": "정정", "content": content}
        if job_id is not None:
            body["narrative_job_id"] = job_id
        return client.post(f"/api/reports/{no}/reissue", json=body)

    assert _code(reissue()) == "judge_job_missing"
    assert _code(reissue("no-such-job")) == "judge_job_not_found"
    app.state.openai_client = None
    held = _job(client, _ed4())
    assert _code(reissue(held)) == "judge_not_passed"
    assert reissue(passed).status_code == 200
    assert reissue(None, {"contentSchema": 3}).status_code == 200  # 판 3 내용은 검사하지 않는다


def test_job_id_field_is_optional_and_bounded():
    from app.issuance.schemas import IssueRequest, ReissueRequest

    assert IssueRequest(run_id="x").narrative_job_id is None
    with pytest.raises(Exception):
        IssueRequest(run_id="x", narrative_job_id="j" * 65)
    assert ReissueRequest(run_id="x", note="n").narrative_job_id is None
