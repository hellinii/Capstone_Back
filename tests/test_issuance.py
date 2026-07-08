"""test_issuance.py — 발급 메타 채번/멱등/재발급/API 테스트 (설계 문서 §9 Phase C).

격리: 테스트마다 in-memory SQLite(StaticPool)로 fresh DB 를 만들어 기관을 시드한다.
API 테스트는 get_db 의존성을 이 세션으로 오버라이드하고, lifespan(실제 data/app.db)은
띄우지 않는다(TestClient 를 context manager 없이 사용).
"""
import re
import threading
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core import database
from app.core.database import Base, configure_sqlite, get_db
from app.issuance.bootstrap import DEFAULT_ORGANIZATION
from app.issuance.models import Issuance, Organization, Report
from app.issuance import service as svc
from app.issuance.service import IssuanceError


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # in-memory DB 를 단일 연결로 공유
    )
    configure_sqlite(engine)  # 프로덕션과 동일 설정(FK + BEGIN IMMEDIATE)
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = TestingSession()
    db.add(Organization(**DEFAULT_ORGANIZATION))
    db.commit()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _make_file_engine(db_path):
    """프로덕션과 동일 설정의 파일 기반 SQLite 엔진(멀티 커넥션·실제 잠금 검증용)."""
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    configure_sqlite(engine)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def client(db_session):
    from app.main import app

    app.dependency_overrides[get_db] = lambda: db_session
    c = TestClient(app)  # context manager 미사용 → lifespan(실제 DB) 미실행
    try:
        yield c
    finally:
        app.dependency_overrides.clear()


# ── 서비스 레벨: 채번 순번 ─────────────────────────────────────────────────────

def test_issue_creates_first_number(db_session):
    r = svc.issue_report(
        db_session, run_id="run-1", model_name="MyModel",
        model_version="v2", now=datetime(2026, 7, 4),
    )
    assert r.report_no == "RPT-2026-0001"
    assert r.year == 2026 and r.seq == 1
    assert r.current_version == "v1.0"
    assert len(r.issuances) == 1
    assert r.issuances[0].version == "v1.0"
    assert r.issuances[0].status == "issued"
    assert r.issuances[0].note == "최초 발급"


def test_sequence_increments_within_year(db_session):
    a = svc.issue_report(db_session, run_id="run-1", now=datetime(2026, 7, 4))
    b = svc.issue_report(db_session, run_id="run-2", now=datetime(2026, 7, 4))
    c = svc.issue_report(db_session, run_id="run-3", now=datetime(2026, 7, 4))
    assert [a.report_no, b.report_no, c.report_no] == [
        "RPT-2026-0001", "RPT-2026-0002", "RPT-2026-0003",
    ]


def test_year_boundary_resets_seq(db_session):
    svc.issue_report(db_session, run_id="r-2026", now=datetime(2026, 12, 31))
    r = svc.issue_report(db_session, run_id="r-2027", now=datetime(2027, 1, 1))
    assert r.report_no == "RPT-2027-0001"


# ── 서비스 레벨: 멱등 ──────────────────────────────────────────────────────────

def test_issue_idempotent_same_run(db_session):
    a = svc.issue_report(db_session, run_id="run-1", now=datetime(2026, 7, 4))
    b = svc.issue_report(db_session, run_id="run-1", now=datetime(2026, 7, 4))
    assert a.id == b.id
    assert a.report_no == b.report_no
    assert db_session.query(Report).count() == 1  # 중복 채번 없음


# ── 서비스 레벨: 재발급(정정) ──────────────────────────────────────────────────

def test_reissue_bumps_version_and_history(db_session):
    svc.issue_report(db_session, run_id="run-1", now=datetime(2026, 7, 4))
    r = svc.reissue_report(
        db_session, report_no="RPT-2026-0001", note="오타 정정", now=datetime(2026, 7, 5),
    )
    assert r.report_no == "RPT-2026-0001"  # 같은 번호 유지
    assert r.current_version == "v1.1"
    assert len(r.issuances) == 2
    assert r.issuances[0].status == "superseded"
    assert r.issuances[1].status == "issued"
    assert r.issuances[1].version == "v1.1"
    assert r.issuances[1].note == "오타 정정"


def test_reissue_twice(db_session):
    svc.issue_report(db_session, run_id="run-1", now=datetime(2026, 7, 4))
    svc.reissue_report(db_session, report_no="RPT-2026-0001", note="1차 정정")
    r = svc.reissue_report(db_session, report_no="RPT-2026-0001", note="2차 정정")
    assert r.current_version == "v1.2"
    assert len(r.issuances) == 3
    assert [i.status for i in r.issuances] == ["superseded", "superseded", "issued"]


def test_reissue_not_found(db_session):
    with pytest.raises(IssuanceError) as ei:
        svc.reissue_report(db_session, report_no="RPT-2099-9999", note="x")
    assert ei.value.code == "not_found"


# ── 서비스 레벨: 버전 bump / issuer 기본값 ────────────────────────────────────

def test_bump_version():
    assert svc.bump_version("v1.0") == "v1.1"
    assert svc.bump_version("v1.9") == "v1.10"  # minor 두 자리 경계
    assert svc.bump_version("v2.3") == "v2.4"
    with pytest.raises(IssuanceError):
        svc.bump_version("bogus")


def test_issuer_defaults_to_org(db_session):
    r = svc.issue_report(db_session, run_id="run-1", now=datetime(2026, 7, 4))
    assert r.issuances[0].issuer == "한국 AI 인증원 평가부"


def test_issuer_explicit_override(db_session):
    r = svc.issue_report(
        db_session, run_id="run-1", issuer="검토자 홍길동", now=datetime(2026, 7, 4),
    )
    assert r.issuances[0].issuer == "검토자 홍길동"


# ── API 레벨 (TestClient) ─────────────────────────────────────────────────────

def test_api_get_organization(client):
    resp = client.get("/api/organization")
    assert resp.status_code == 200
    body = resp.json()
    assert body["org_name"] == "한국 AI 인증원"
    assert body["evaluator"] == "자동 평가 엔진"


def test_api_issue_reissue_reopen_flow(client):
    # 발급
    resp = client.post(
        "/api/reports/issue",
        json={"run_id": "run-1", "model_name": "MyModel", "model_version": "v2"},
    )
    assert resp.status_code == 200
    d = resp.json()
    assert re.match(r"^RPT-\d{4}-0001$", d["report_no"])
    assert d["version"] == "v1.0"
    assert d["issuer"] == "한국 AI 인증원 평가부"
    assert d["organization"]["org_name"] == "한국 AI 인증원"
    assert len(d["history"]) == 1
    # 시각은 offset 포함 ISO8601(프론트가 KST 로 변환 가능해야 함)
    assert d["issued_at"].endswith("+00:00")
    assert d["history"][0]["issued_at"].endswith("+00:00")
    report_no = d["report_no"]

    # 멱등: 같은 run 재발급 요청 → 같은 번호
    resp2 = client.post("/api/reports/issue", json={"run_id": "run-1"})
    assert resp2.status_code == 200
    assert resp2.json()["report_no"] == report_no

    # 재오픈(조회)
    g = client.get(f"/api/reports/{report_no}")
    assert g.status_code == 200
    assert g.json()["report_no"] == report_no

    # 정정 발급
    rr = client.post(f"/api/reports/{report_no}/reissue", json={"note": "정정"})
    assert rr.status_code == 200
    rd = rr.json()
    assert rd["report_no"] == report_no
    assert rd["version"] == "v1.1"
    assert len(rd["history"]) == 2


def test_api_not_found(client):
    assert client.get("/api/reports/RPT-2099-9999").status_code == 404
    assert (
        client.post("/api/reports/RPT-2099-9999/reissue", json={"note": "x"}).status_code
        == 404
    )


def test_api_issue_requires_run_id(client):
    # run_id 누락 → pydantic 422
    assert client.post("/api/reports/issue", json={"model_name": "M"}).status_code == 422


def test_api_issue_rejects_blank_run_id(client):
    # 빈/공백 run_id → 422 (서로 다른 평가가 한 번호로 병합되는 것 차단)
    assert client.post("/api/reports/issue", json={"run_id": ""}).status_code == 422
    assert client.post("/api/reports/issue", json={"run_id": "   "}).status_code == 422


# ── 파일 DB(프로덕션 배선) 레벨 — 리뷰 #7, 설계 §9 ────────────────────────────

def test_file_db_seed_and_fk(tmp_path):
    """실제 파일 DB: 테이블·파일 생성 + FK 강제가 새 커넥션에서도 작동."""
    db_file = tmp_path / "app.db"
    engine = _make_file_engine(db_file)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    seed = Session()
    seed.add(Organization(**DEFAULT_ORGANIZATION))
    seed.commit()
    seed.close()
    assert db_file.exists()

    # 별도 커넥션(멀티 커넥션 환경)에서도 FK 위반 거부
    other = Session()
    other.add(Issuance(report_id=999, version="v1.0", issuer="x"))
    with pytest.raises(IntegrityError):
        other.commit()
    other.rollback()
    other.close()
    engine.dispose()


def test_concurrent_issue_distinct_numbers(tmp_path):
    """동시 발급(2 스레드·2 커넥션): BEGIN IMMEDIATE 직렬화로 500 없이 연속 번호 채번."""
    engine = _make_file_engine(tmp_path / "app.db")
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    seed = Session()
    seed.add(Organization(**DEFAULT_ORGANIZATION))
    seed.commit()
    seed.close()

    results: dict[str, str] = {}
    errors: list[Exception] = []
    barrier = threading.Barrier(2)  # 두 스레드가 최대한 동시에 진입하도록

    def worker(run_id: str):
        db = Session()
        try:
            barrier.wait()
            r = svc.issue_report(db, run_id=run_id, now=datetime(2026, 7, 4))
            results[run_id] = r.report_no
        except Exception as e:  # noqa: BLE001 — 테스트에서 모든 실패를 수집
            errors.append(e)
        finally:
            db.close()

    threads = [threading.Thread(target=worker, args=(f"run-{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"동시 발급 중 예외: {errors}"
    assert set(results.values()) == {"RPT-2026-0001", "RPT-2026-0002"}
    engine.dispose()


def test_concurrent_reissue_no_duplicate_version(tmp_path):
    """동시 재발급(2 스레드): UNIQUE(report_id, version)+재시도로 중복 버전 없이 v1.1→v1.2."""
    engine = _make_file_engine(tmp_path / "app.db")
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    seed = Session()
    seed.add(Organization(**DEFAULT_ORGANIZATION))
    seed.commit()
    svc.issue_report(seed, run_id="run-1", now=datetime(2026, 7, 4))
    report_no = seed.query(Report).one().report_no
    seed.close()

    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def worker(note: str):
        db = Session()
        try:
            barrier.wait()
            svc.reissue_report(db, report_no=report_no, note=note)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            db.close()

    threads = [threading.Thread(target=worker, args=(f"정정{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"동시 재발급 중 예외: {errors}"

    check = Session()
    report = check.query(Report).one()
    versions = [i.version for i in report.issuances]
    issued = [i for i in report.issuances if i.status == "issued"]
    check.close()
    engine.dispose()

    # 중복 버전 없음 + 현행(issued) 1건 + 최종 v1.2
    assert len(versions) == len(set(versions)), f"중복 버전: {versions}"
    assert versions == ["v1.0", "v1.1", "v1.2"]
    assert len(issued) == 1 and issued[0].version == "v1.2"
    assert report.current_version == "v1.2"
