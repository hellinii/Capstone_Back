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


# ── F-07: 채번 연도는 성적서에 인쇄되는 발급일(KST)과 같은 해여야 한다 ──────────

def test_numbering_year_follows_kst_not_utc(db_session):
    """[F-07] 12/31 15:00 UTC 는 KST 로 이미 새해다 — 번호도 새해여야 한다.

    발급 시각은 UTC 로 저장되지만 성적서에 인쇄되는 발급일은 프론트가 KST 로 변환한다
    (issuanceApi.ts formatKstDate). 채번만 UTC 연도를 쓰면 매년 12/31 15:00~24:00 UTC
    (= 1/1 00:00~09:00 KST) 구간에서 **번호는 전년도, 인쇄된 발급일은 새해**가 된다.
    연도별 채번 대장과 문서를 대조할 수 없게 된다.
    """
    r = svc.issue_report(db_session, run_id="r-newyear", now=datetime(2026, 12, 31, 15, 0))
    assert r.report_no == "RPT-2027-0001"
    assert r.year == 2027


def test_numbering_year_is_still_previous_year_just_before_kst_midnight(db_session):
    """[F-07] 12/31 14:59 UTC 는 KST 로 아직 12/31 23:59 다 — 전년도 번호가 맞다."""
    r = svc.issue_report(db_session, run_id="r-eve", now=datetime(2026, 12, 31, 14, 59))
    assert r.report_no == "RPT-2026-0001"
    assert r.year == 2026


def test_kst_boundary_starts_a_new_sequence(db_session):
    """[F-07] KST 연 경계를 넘으면 순번이 1 로 리셋된다(경계 양쪽을 한 테스트에서)."""
    a = svc.issue_report(db_session, run_id="r-a", now=datetime(2026, 12, 31, 14, 0))
    b = svc.issue_report(db_session, run_id="r-b", now=datetime(2026, 12, 31, 16, 0))
    assert a.report_no == "RPT-2026-0001"
    assert b.report_no == "RPT-2027-0001"


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
    rr = client.post(f"/api/reports/{report_no}/reissue", json={"run_id": "run-1", "note": "정정"})
    assert rr.status_code == 200
    rd = rr.json()
    assert rd["report_no"] == report_no
    assert rd["version"] == "v1.1"
    assert len(rd["history"]) == 2


def test_api_not_found(client):
    assert client.get("/api/reports/RPT-2099-9999").status_code == 404
    assert (
        client.post("/api/reports/RPT-2099-9999/reissue", json={"run_id": "run-x", "note": "x"}).status_code
        == 404
    )


def test_api_issue_requires_run_id(client):
    # run_id 누락 → pydantic 422
    assert client.post("/api/reports/issue", json={"model_name": "M"}).status_code == 422


def test_api_issue_rejects_blank_run_id(client):
    # 빈/공백 run_id → 422 (서로 다른 평가가 한 번호로 병합되는 것 차단)
    assert client.post("/api/reports/issue", json={"run_id": ""}).status_code == 422
    assert client.post("/api/reports/issue", json={"run_id": "   "}).status_code == 422


# ── F-06(a): 정정 사유 공란이 API 로 통과하던 문제 ────────────────────────────

def test_api_reissue_rejects_blank_note(client):
    """[F-06] 공란 정정 사유는 422 로 막는다.

    재발급은 이전 차수를 superseded 로 만들고 이력에 새 행을 남긴다. 그 이력은
    SignatureSection 을 통해 성적서에 그대로 인쇄되므로, 사유 없는 정정 이력이
    남으면 제3자가 무엇이 왜 바뀌었는지 판별할 근거가 사라진다.
    UI 는 ReportLayout.tsx:42 에서 note.trim() 으로 막고 있었지만 API 는 통과했다.
    """
    issued = client.post("/api/reports/issue", json={"run_id": "run-note"}).json()
    no = issued["report_no"]

    assert client.post(f"/api/reports/{no}/reissue", json={"run_id": "run-note", "note": ""}).status_code == 422
    assert client.post(f"/api/reports/{no}/reissue", json={"run_id": "run-note", "note": "   "}).status_code == 422

    # 막혔으므로 버전과 이력이 그대로여야 한다
    after = client.get(f"/api/reports/{no}").json()
    assert after["version"] == "v1.0"
    assert len(after["history"]) == 1


def test_api_reissue_accepts_real_note(client):
    """[F-06] 정상 사유는 그대로 통과한다(수정이 기능을 막지 않는다)."""
    issued = client.post("/api/reports/issue", json={"run_id": "run-note-ok"}).json()
    no = issued["report_no"]

    r = client.post(f"/api/reports/{no}/reissue", json={"run_id": "run-note-ok", "note": "지표 표기 오류 정정"})
    assert r.status_code == 200
    assert r.json()["version"] == "v1.1"
    assert r.json()["history"][-1]["note"] == "지표 표기 오류 정정"


def test_api_reissue_note_is_trimmed(client):
    """[F-06] 앞뒤 공백은 제거해 저장한다 — 인쇄물에 들쭉날쭉한 여백이 남지 않게."""
    issued = client.post("/api/reports/issue", json={"run_id": "run-note-trim"}).json()
    no = issued["report_no"]

    r = client.post(f"/api/reports/{no}/reissue", json={"run_id": "run-note-trim", "note": "  오탈자 정정  "})
    assert r.status_code == 200
    assert r.json()["history"][-1]["note"] == "오탈자 정정"


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


# ══════════════════════════════════════════════════════════════════════════════
# F-01 — 발급 DB 가 성적서 내용을 저장한다
# ══════════════════════════════════════════════════════════════════════════════

def _content(accuracy: float = 0.944, verdict: str = "PASS") -> dict:
    """성적서 내용의 축소판(실물 FinalReportData 는 최상위 25필드)."""
    return {
        "meta": {"taskType": "binary", "taskTypeLabel": "이진 분류"},
        "kpiResults": [
            {"metricId": "M1", "name": "Accuracy", "value": accuracy,
             "threshold": 0.9, "status": "pass"}
        ],
        "conclusion": {"verdict": verdict, "score": 92},
        "datasetInfo": {"sampleCount": 200},
    }


def test_issue_stores_and_returns_report_content(client):
    """[F-01] 발급 시 보낸 성적서 내용을 번호로 되찾을 수 있어야 한다.

    지금까지 발급 DB 에 남는 것은 번호·연도·순번·모델명·발급자·시각뿐이라
    "RPT-2026-0001 의 M1 값이 0.944 였는가"에 서버가 답할 수 없었다.
    """
    issued = client.post(
        "/api/reports/issue",
        json={"run_id": "run-content", "model_name": "M", "content": _content()},
    )
    assert issued.status_code == 200, issued.text
    no = issued.json()["report_no"]

    got = client.get(f"/api/reports/{no}/content")
    assert got.status_code == 200, got.text
    body = got.json()

    assert body["report_no"] == no
    assert body["version"] == "v1.0"
    assert body["content"]["kpiResults"][0]["value"] == 0.944
    assert body["content"]["conclusion"]["verdict"] == "PASS"
    assert len(body["content_hash"]) == 64  # sha256 hex


def test_content_hash_is_stable_for_same_content(client):
    """[F-01] 같은 내용이면 같은 해시 — 인쇄물 진위 대조의 근거."""
    a = client.post("/api/reports/issue", json={"run_id": "h-1", "content": _content()}).json()
    b = client.post("/api/reports/issue", json={"run_id": "h-2", "content": _content()}).json()

    ha = client.get(f"/api/reports/{a['report_no']}/content").json()["content_hash"]
    hb = client.get(f"/api/reports/{b['report_no']}/content").json()["content_hash"]
    assert ha == hb

    c = client.post("/api/reports/issue",
                    json={"run_id": "h-3", "content": _content(accuracy=0.5)}).json()
    hc = client.get(f"/api/reports/{c['report_no']}/content").json()["content_hash"]
    assert hc != ha


def test_reissue_preserves_previous_version_snapshot(client):
    """[F-01][F-06] 정정 발급 후에도 이전 차수의 내용이 남아야 정정 전후를 대조할 수 있다."""
    issued = client.post(
        "/api/reports/issue", json={"run_id": "run-reissue", "content": _content(0.944)}
    ).json()
    no = issued["report_no"]

    r = client.post(
        f"/api/reports/{no}/reissue",
        json={"run_id": "run-reissue", "note": "지표 정정", "content": _content(0.955)},
    )
    assert r.status_code == 200, r.text
    assert r.json()["version"] == "v1.1"

    latest = client.get(f"/api/reports/{no}/content").json()
    assert latest["version"] == "v1.1"
    assert latest["content"]["kpiResults"][0]["value"] == 0.955

    prior = client.get(f"/api/reports/{no}/content", params={"version": "v1.0"}).json()
    assert prior["version"] == "v1.0"
    assert prior["content"]["kpiResults"][0]["value"] == 0.944


def test_issue_without_content_still_succeeds(client):
    """[F-01] content 는 선택 — 기존 클라이언트 호환(하위호환 계약)."""
    r = client.post("/api/reports/issue", json={"run_id": "run-nocontent"})
    assert r.status_code == 200
    no = r.json()["report_no"]

    got = client.get(f"/api/reports/{no}/content")
    assert got.status_code == 404  # 저장된 내용이 없다는 사실을 정직하게 알린다
    assert "내용" in got.json()["detail"]


def test_content_over_size_limit_is_rejected(client):
    """[F-01][G-03] 무인증 POST 로 임의 크기 JSON 이 DB 에 영구 저장되는 것을 막는다."""
    huge = {"blob": "x" * (2 * 1024 * 1024)}  # 2MB
    r = client.post("/api/reports/issue", json={"run_id": "run-huge", "content": huge})
    assert r.status_code == 422, r.text


def test_content_unknown_report_no_is_404(client):
    r = client.get("/api/reports/RPT-2099-9999/content")
    assert r.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# G-01 — 발급 시점 기관 스냅샷 / 무인가 PUT 제거
# ══════════════════════════════════════════════════════════════════════════════

def test_organization_is_frozen_at_issue_time(db_session, client):
    """[G-01] 발급 후 기관 정보를 바꿔도 이미 발급된 성적서의 기관 표기는 그대로여야 한다.

    종전에는 issuance_out() 이 조회할 때마다 현재 singleton 행을 조인해 조립했다.
    한 응답 안에서 issuer 는 발급 당시 값, organization 은 현재 값이라 두 필드가
    서로 다른 시점을 가리켰다.
    """
    issued = client.post("/api/reports/issue", json={"run_id": "run-org", "content": _content()}).json()
    no = issued["report_no"]
    original_org = issued["organization"]["org_name"]

    svc.update_organization(db_session, {"org_name": "변경된 기관", "evaluator": "다른 평가자"})

    after = client.get(f"/api/reports/{no}").json()
    assert after["organization"]["org_name"] == original_org, (
        "이미 발급된 성적서의 기관 표기가 소급 변경됐다"
    )
    assert after["organization"]["org_name"] != "변경된 기관"


def test_new_issue_uses_current_organization(db_session, client):
    """[G-01] 기관을 바꾼 뒤 새로 발급하면 새 기관이 찍힌다(동결이 갱신을 막지 않는다)."""
    svc.update_organization(db_session, {"org_name": "새 기관", "evaluator": "평가자"})
    issued = client.post("/api/reports/issue", json={"run_id": "run-neworg"}).json()
    assert issued["organization"]["org_name"] == "새 기관"


def test_put_organization_endpoint_is_removed(client):
    """[G-01] 무인가 기관 수정 엔드포인트는 더 이상 존재하지 않는다.

    curl 한 줄로 발급기관명을 바꾸면 이미 발급된 모든 성적서 표기가 함께 바뀌었다.
    프론트는 이 API 를 한 번도 호출하지 않으므로(issuanceApi.ts 전수 확인) 제거한다.
    """
    r = client.put("/api/organization", json={"org_name": "HACKED", "evaluator": "attacker"})
    assert r.status_code == 405, f"PUT 이 아직 살아 있다: {r.status_code}"

    # GET 은 그대로 동작해야 한다
    assert client.get("/api/organization").status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# G-02 — 재발급에 run_id 소지 증명 요구
# ══════════════════════════════════════════════════════════════════════════════

def test_reissue_requires_matching_run_id(client):
    """[G-02] 번호만 알아도 남의 성적서에 강제 재발급 이력을 남길 수 없어야 한다.

    report_no 는 RPT-{year}-{seq:04d} 로 전수 열거 가능하지만, run_id 는
    crypto.randomUUID 라 추측할 수 없다. 소지 증명으로 쓴다.
    """
    issued = client.post("/api/reports/issue", json={"run_id": "run-secret"}).json()
    no = issued["report_no"]

    r = client.post(
        f"/api/reports/{no}/reissue",
        json={"run_id": "run-guessed", "note": "attacker forced reissue"},
    )
    assert r.status_code == 403, r.text

    # 버전과 이력이 오염되지 않았는지 확인
    after = client.get(f"/api/reports/{no}").json()
    assert after["version"] == "v1.0"
    assert len(after["history"]) == 1
    assert all("attacker" not in (h["note"] or "") for h in after["history"])


def test_reissue_with_correct_run_id_succeeds(client):
    """[G-02] 정당한 소지자는 그대로 재발급할 수 있다."""
    issued = client.post("/api/reports/issue", json={"run_id": "run-owner"}).json()
    no = issued["report_no"]

    r = client.post(f"/api/reports/{no}/reissue", json={"run_id": "run-owner", "note": "정정"})
    assert r.status_code == 200
    assert r.json()["version"] == "v1.1"
