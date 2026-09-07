"""tests/test_volatile_db_visibility.py — 휘발성 DB 강등이 눈에 보인다.

ISSUES.md G-07 (부분 해결분의 잔여).

`DATABASE_URL` 이 없으면 예외 없이 로컬 SQLite 로 내려가고 기동을 계속한다.
하드 실패 가드(`REQUIRE_PERSISTENT_DB=1`)는 **기본 꺼짐**이고 그 판단은 옳다 —
프로덕션이 이미 SQLite 로 돌고 있다면 켜는 순간 '조용히 열화된 서비스'가 '기동 실패로
죽은 서비스'가 된다.

그러나 그 절차("DIAG=1 로 확인한 뒤 켠다")는 운영자가 **특별한 플래그를 켜야만**
실태를 볼 수 있다는 뜻이었다. 1차 라운드가 성적서 원본(JSON+해시)을 DB 에 넣은 뒤로
휘발성 DB 는 채번 중복만이 아니라 **발급된 성적서 자체를 잃는다.** 위험이 커졌으므로
강등 사실은 평시에도 보여야 한다.
"""
import logging

from fastapi.testclient import TestClient

from app.core.database import describe_backend
from app.main import app

client = TestClient(app)


def test_health_reports_persistence_without_a_diagnostic_flag():
    """DIAG=1 없이도 영속 여부를 알 수 있다 — 운영자가 매 헬스체크에서 본다."""
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert "persistent" in body
    assert isinstance(body["persistent"], bool)


def test_health_does_not_leak_the_connection_string():
    """백엔드 종류만 알린다 — 자격증명이 들어 있는 URL 은 노출하지 않는다."""
    body = client.get("/health").json()

    assert body.get("db_backend") in ("sqlite", "postgresql")
    assert "://" not in str(body)


def test_describe_backend_classifies_urls():
    assert describe_backend("sqlite:///./x.db") == ("sqlite", False)
    assert describe_backend("postgresql://u:p@h/db") == ("postgresql", True)
    assert describe_backend("postgresql+psycopg2://u:p@h/db") == ("postgresql", True)


def test_volatile_backend_logs_a_warning_at_import(caplog):
    """조용한 강등은 흔적을 남겨야 한다(G-06 과 같은 원칙)."""
    from app.core.database import warn_if_volatile

    with caplog.at_level(logging.WARNING):
        warn_if_volatile("sqlite:///./local.db")

    assert any("휘발성" in r.message or "sqlite" in r.message.lower() for r in caplog.records)


def test_persistent_backend_logs_nothing(caplog):
    from app.core.database import warn_if_volatile

    with caplog.at_level(logging.WARNING):
        warn_if_volatile("postgresql://u:p@h/db")

    assert not caplog.records
