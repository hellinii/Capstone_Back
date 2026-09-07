"""[G-07] 휘발성 DB 로의 조용한 강등 가드 + /health 진단 필드.

두 가지가 한 커밋에 있는 이유:

1. `DATABASE_URL` 이 없으면 database.py 가 예외 없이 로컬 SQLite 로 강등하고 기동을
   계속한다. 그 상태에서는 채번 시퀀스가 재시작마다 초기화돼 **성적서 번호가 중복
   발급**된다. 그런데 가드를 무조건 켜면 위험하다 — 지금 프로덕션이 SQLite 로 돌고
   있다면 켜는 순간 '조용히 열화된 채 살아 있는 서비스'가 '기동 실패로 죽은 서비스'가
   된다. 코드만 봐서는 어느 경우인지 알 수 없다.
2. 그래서 가드는 **기본 꺼짐**(REQUIRE_PERSISTENT_DB=1 일 때만 동작)으로 넣고,
   켜도 되는지 판단할 관측을 `/health` 진단으로 자가 조달한다. 진단 자체도
   환경변수(DIAG=1) 뒤에 둔다 — 무인증 공개 엔드포인트이기 때문이다.

같은 진단이 다음 라운드 레이트리밋의 선행 관측(프록시 뒤 client.host 실제 값)도
함께 조달한다.
"""
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from app.core.database import assert_persistent_backend
from app.main import app

client = TestClient(app)


# ── 가드 (순수 함수) ───────────────────────────────────────────────────────

def test_guard_rejects_sqlite_when_required():
    """[G-07] REQUIRE_PERSISTENT_DB 가 켜져 있으면 SQLite 는 거부한다."""
    with pytest.raises(RuntimeError, match="REQUIRE_PERSISTENT_DB"):
        assert_persistent_backend("sqlite:///./x.db", require=True)


def test_guard_allows_postgres_when_required():
    """[G-07] 영속 백엔드는 통과시킨다."""
    assert_persistent_backend("postgresql://u:p@h/db", require=True)


def test_guard_is_off_by_default():
    """[G-07] 기본값에서는 SQLite 로도 기동한다 — 넣는 것만으로 배포가 바뀌면 안 된다."""
    assert_persistent_backend("sqlite:///./x.db", require=False)


def test_guard_is_wired_at_import_time():
    """[G-07] 순수 함수가 실제로 import 경로에 배선돼 있는지 별도 프로세스로 확인한다.

    같은 프로세스에서 reload 하면 예외로 모듈이 반쯤 초기화된 상태로 남아 뒤따르는
    테스트를 오염시킨다.
    """
    env = {**os.environ, "REQUIRE_PERSISTENT_DB": "1"}
    env.pop("DATABASE_URL", None)
    r = subprocess.run(
        [sys.executable, "-c", "import app.core.database"],
        env=env, capture_output=True, text=True, cwd=os.getcwd(),
    )
    assert r.returncode != 0, "가드가 배선되지 않았다 — SQLite 로 조용히 기동했다"
    assert "REQUIRE_PERSISTENT_DB" in r.stderr, r.stderr[-500:]


def test_startup_is_unchanged_without_the_flag():
    """[G-07] 플래그가 없으면 import 경로가 오늘과 동일해야 한다."""
    env = {**os.environ}
    env.pop("REQUIRE_PERSISTENT_DB", None)
    env.pop("DATABASE_URL", None)
    r = subprocess.run(
        [sys.executable, "-c", "import app.core.database"],
        env=env, capture_output=True, text=True, cwd=os.getcwd(),
    )
    assert r.returncode == 0, r.stderr[-500:]


# ── /health 진단 ──────────────────────────────────────────────────────────

def test_health_reports_persistence_without_diag(monkeypatch):
    """[G-07] DIAG 없이도 **영속 여부**는 보인다.

    종전에는 DIAG=1 을 켜야만 실태를 볼 수 있었다. 그것은 운영자가 특별한 절차를 밟아야
    조용한 강등을 안다는 뜻이었고, 1차 라운드가 성적서 원본을 DB 에 넣은 뒤로 휘발성
    DB 는 채번 중복만이 아니라 **발급된 성적서 자체를 잃는다**. 위험이 커졌으므로
    강등 사실은 평시에도 보여야 한다.

    진단 상세(client_host·forwarded_for)는 여전히 DIAG 뒤에 둔다 — 그쪽은 운영 관측용
    이지 상시 필요한 정보가 아니다.
    """
    monkeypatch.delenv("DIAG", raising=False)
    r = client.get("/health")
    assert r.status_code == 200

    body = r.json()
    assert body["status"] == "ok"
    assert body["db_backend"] in ("sqlite", "postgresql")
    assert isinstance(body["persistent"], bool)
    assert "diagnostics" not in body


def test_health_exposes_diagnostics_when_enabled(monkeypatch):
    """[G-07] DIAG=1 이면 관측에 필요한 값을 싣는다 — curl 한 줄로 조달할 수 있어야 한다."""
    monkeypatch.setenv("DIAG", "1")
    r = client.get("/health", headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.1"})
    assert r.status_code == 200
    diag = r.json().get("diagnostics")
    assert diag is not None, "DIAG=1 인데 진단이 없다"
    assert diag["forwarded_for"] == "203.0.113.7, 10.0.0.1"
    assert "client_host" in diag
