"""tests/test_dev_fake_llm_server.py — 가짜 LLM 개발 서버(scripts/dev_fake_llm_server.py), 심판 라운드 J5b(계획 14-7).

앱 import 가 환경변수(키·DATABASE_URL)를 읽으므로 실제 모양은 하위 프로세스에서 본다(이 시험 프로세스는 이미 앱을 import 했다).
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import dev_fake_llm_server as dev  # noqa: E402

_CHILD = r"""
import json, sys
from pathlib import Path
from fastapi.testclient import TestClient
from scripts.dev_fake_llm_server import build_app
app = build_app(sys.argv[1])
raw = json.loads(Path("tests/data/judge_dev/dev_binary.request.json").read_text(encoding="utf-8"))
ed3 = {**raw, "report_edition": 3}; ed3.pop("judge_material")
out = {}
with TestClient(app) as c:
    out["clients"] = [type(app.state.anthropic_client).__name__, type(app.state.openai_client).__name__]
    for tag, body in (("ed4", raw), ("ed3", ed3)):
        job = c.post("/api/narrative/jobs", json=body).json()["job_id"]
        view = c.get(f"/api/narrative/jobs/{job}").json()
        out[tag] = {"status": view["status"], "judge": (view["result"] or {}).get("judge"),
                    "sources": sorted(set((view["result"] or {}).get("section_sources", {}).values()))}
    out["anthropic_calls"] = app.state.anthropic_client.calls
    out["judge_calls"] = app.state.openai_client.calls
from app.core import database
out["db"] = database.DATABASE_URL
print("RESULT " + json.dumps(out, ensure_ascii=False))
"""


def _child(mode: str, env_extra: dict | None = None):
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", _CHILD, mode], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


def _result(proc) -> dict:
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, proc.stdout[-2000:] + proc.stderr[-3000:]
    return json.loads(line[len("RESULT "):])


def test_pass_mode_runs_edition4_to_passed_with_fake_clients_only():
    out = _result(_child("pass"))
    assert out["clients"] == ["FakeAnthropic", "FakeOpenAI"]
    assert out["ed4"]["status"] == "done" and out["ed4"]["judge"]["gate"] == "passed"
    assert out["ed3"]["judge"] is None and out["ed4"]["sources"] != ["fallback", "fixed"]  # AI(가짜) 서술이 실렸다
    assert out["anthropic_calls"] > 0 and out["judge_calls"] > 0
    assert out["db"].startswith("sqlite:///") and "dev_fake_llm_" in out["db"]


@pytest.mark.parametrize("mode, retry", [("error", True), ("refuse", False)])
def test_failure_modes_hold(mode, retry):
    j = _result(_child(mode))["ed4"]["judge"]
    assert (j["gate"], j["held_kind"], j["retry_allowed"]) == ("held", "judge_error", retry)


def test_refuses_a_non_temporary_database(tmp_path):
    proc = _child("pass", {"DATABASE_URL": f"sqlite:///{ROOT / 'data' / 'dev_fake_should_not_exist.db'}"})
    assert proc.returncode != 0 and "임시 sqlite" in (proc.stderr + proc.stdout)
    assert not (ROOT / "data" / "dev_fake_should_not_exist.db").exists()


def _pretend_temp_dir(monkeypatch, path) -> None:
    """TMPDIR 을 바꿔 gettempdir 가 다시 읽게 한다(캐시 tempfile.tempdir 를 비움 — monkeypatch 가 되돌린다)."""
    monkeypatch.setenv("TMPDIR", str(path))
    monkeypatch.setattr(tempfile, "tempdir", None)


def test_refuses_repo_database_even_when_the_clone_is_under_the_temp_dir(monkeypatch):
    """/tmp 아래 clone 흉내: 임시 폴더를 저장소의 부모로 두면 저장소 data/ 도 "임시 폴더 아래"가 된다 — 그래도 거부."""
    _pretend_temp_dir(monkeypatch, ROOT.parent)
    assert os.path.realpath(tempfile.gettempdir()) == os.path.realpath(ROOT.parent)
    with pytest.raises(SystemExit, match="임시 sqlite"):
        dev.temp_sqlite_only(f"sqlite:///{ROOT / 'data' / 'x.db'}")
    with pytest.raises(SystemExit, match="임시 sqlite"):
        dev.temp_sqlite_only(f"sqlite:///{ROOT / 'x.db'}")
    dev.temp_sqlite_only(f"sqlite:///{ROOT.parent / 's5e_not_the_repo' / 'dev.db'}")  # 저장소 밖 임시 경로는 받는다


def test_refuses_a_sibling_that_only_shares_the_temp_dir_prefix(monkeypatch, tmp_path):
    base = tmp_path / "t"
    base.mkdir()
    _pretend_temp_dir(monkeypatch, base)
    dev.temp_sqlite_only(f"sqlite:///{base / 'dev.db'}")
    with pytest.raises(SystemExit, match="임시 sqlite"):
        dev.temp_sqlite_only(f"sqlite:///{tmp_path / 'tx' / 'dev.db'}")  # /…/tx 는 /…/t 아래가 아니다


def test_refuses_postgres_url():
    with pytest.raises(SystemExit, match="임시 sqlite"):
        dev.temp_sqlite_only("postgresql://u:p@h/db")


def test_assert_all_fake_rejects_a_real_looking_client():
    class State:
        def __init__(self, **kw):
            self._state = kw
            for k, v in kw.items():
                setattr(self, k, v)

    dev.assert_all_fake(State(anthropic_client=dev.FakeAnthropic(), openai_client=dev.FakeOpenAI("pass")))
    with pytest.raises(SystemExit, match="openai_client"):
        dev.assert_all_fake(State(anthropic_client=dev.FakeAnthropic(), openai_client=SimpleNamespace()))
    with pytest.raises(SystemExit, match="클라이언트 없음"):
        dev.assert_all_fake(State())


def test_app_does_not_import_scripts():
    bad = []
    for path in (ROOT / "app").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(n == "scripts" or n.startswith("scripts.") for n in names):
                bad.append(str(path.relative_to(ROOT)))
    assert bad == []


def test_judge_delay_waits_before_each_judge_answer_and_is_bounded():
    """--judge-delay(심판 라운드 J6 스모크용) — 심판 호출마다 기다린 뒤 답하고, 범위 밖 값은 받지 않는다."""
    import asyncio
    import time

    from scripts.dev_fake_llm_server import FakeOpenAI

    fake = FakeOpenAI("error", 0.2)
    start = time.monotonic()
    with pytest.raises(Exception):
        asyncio.run(fake.responses.create(model="m", input=[]))
    assert time.monotonic() - start >= 0.2
    assert FakeOpenAI("pass").delay == 0.0
    for bad in (-1, 61):
        with pytest.raises(ValueError):
            FakeOpenAI("pass", bad)
