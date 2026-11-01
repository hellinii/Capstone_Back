"""tests/test_validation_lone_surrogate.py — 요청 JSON 의 짝 없는 대리 글자(JSON "\\ud800")가 500 이 아니라 422 가 되는지(app.main 처리기).

FastAPI 기본 422 처리기는 오류 목록이 되돌려 싣는 입력을 UTF-8 로 만들다 UnicodeEncodeError → 500 이었다(심판 라운드 J10 인계,
앱 전역). 발급 API 와 서술 작업 API 두 곳에서 본다. 보통 422 응답 바이트는 그대로인지도 본다.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)  # 처리기 밖 예외도 500 응답으로 보이게


def _post(path: str, raw: bytes):
    return client.post(path, content=raw, headers={"content-type": "application/json"})


def test_issue_api_lone_surrogate_is_422():
    resp = _post("/api/reports/issue", b'{"run_id": "r", "model_name": "m", "content": {"contentSchema": 4, "x": "\\ud800"}}')
    assert resp.status_code == 422, resp.text
    detail = json.loads(resp.content)["detail"]
    assert any(e["loc"][:2] == ["body", "content"] for e in detail), detail


def test_narrative_jobs_api_lone_surrogate_is_422():
    resp = _post("/api/narrative/jobs", b'{"x": "\\ud800"}')
    assert resp.status_code == 422, resp.text
    detail = json.loads(resp.content)["detail"]
    assert detail and detail[0]["input"] == {"x": "\ud800"}  # 입력은 같은 JSON 값으로(대리 글자는 이스케이프로) 되돌아온다


def test_ordinary_422_keeps_utf8_body():
    resp = _post("/api/narrative/jobs", '{"x": "한글"}'.encode())
    assert resp.status_code == 422
    assert "한글".encode() in resp.content  # 보통 응답은 기본 처리기와 같은 UTF-8 바이트(이스케이프 아님)
