"""[G-04a][G-08] 업로드 경계 가드 — 크기 상한과 거절 응답 통일.

세 업로드 라우터(analyze-columns / validate-data / evaluate)는 같은 종류의 잘못된
업로드를 같은 상태코드·같은 문구로 거절해야 하고(G-08), 상한을 넘는 파일은 pandas
파싱에 들어가기 전에 차단되어야 한다(G-04a).

종전 상태(이 테스트가 red 로 잡는 것):
- 크기 상한이 세 라우터 어디에도 없어 26MB CSV 가 200 으로 통과했다(실측).
- 확장자 사전 검사가 analyze-columns 에만 있어 .xlsx 가 analyze 는 400,
  validate/evaluate 는 422 로 거절됐다.
"""
import io
import json

import pytest
from fastapi.testclient import TestClient

from app.core import upload as upload_mod
from app.main import app
from router_cases import request_json

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_llm_client():
    """analyze-columns 는 app.state.openai_client 를 읽는다. lifespan 을 띄우지 않으므로
    규칙 폴백 경로(None)를 직접 세운다 — 이 스위트의 관심사는 업로드 경계뿐이다."""
    app.state.openai_client = None

# 세 업로드 라우터 — (경로, 추가 폼 필드)
_EVAL_DATA = request_json("binary")
UPLOAD_ROUTES = [
    ("/api/analyze-columns", {"task_type": "binary"}),
    ("/api/validate-data", {"data": _EVAL_DATA}),
    ("/api/evaluate", {"data": _EVAL_DATA}),
]

_VALID_CSV = b"id,y_true,y_pred\n1,1,1\n2,0,0\n"


def _post(path: str, form: dict, filename: str, content: bytes):
    return client.post(
        path,
        files={"file": (filename, io.BytesIO(content), "text/csv")},
        data=form,
    )


@pytest.mark.parametrize("path,form", UPLOAD_ROUTES)
def test_oversized_upload_is_rejected(path, form, monkeypatch):
    """[G-04a] 상한을 넘는 업로드는 413 으로 거절된다 (종전: 200)."""
    monkeypatch.setattr(upload_mod, "MAX_UPLOAD_BYTES", 1024)
    big = b"id,y_true,y_pred\n" + b"1,1,1\n" * 500  # 3KB > 1KB 상한
    r = _post(path, form, "big.csv", big)
    assert r.status_code == 413, f"{path}: {r.status_code} {r.text[:200]}"


def test_oversized_upload_rejected_at_real_limit():
    """[G-04a] 실제 설정된 상한(20 MiB)에서도 차단된다 — 상수 자체가 유효한지 확인."""
    assert upload_mod.MAX_UPLOAD_BYTES == 20 * 1024 * 1024
    row = b"1,1,1\n"
    big = b"id,y_true,y_pred\n" + row * ((upload_mod.MAX_UPLOAD_BYTES // len(row)) + 1)
    assert len(big) > upload_mod.MAX_UPLOAD_BYTES
    r = _post("/api/evaluate", {"data": _EVAL_DATA}, "big.csv", big)
    assert r.status_code == 413, r.text[:200]


@pytest.mark.parametrize("path,form", UPLOAD_ROUTES)
def test_upload_just_under_limit_passes_size_gate(path, form, monkeypatch):
    """[G-04a] 상한 이하 업로드는 크기 게이트를 통과한다 (413 이 아니어야 한다)."""
    monkeypatch.setattr(upload_mod, "MAX_UPLOAD_BYTES", len(_VALID_CSV))
    r = _post(path, form, "ok.csv", _VALID_CSV)
    assert r.status_code != 413, f"{path}: 경계값이 잘못 거절됐다 — {r.text[:200]}"


def test_bad_extension_rejected_identically_across_routers():
    """[G-08] .xlsx 는 세 라우터에서 같은 상태코드·같은 문구로 거절된다.

    종전: analyze 400 '지원하지 않는 파일 형식입니다: .xlsx …'
          validate/evaluate 422 '파일 파싱 실패: 지원하지 않는 파일 형식: .xlsx …'
    """
    responses = {
        path: _post(path, form, "data.xlsx", _VALID_CSV)
        for path, form in UPLOAD_ROUTES
    }
    codes = {p: r.status_code for p, r in responses.items()}
    details = {p: r.json().get("detail") for p, r in responses.items()}
    assert len(set(codes.values())) == 1, f"상태코드가 갈린다: {codes}"
    assert len(set(details.values())) == 1, f"문구가 갈린다: {details}"
    assert next(iter(codes.values())) == 400


def test_missing_extension_rejected_identically_across_routers():
    """[G-08] 확장자 없는 파일도 세 라우터가 같게 거절한다."""
    responses = {
        path: _post(path, form, "noext", _VALID_CSV)
        for path, form in UPLOAD_ROUTES
    }
    codes = {p: r.status_code for p, r in responses.items()}
    details = {p: r.json().get("detail") for p, r in responses.items()}
    assert len(set(codes.values())) == 1, f"상태코드가 갈린다: {codes}"
    assert len(set(details.values())) == 1, f"문구가 갈린다: {details}"


def test_empty_file_rejected_identically_across_routers():
    """[G-08] 빈 파일 거절은 종전에도 일치했다 — 통합 과정에서 깨지지 않게 고정한다."""
    responses = {
        path: _post(path, form, "empty.csv", b"")
        for path, form in UPLOAD_ROUTES
    }
    codes = {p: r.status_code for p, r in responses.items()}
    details = {p: r.json().get("detail") for p, r in responses.items()}
    assert set(codes.values()) == {400}, codes
    assert len(set(details.values())) == 1, details
