"""test_p4_evaluate_trace.py — 판 4 추적 정보(판 4 라운드 S1 ①): 파일 해시·크기, 평가 엔진 버전.

- 판 4 성적서 2.2 '파일 해시(SHA-256)'·'파일 크기'와 3.5 '평가 엔진 버전'의 출처다(대응표 20).
- `/api/evaluate` 응답 최상위 `source_file{sha256, size_bytes}`·`engine_version` 에만 싣는다. `environment` 안에 두지
  않는다 — 프론트가 그 객체를 판 3 저장본에 통째로 저장한다. 다른 두 업로드 라우터 응답에는 없다.
- 해시는 업로드한 **원본 바이트**로 계산한다(파싱한 표가 아니다).
"""
import hashlib
import io
import json
import re

from fastapi.testclient import TestClient

from app.evaluation import environment
from app.main import app
from router_cases import CASES, request_json

client = TestClient(app)


def _evaluate(content: bytes, name: str = "d.csv", task: str = "binary"):
    return client.post(
        "/api/evaluate",
        files={"file": (name, io.BytesIO(content), "text/csv")},
        data={"data": request_json(task)},
    )


def test_engine_version_is_the_code_constant():
    # 첫 값은 1.0.0(정해진 것 4). 평가 계산이 바뀌면 올린다 — 이 시험도 같이 고친다.
    assert environment.ENGINE_VERSION == "1.0.0"


def test_evaluate_response_carries_hash_size_and_engine_version():
    content = CASES["binary"]["csv"].read_bytes()
    resp = _evaluate(content)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source_file"] == {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    assert re.fullmatch(r"[0-9a-f]{64}", body["source_file"]["sha256"])
    assert body["engine_version"] == "1.0.0"
    # environment 안에는 넣지 않는다(판 3 저장본에 통째로 저장되는 객체).
    assert set(body["environment"]) == {"libraries", "evaluated_at"}
    assert set(body["environment"]["libraries"]) == {"python", "scikit-learn", "pandas", "numpy"}


def test_hash_is_of_the_uploaded_bytes_not_the_parsed_table():
    """같은 표라도 바이트가 다르면(줄 끝 CRLF) 해시·크기가 다르다 — 파싱한 값이 아니라 원본 바이트."""
    lf = CASES["binary"]["csv"].read_bytes().replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    a, b = _evaluate(lf).json(), _evaluate(crlf).json()
    assert a["results"]["success_metrics"]["M1"] == b["results"]["success_metrics"]["M1"]
    assert a["source_file"]["sha256"] == hashlib.sha256(lf).hexdigest()
    assert b["source_file"]["sha256"] == hashlib.sha256(crlf).hexdigest()
    assert a["source_file"]["size_bytes"] == len(lf)
    assert b["source_file"]["size_bytes"] == len(crlf)
    assert a["source_file"]["sha256"] != b["source_file"]["sha256"]


def test_source_file_info_known_value():
    assert environment.source_file_info(b"") == {
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "size_bytes": 0,
    }
    assert environment.source_file_info(b"abc") == {
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "size_bytes": 3,
    }


def test_other_upload_routes_do_not_carry_trace_fields():
    c = CASES["binary"]
    content = c["csv"].read_bytes()
    validate = client.post(
        "/api/validate-data",
        files={"file": (c["csv"].name, content, "text/csv")},
        data={"data": request_json("binary")},
    )
    assert validate.status_code == 200, validate.text
    prev = getattr(app.state, "openai_client", None)
    app.state.openai_client = None  # LLM 없이 폴백 매핑으로 응답한다
    try:
        analyze = client.post(
            "/api/analyze-columns", data={"task_type": "binary"}, files={"file": (c["csv"].name, content, "text/csv")}
        )
    finally:
        app.state.openai_client = prev
    assert analyze.status_code == 200, analyze.text
    for resp in (validate, analyze):
        text = json.dumps(resp.json())
        assert "source_file" not in text
        assert "engine_version" not in text
        assert "size_bytes" not in text


def test_openapi_schema_has_trace_fields():
    schemas = app.openapi()["components"]["schemas"]
    props = schemas["EvaluateResponse"]["properties"]
    assert {"source_file", "engine_version"} <= set(props)
    assert set(schemas["SourceFileInfo"]["properties"]) == {"sha256", "size_bytes"}
