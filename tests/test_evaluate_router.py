"""tests/test_evaluate_router.py — POST /api/evaluate 골든 characterization 테스트.

evaluate 라우터는 현재 엔드포인트 단위 테스트가 없어, 계층화 리팩토링(PR-E: service.py
분리·preprocess 단계화·side_metrics 추출) 시 회귀 위험이 크다. 이 테스트가 현재 응답을
골든으로 고정해 "동작 불변"을 보증한다.

- binary/multiclass/multilabel 각 CSV 로 평가 → EvaluateResponse 전체를 골든과 비교.
- binary 는 JSON 파일 업로드 경로도 별도 고정(PR-B parsing.py 분리 안전망).
evaluate 는 LLM 을 쓰지 않으므로 완전 결정론적이다.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from golden_utils import assert_golden
from router_cases import CASES, request_json

client = TestClient(app)  # lifespan 미실행: evaluate 는 DB/LLM 불필요


def _post_evaluate(file_path, filename, content_type, data_json):
    with open(file_path, "rb") as f:
        return client.post(
            "/api/evaluate",
            files={"file": (filename, f.read(), content_type)},
            data={"data": data_json},
        )


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_evaluate_csv_golden(task):
    c = CASES[task]
    resp = _post_evaluate(c["csv"], c["csv"].name, "text/csv", request_json(task))
    assert resp.status_code == 200, resp.text
    assert_golden(f"evaluate_{task}_csv", resp.json())


def test_evaluate_binary_json_golden():
    """JSON 형식 업로드 경로 characterization (parse_file_content 의 JSON 분기 보호)."""
    c = CASES["binary"]
    resp = _post_evaluate(c["json"], c["json"].name, "application/json", request_json("binary"))
    assert resp.status_code == 200, resp.text
    assert_golden("evaluate_binary_json", resp.json())
