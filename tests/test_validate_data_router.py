"""tests/test_validate_data_router.py — POST /api/validate-data 골든 characterization 테스트.

validate_data 는 490줄짜리 단일 라우터 함수(스파게티 1위)로, 엔드포인트 테스트가 없다.
PR-C(validation_service.py + validation_checks.py 로 분해)의 안전망으로, 현재 응답
(ValidateDataResponse: validation_details 항목 name/status/group, execution_summary,
selected_metric_ids)을 골든으로 고정한다.

- clean 데이터셋: 정상(pass) 경로.
- _with_errors 데이터셋: 결측/중복/클래스불일치 등 error·warning 분기까지 커버.
검증 로직은 LLM 을 쓰지 않으므로 결정론적이다.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from golden_utils import assert_golden
from router_cases import CASES, request_json

client = TestClient(app)


def _post_validate(file_path, filename, data_json):
    with open(file_path, "rb") as f:
        return client.post(
            "/api/validate-data",
            files={"file": (filename, f.read(), "text/csv")},
            data={"data": data_json},
        )


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_validate_data_clean_golden(task):
    c = CASES[task]
    resp = _post_validate(c["csv"], c["csv"].name, request_json(task))
    assert resp.status_code == 200, resp.text
    assert_golden(f"validate_{task}_clean", resp.json())


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_validate_data_errors_golden(task):
    """오류 포함 데이터셋 — error/warning 검증 항목 분기 characterization."""
    c = CASES[task]
    resp = _post_validate(c["csv_errors"], c["csv_errors"].name, request_json(task))
    assert resp.status_code == 200, resp.text
    assert_golden(f"validate_{task}_errors", resp.json())
