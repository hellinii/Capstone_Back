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


# ── D-12: 헤더만 있는 빈 데이터셋 ─────────────────────────────────────────────

def _empty_csv_bytes(task: str) -> bytes:
    """해당 task 의 매핑이 요구하는 컬럼 헤더만 있고 데이터 행이 0개인 CSV."""
    import json
    cols = [m["column"] for m in json.loads(request_json(task))["column_mappings"]]
    return (",".join(cols) + "\n").encode()


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_empty_dataset_is_blocked_as_error(task):
    """[D-12] 헤더만 있는 파일은 검증 단계에서 error 로 막혀야 한다.

    현재는 error_count=0 으로 통과해 사용자가 6단계를 다 지난 뒤 /api/evaluate 가
    400 으로 실패한다. 그 시점에는 어디로 돌아가야 하는지 안내가 없고, 프론트
    게이트(DataValidation.tsx)는 error_count 만 보므로 '다음' 버튼이 열려 있다.
    """
    resp = client.post(
        "/api/validate-data",
        files={"file": (f"empty_{task}.csv", _empty_csv_bytes(task), "text/csv")},
        data={"data": request_json(task)},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["error_count"] >= 1, (
        f"빈 데이터셋이 오류 0건으로 통과했다: execution_summary={body['execution_summary']}"
    )
    names = [d["name"] for d in body["validation_details"] if d["status"] == "error"]
    assert any("row" in n.lower() or "empty" in n.lower() for n in names), (
        f"행 수 관련 오류 항목이 없다: {names}"
    )


def test_non_empty_dataset_still_passes_row_count_check():
    """[D-12] 정상 데이터셋은 행 수 검사에 걸리지 않는다(수정이 정상 경로를 막지 않는다)."""
    c = CASES["binary"]
    resp = _post_validate(c["csv"], c["csv"].name, request_json("binary"))
    assert resp.status_code == 200
    body = resp.json()
    row_errors = [
        d for d in body["validation_details"]
        if d["status"] == "error" and "row" in d["name"].lower()
    ]
    assert row_errors == []
