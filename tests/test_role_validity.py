"""tests/test_role_validity.py — 역할 유효성 검사(SPEC §4 Error 3건).

ISSUES.md A-10 (2026-09-07 ★확정된 제품 결정 3b).

SPEC §4 는 task_type 에 맞지 않는 역할을 **진행 차단 Error** 로 규정한다.

  - Binary 인데 `prob_class_*`(prob_per_class) 가 매핑됨
  - Multiclass 인데 `score`(score_positive) 가 매핑됨
  - Multilabel 인데 `score` 또는 `prob_class_*` 가 매핑됨

이 검사는 어디에도 없었다. `validator.py` 의 docstring 이 "1. 역할 유효성" 을 한다고
적고 있었으나 코드에는 그 단계가 없어 **문서가 거짓**이었다.

피해는 대장 서술('그 컬럼은 무시된다')보다 크다 — `frame.required_columns` 가 ignore 가
아닌 모든 역할의 컬럼을 dropna 대상에 넣으므로, 엉뚱한 역할로 매핑된 컬럼의 결측이
평가 표본을 **조용히 깎는다.**
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.core.schemas import (
    ColumnMapping,
    ColumnRole,
    DataMetadata,
    TaskType,
    VALID_ROLES_BY_TASK,
)
from app.analysis.schemas import ConfirmMappingRequest
from app.analysis.validator import validate_mapping
from app.evaluation.schemas import EvaluateRequest
from app.main import app

client = TestClient(app)


def _cm(column, role):
    return ColumnMapping(column=column, role=role)


# SPEC §4 가 열거한 역할 불일치 3건 + 일반화로 함께 막히는 조합.
_MISMATCHES = [
    (TaskType.binary, ColumnRole.y_true, ColumnRole.prob_per_class),
    (TaskType.binary, ColumnRole.y_true, ColumnRole.score_per_label),
    (TaskType.binary, ColumnRole.y_true, ColumnRole.true_labels),
    (TaskType.multiclass, ColumnRole.y_true, ColumnRole.score_positive),
    (TaskType.multiclass, ColumnRole.y_true, ColumnRole.score_per_label),
    (TaskType.multiclass, ColumnRole.y_true, ColumnRole.pred_labels),
    (TaskType.multilabel, ColumnRole.true_labels, ColumnRole.score_positive),
    (TaskType.multilabel, ColumnRole.true_labels, ColumnRole.prob_per_class),
    (TaskType.multilabel, ColumnRole.true_labels, ColumnRole.y_pred),
]


@pytest.mark.parametrize("task, truth_role, bad_role", _MISMATCHES)
def test_confirm_mapping_blocks_role_not_valid_for_task(task, truth_role, bad_role):
    """[A-10] 허용되지 않는 역할은 진행을 차단한다."""
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=task,
        column_mappings=[_cm("t", truth_role), _cm("x", bad_role)],
        selected_metric_ids=["M23"],
    ))
    assert not resp.is_valid
    codes = [e.code for e in resp.errors]
    assert "INVALID_ROLE_FOR_TASK" in codes, codes
    message = next(e.message for e in resp.errors if e.code == "INVALID_ROLE_FOR_TASK")
    assert bad_role.value in message and "x" in message


@pytest.mark.parametrize("task", list(TaskType))
def test_every_valid_role_passes_the_check(task):
    """[A-10] 허용 역할은 하나도 걸리지 않는다(과잉 차단 방지)."""
    mappings = [
        _cm(f"c{i}", role) for i, role in enumerate(VALID_ROLES_BY_TASK[task])
    ]
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=task, column_mappings=mappings, selected_metric_ids=["M23"],
    ))
    assert not [e for e in resp.errors if e.code == "INVALID_ROLE_FOR_TASK"], resp.errors


def test_invalid_role_is_reported_before_duplicate_role():
    """[A-10] 잘못된 역할이 DUPLICATE_ROLE 로 오보고되지 않는다.

    두 컬럼에 같은 '허용되지 않는 역할'을 주면, 사용자가 고쳐야 할 것은 중복이 아니라
    역할 자체다. 두 오류가 함께 나오더라도 역할 오류가 먼저 와야 안내가 맞는다.
    """
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=TaskType.multiclass,
        column_mappings=[
            _cm("t", ColumnRole.y_true),
            _cm("a", ColumnRole.score_positive),
            _cm("b", ColumnRole.score_positive),
        ],
        selected_metric_ids=["M23"],
    ))
    codes = [e.code for e in resp.errors]
    assert codes[0] == "INVALID_ROLE_FOR_TASK", codes


# ── API 직접 호출 백스톱 (A-10 의 본체) ────────────────────────────────────

_CSV = "t,x\ncat,0.9\ndog,0.1\n"


def _mismatched_evaluate_payload():
    return EvaluateRequest(
        task_type=TaskType.multiclass,
        column_mappings=[
            _cm("t", ColumnRole.y_true),
            _cm("x", ColumnRole.score_positive),   # multiclass 에 허용되지 않음
        ],
        selected_metric_ids=["M23"],
        metadata=DataMetadata(),
    ).model_dump_json()


def test_evaluate_rejects_mismatched_role_without_confirm_mapping():
    """[A-10] confirm-mapping 을 건너뛰고 직접 호출해도 통과하면 안 된다.

    종전에는 통과했고, 그 컬럼이 `required_columns` 에 들어가 결측이 평가 표본을
    조용히 깎았다. 프론트 드롭다운 제한은 UI 경로만 막는다.
    """
    resp = client.post(
        "/api/evaluate",
        files={"file": ("d.csv", _CSV.encode("utf-8"), "text/csv")},
        data={"data": _mismatched_evaluate_payload()},
    )
    assert resp.status_code == 400, resp.text
    assert "score_positive" in resp.json()["detail"]


def test_validate_data_reports_mismatched_role_as_error():
    """[A-10] 검증은 HTTP 상태가 아니라 error_count 로 보고한다(프론트 게이트가 그것을 본다).

    검증이 이 항목을 잡지 못하면 사용자는 6단계를 통과한 뒤 평가에서 막힌다.
    """
    resp = client.post(
        "/api/validate-data",
        files={"file": ("d.csv", _CSV.encode("utf-8"), "text/csv")},
        data={"data": _mismatched_evaluate_payload()},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["error_count"] >= 1
    item = next(i for i in body["validation_details"] if i["name"] == "Invalid role for task type")
    assert item["status"] == "error"
    assert "score_positive" in item["result"]


def test_valid_roles_still_pass_the_api_backstop():
    """백스톱이 정상 요청을 막지 않는다.

    M23 은 정답 분포만으로 계산된다 — 확률이 매핑돼 있어도 예측을 파생할 이유가 없고,
    파생을 시도하면 컬럼명에서 클래스를 확정하지 못해 400 이 된다. 선행 라운드가 연
    '정답 컬럼만으로 진행' 경로를 결정 1 이 다시 닫으면 안 된다.
    """
    payload = EvaluateRequest(
        task_type=TaskType.multiclass,
        column_mappings=[
            _cm("t", ColumnRole.y_true),
            _cm("x", ColumnRole.prob_per_class),
        ],
        selected_metric_ids=["M23"],
        metadata=DataMetadata(),
    ).model_dump_json()
    resp = client.post(
        "/api/evaluate",
        files={"file": ("d.csv", _CSV.encode("utf-8"), "text/csv")},
        data={"data": payload},
    )
    assert resp.status_code == 200, resp.text


# ── D-17: 필수 컬럼 목록의 순서가 결정적이다 ───────────────────────────────

def test_required_column_order_is_deterministic():
    """[D-17] `list(set(...))` 이 남아 있으면 오류 문구가 실행마다 달라진다.

    검증 응답의 'Missing required column' 결과 문자열이 그 순서를 그대로 인쇄한다.
    """
    from app.evaluation.frame import required_columns

    mappings = [
        {"column": "zz", "role": "y_true"},
        {"column": "aa", "role": "y_pred"},
        {"column": "mm", "role": "sample_id"},
        {"column": "nn", "role": "ignore"},
    ]
    assert required_columns(mappings) == ["zz", "aa", "mm"]


def test_missing_column_message_preserves_mapping_order():
    """[D-17·H-07] 누락 컬럼 안내가 매핑 순서를 보존한다(집합 순서가 아니라)."""
    payload = EvaluateRequest(
        task_type=TaskType.binary,
        column_mappings=[
            _cm("zz", ColumnRole.y_true),
            _cm("aa", ColumnRole.y_pred),
        ],
        selected_metric_ids=["M1"],
        metadata=DataMetadata(positive_class="1", negative_class="0"),
    ).model_dump_json()
    resp = client.post(
        "/api/validate-data",
        files={"file": ("d.csv", b"other\n1\n", "text/csv")},
        data={"data": payload},
    )
    assert resp.status_code == 200, resp.text
    item = next(
        i for i in resp.json()["validation_details"] if i["name"] == "Missing required column"
    )
    assert item["result"] == "zz, aa", item["result"]
