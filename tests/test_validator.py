"""D5b — 컬럼 단위 상호배타 검증 (정답=예측 동일 컬럼 = 가짜 100% 차단).

confirm-mapping(validate_mapping) / evaluate·validate-data(find_column_conflicts) /
preprocess(백스톱) 세 방어 지점을 모두 커버한다.
"""
import pandas as pd
import pytest

from app.core.schemas import ColumnMapping, ColumnRole, TaskType
from app.analysis.schemas import ConfirmMappingRequest
from app.analysis.validator import find_column_conflicts, validate_mapping
from app.evaluation.preprocessor import preprocess_data
from app.evaluation.engine import evaluate as run_evaluation


def _cm(column, role):
    return ColumnMapping(column=column, role=role)


# ── find_column_conflicts (공유 헬퍼) ───────────────────────────────────────

def test_same_column_true_pred_binary():
    errors = find_column_conflicts(
        [_cm("label", ColumnRole.y_true), _cm("label", ColumnRole.y_pred)],
        TaskType.binary,
    )
    assert any(e.code == "SAME_COLUMN_TRUE_PRED" for e in errors)


def test_same_column_true_pred_multilabel():
    errors = find_column_conflicts(
        [_cm("tags", ColumnRole.true_labels), _cm("tags", ColumnRole.pred_labels)],
        TaskType.multilabel,
    )
    assert any(e.code == "SAME_COLUMN_TRUE_PRED" for e in errors)


def test_column_multiple_roles_non_true_pred():
    errors = find_column_conflicts(
        [_cm("id", ColumnRole.sample_id), _cm("id", ColumnRole.y_true),
         _cm("pred", ColumnRole.y_pred)],
        TaskType.binary,
    )
    codes = {e.code for e in errors}
    assert "COLUMN_MULTIPLE_ROLES" in codes
    assert "SAME_COLUMN_TRUE_PRED" not in codes


def test_distinct_columns_no_conflict():
    assert find_column_conflicts(
        [_cm("y", ColumnRole.y_true), _cm("p", ColumnRole.y_pred)],
        TaskType.binary,
    ) == []


def test_multiple_ignore_columns_allowed():
    assert find_column_conflicts(
        [_cm("a", ColumnRole.ignore), _cm("b", ColumnRole.ignore),
         _cm("y", ColumnRole.y_true), _cm("p", ColumnRole.y_pred)],
        TaskType.binary,
    ) == []


# ── validate_mapping (confirm-mapping 경로) ─────────────────────────────────

def test_validate_mapping_flags_same_column():
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=TaskType.binary,
        column_mappings=[_cm("label", ColumnRole.y_true), _cm("label", ColumnRole.y_pred)],
        selected_metric_ids=["M1"],
    ))
    assert resp.is_valid is False
    assert any(e.code == "SAME_COLUMN_TRUE_PRED" for e in resp.errors)


def test_validate_mapping_ok_when_distinct():
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=TaskType.binary,
        column_mappings=[_cm("y", ColumnRole.y_true), _cm("p", ColumnRole.y_pred)],
        selected_metric_ids=["M1"],
    ))
    assert not any(e.code in ("SAME_COLUMN_TRUE_PRED", "COLUMN_MULTIPLE_ROLES") for e in resp.errors)


# ── 예측 역할 필수 여부는 선택한 지표에 따라 달라진다 ──────────────────────────
# M23(Imbalance Ratio)은 정답 분포만으로 계산되므로 예측 컬럼 없이도 진행할 수 있어야 한다.
# 반대로 지표를 명시하지 않은 호출자에게는 종전의 엄격한 규칙을 유지해야 한다.

@pytest.mark.parametrize("task_type,truth_role", [
    (TaskType.multiclass, ColumnRole.y_true),
    (TaskType.multilabel, ColumnRole.true_labels),
    (TaskType.binary, ColumnRole.y_true),
])
def test_m23_only_selection_does_not_require_pred_role(task_type, truth_role):
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=task_type,
        column_mappings=[_cm("id", ColumnRole.sample_id), _cm("y", truth_role)],
        selected_metric_ids=["M23"],
    ))
    assert resp.is_valid is True, [e.message for e in resp.errors]
    assert "M23" in resp.available_metric_ids


@pytest.mark.parametrize("task_type,truth_role,expected_code", [
    (TaskType.multiclass, ColumnRole.y_true, "MISSING_REQUIRED"),
    (TaskType.multilabel, ColumnRole.true_labels, "MISSING_REQUIRED"),
    (TaskType.binary, ColumnRole.y_true, "MISSING_PRED_OR_SCORE"),
])
def test_empty_selection_keeps_strict_pred_requirement(task_type, truth_role, expected_code):
    """지표 미지정(레거시 호출자)이면 무엇이 필요한지 알 수 없으므로 예측 역할을 계속 요구한다."""
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=task_type,
        column_mappings=[_cm("id", ColumnRole.sample_id), _cm("y", truth_role)],
        selected_metric_ids=[],
    ))
    assert resp.is_valid is False
    assert any(e.code == expected_code for e in resp.errors)


def test_pred_using_metric_still_requires_pred_role():
    """예측을 쓰는 지표가 하나라도 섞이면 예측 역할은 여전히 필수다."""
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=TaskType.multiclass,
        column_mappings=[_cm("id", ColumnRole.sample_id), _cm("y", ColumnRole.y_true)],
        selected_metric_ids=["M23", "M1"],
    ))
    assert resp.is_valid is False
    assert any(e.code == "MISSING_REQUIRED" and "y_pred" in e.message for e in resp.errors)


def test_missing_truth_role_still_reported_first():
    """정답 역할 누락 메시지가 예측 역할 메시지보다 먼저 온다(프론트가 순서대로 이어 붙여 노출)."""
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=TaskType.multiclass,
        column_mappings=[_cm("id", ColumnRole.sample_id)],
        selected_metric_ids=["M1"],
    ))
    codes = [(e.code, e.message) for e in resp.errors]
    assert resp.is_valid is False
    assert "y_true" in codes[0][1]
    assert "y_pred" in codes[1][1]


# ── preprocess / engine 백스톱 ──────────────────────────────────────────────

def test_preprocess_rejects_same_true_pred_column():
    df = pd.DataFrame({"label": [1, 0, 1, 0]})
    mappings = [{"column": "label", "role": "y_true"}, {"column": "label", "role": "y_pred"}]
    with pytest.raises(ValueError, match="동일한 컬럼"):
        preprocess_data(df, mappings, "binary")


def test_engine_rejects_same_true_pred_column():
    df = pd.DataFrame({"label": [1, 0, 1, 0], "other": [1, 1, 0, 0]})
    mappings = [{"column": "label", "role": "y_true"}, {"column": "label", "role": "y_pred"}]
    result = run_evaluation(df, mappings, "binary", ["M1"])
    assert "error" in result           # 전처리 백스톱이 잡아 error 반환
    assert result.get("M1") != 1.0    # 가짜 100% 아님
