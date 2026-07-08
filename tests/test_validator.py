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
        selected_tcs=["TC1"],
    ))
    assert resp.is_valid is False
    assert any(e.code == "SAME_COLUMN_TRUE_PRED" for e in resp.errors)


def test_validate_mapping_ok_when_distinct():
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=TaskType.binary,
        column_mappings=[_cm("y", ColumnRole.y_true), _cm("p", ColumnRole.y_pred)],
        selected_tcs=["TC1"],
    ))
    assert not any(e.code in ("SAME_COLUMN_TRUE_PRED", "COLUMN_MULTIPLE_ROLES") for e in resp.errors)


# ── preprocess / engine 백스톱 ──────────────────────────────────────────────

def test_preprocess_rejects_same_true_pred_column():
    df = pd.DataFrame({"label": [1, 0, 1, 0]})
    mappings = [{"column": "label", "role": "y_true"}, {"column": "label", "role": "y_pred"}]
    with pytest.raises(ValueError, match="동일한 컬럼"):
        preprocess_data(df, mappings, "binary")


def test_engine_rejects_same_true_pred_column():
    df = pd.DataFrame({"label": [1, 0, 1, 0], "other": [1, 1, 0, 0]})
    mappings = [{"column": "label", "role": "y_true"}, {"column": "label", "role": "y_pred"}]
    result = run_evaluation(df, mappings, "binary", ["TC1"])
    assert "error" in result           # 전처리 백스톱이 잡아 error 반환
    assert result.get("TC1") != 1.0    # 가짜 100% 아님
