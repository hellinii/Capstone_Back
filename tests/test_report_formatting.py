"""tests/test_report_formatting.py — generate_report 의 성공/실패 분류 검증.

ISSUES.md C-07 — `report.py` 가 지표 결과를 "dict 이고 'error' 키가 있는가"로만
분류한다. dict 를 정상 반환하는 지표(M21 혼동행렬, M22 클래스별 지표)가 에러 표현과
같은 네임스페이스를 쓰기 때문에, 데이터에 'error' 라는 이름의 클래스가 있으면
정상 계산된 M22 가 실패로 분류되고 그 값이 실패 사유 문자열 자리에 들어간다.
"""
import pandas as pd
import pytest

from app.evaluation.engine import evaluate
from app.evaluation.metrics import common
from app.evaluation.report import generate_report


def test_class_named_error_does_not_make_m22_a_failure():
    """[C-07] 'error' 라는 클래스명이 있어도 M22 는 성공으로 분류되어야 한다.

    classification_report 는 클래스명을 최상위 키로 쓰므로, 'error' 클래스가 있으면
    반환 dict 에 'error' 키가 생긴다. 이것이 에러 표현과 구분되지 않는다.
    """
    df = pd.DataFrame({
        "t": ["error", "error", "ok", "ok"],
        "p": ["error", "ok", "ok", "ok"],
    })
    mapping = {"y_true": "t", "y_pred": "p", "_task_type": "multiclass"}

    m22 = common.calculate_class_metrics(df, mapping)
    assert "error" in m22, "픽스처 전제: 'error' 클래스가 보고서 키로 등장해야 한다"

    out = generate_report({"M22": m22})

    assert "M22" not in out["failed_metrics"], (
        "정상 계산된 M22 가 실패로 분류됐다 — 실패 사유 자리에 들어간 값: "
        f"{out['failed_metrics'].get('M22')!r}"
    )
    assert out["success_metrics"]["M22"] == m22


def test_real_metric_failure_is_still_classified_as_failed():
    """[C-07] 진짜 실패는 여전히 failed_metrics 로 가야 한다(수정이 분류를 망치지 않는다)."""
    df = pd.DataFrame({"t": ["A", "B"], "p": ["A", "B"]})
    results = evaluate(
        df,
        [{"role": "y_true", "column": "t"}, {"role": "y_pred", "column": "p"}],
        task_type="multiclass",
        selected_metric_ids=["M99"],  # 존재하지 않는 지표
    )
    report = generate_report(results)

    assert "M99" in report["failed_metrics"]
    assert isinstance(report["failed_metrics"]["M99"], str)
    assert "M99" not in report["success_metrics"]


def test_unsupported_metric_for_task_is_failed_with_reason():
    """[C-07] task 미지원 지표도 사유 문자열과 함께 실패로 분류된다."""
    df = pd.DataFrame({"t": [0, 1], "p": [0, 1]})
    results = evaluate(
        df,
        [{"role": "y_true", "column": "t"}, {"role": "y_pred", "column": "p"}],
        task_type="binary",
        selected_metric_ids=["M15"],  # multilabel 전용
    )
    report = generate_report(results)

    assert "M15" in report["failed_metrics"]
    assert "지원하지 않는" in report["failed_metrics"]["M15"]


def test_confusion_matrix_dict_is_success():
    """[C-07] M21 도 dict 를 정상 반환한다 — 실패로 새면 안 된다."""
    df = pd.DataFrame({"t": ["A", "B"], "p": ["A", "B"]})
    m21 = common.calculate_confusion_matrix(df, {"y_true": "t", "y_pred": "p"})
    out = generate_report({"M21": m21})

    assert out["failed_metrics"] == {}
    assert out["success_metrics"]["M21"]["type"] == "multiclass_or_binary"


def test_underscore_keys_are_dropped_from_both_buckets():
    """전처리 메타데이터(_ 접두) 키는 어느 쪽에도 들어가지 않는다(기존 동작 보존)."""
    out = generate_report({"M1": 0.9, "_dropped_rows": 3})

    assert out["success_metrics"] == {"M1": 0.9}
    assert out["failed_metrics"] == {}
