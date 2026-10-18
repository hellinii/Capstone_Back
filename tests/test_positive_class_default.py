"""양성 클래스 기본값 한 함수(M5) — 자동 판단·지표 계산·예측 파생이 같은 값을 쓴다.

종전: 모르는 두 문자열('cat','dog')이면 자동 판단(metadata)은 앞쪽 cat 을, 지표 계산(binary)과 예측
파생(preprocessor)은 뒤쪽 dog 를 양성으로 봐서 화면에 보인 양성과 성적서 수치의 양성이 달랐다.
"""
import numpy as np
import pandas as pd
import pytest

from app.analysis.metadata import _detect_binary_classes
from app.evaluation.labels import default_positive_class
from app.evaluation.metrics.binary import _resolve_positive_value
from app.evaluation.preprocessor import preprocess_data


@pytest.mark.parametrize(
    "values,expected",
    [
        (["cat", "dog"], ("dog", "cat", True)),
        (["dog", "cat", "dog"], ("dog", "cat", True)),
        ([0, 1], (1, 0, False)),
        ([1, 0, 1], (1, 0, False)),
        ([2, 10], (10, 2, False)),          # 수치 순서 — 문자열 순서면 '2' 가 뒤
        (["2", "10"], ("10", "2", False)),
        ([True, False], (True, False, False)),
        (["yes", "no"], ("yes", "no", False)),
        (["fraud", "normal"], ("fraud", "normal", False)),   # 알려진 짝: 정렬 뒤쪽(normal)이 아니라 fraud
        (["Benign", "Malignant"], ("Malignant", "Benign", False)),
        (["spam", "ham"], ("spam", "ham", False)),
    ],
)
def test_default_positive_class_rules(values, expected):
    assert default_positive_class(values) == expected


def test_default_positive_class_non_binary_values():
    assert default_positive_class([]) == (None, None, True)
    assert default_positive_class([None, float("nan")]) == (None, None, True)
    assert default_positive_class(["a"]) == ("a", None, True)
    assert default_positive_class(["a", "c", "b"]) == ("c", None, True)


def test_default_positive_class_keeps_original_dtype():
    pos, neg, _ = default_positive_class(np.array([0, 1, 1]))
    assert isinstance(pos, np.integer) and pos == 1 and neg == 0


def _derived_positive(truth: list) -> str:
    """확률에서 예측을 파생할 때 고른 양성(metadata 에 양성이 없을 때)."""
    df = pd.DataFrame({"t": truth, "s": [0.9] * len(truth)})
    mappings = [{"column": "t", "role": "y_true"}, {"column": "s", "role": "score_positive"}]
    _, logs = preprocess_data(df, mappings, "binary", metadata={}, selected_metric_ids=["M1"])
    return logs["derived_prediction"]["positive_class"]


@pytest.mark.parametrize(
    "truth",
    [["cat", "dog", "cat", "dog"], [0, 1, 0, 1], ["normal", "fraud", "normal", "fraud"], ["b", "a", "b", "a"]],
)
def test_three_sites_agree(truth):
    """자동 판단(metadata)·지표 계산(binary)·예측 파생(preprocessor)이 같은 입력에 같은 양성을 고른다."""
    expected, _, _ = default_positive_class(truth)
    series = pd.Series(truth)

    detected, _, _ = _detect_binary_classes(series)
    resolved = _resolve_positive_value(series)
    derived = _derived_positive(truth)

    assert detected == str(expected)
    assert str(resolved) == str(expected)
    assert derived == str(expected)


def test_cat_dog_is_dog_and_ambiguous_on_screen():
    pos, neg, ambiguous = _detect_binary_classes(pd.Series(["cat", "dog", "cat"]))
    assert (pos, neg, ambiguous) == ("dog", "cat", True)
