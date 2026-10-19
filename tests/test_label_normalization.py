"""정답·예측 하드 라벨 표기 맞추기(S2 계산 정확도) — 점검에서 재현한 값으로 고정한다.

화면·metadata·분포는 normalize_label 로 ' cat' 과 'cat', 1.0 과 1 을 같은 클래스로 보여 주는데, 지표 계산은
원래 값을 그대로 비교해 지표가 조용히 틀렸다.
- 정답 ' cat' · 예측 'cat'(이진) → Accuracy 0.5(실제 1.0), M21 이 3×3.
- ', ' 로 구분한 CSV → 예측 값이 ' cat' 이 되어 다중 클래스 M1 이 0.0.
- 정답 안에 'cat'·' cat' 이 섞이면 M22 에서 'cat' 3건이 빠지고 M4 0.375(실제 0.75), IR 은 4.0 과 1.0 두 값.
- 빈 칸 섞인 숫자 정답(float) → M21·M22·건수의 클래스 이름이 '1.0' 인데 분포는 '1'.
- 문자열 숫자 클래스 → M21 순서 ['1','10','2'], 분포·M22 는 ['1','2','10'].
이제 전처리가 정답·예측을 같은 표기(normalize_label)로 맞추고, 표기가 합쳐지면 경고하며, 이진에서 클래스가
둘을 넘으면 막는다. 모든 산출물의 클래스 이름·순서는 sort_labels 한 벌이다.
"""
import pandas as pd
import pytest

from app.core.parsing import parse_file_content
from app.evaluation.engine import evaluate

PRED = [{"role": "y_true", "column": "y_true"}, {"role": "y_pred", "column": "y_pred"}]


def _eval(truth, pred, task, metrics, positive_class=None):
    df = pd.DataFrame({"y_true": truth, "y_pred": pred})
    return evaluate(df, PRED, task, metrics, positive_class=positive_class)


def _notation_warnings(result) -> list[str]:
    return [w for w in result["_metadata"]["warnings"] if "표기" in w]


# ── 앞뒤 공백 ──

def test_binary_truth_with_leading_space_matches_prediction():
    r = _eval([" cat", "dog", " cat", "dog"], ["cat", "dog", "cat", "dog"], "binary",
              ["M1", "M4", "M21"], positive_class="cat")
    assert r["M1"] == 1.0
    assert r["M4"] == 1.0
    assert r["M21"]["labels"] == ["cat", "dog"]
    assert r["M21"]["matrix"] == [[2, 0], [0, 2]]
    assert r["confusion_counts"]["positive_class"] == "cat"
    assert (r["confusion_counts"]["tp"], r["confusion_counts"]["tn"]) == (2, 2)


def test_space_difference_is_reported_as_warning():
    r = _eval([" cat", "dog", " cat", "dog"], ["cat", "dog", "cat", "dog"], "binary", ["M1"],
              positive_class="cat")
    warnings = _notation_warnings(r)
    assert len(warnings) == 1
    assert "' cat'" in warnings[0] and "'cat'" in warnings[0]


def test_comma_space_csv_does_not_zero_the_accuracy():
    # 쉼표 뒤에 공백을 둔 CSV — 셀 값이 ' cat' 이 된다(컬럼 이름도 ' y_pred').
    csv = "y_true, y_pred\ncat, cat\ndog, dog\nbird, bird\ncat, dog\n"
    _, df = parse_file_content(csv.encode(), "d.csv")
    assert " y_pred" in df.columns and df[" y_pred"].iloc[0] == " cat"
    r = evaluate(df, [{"role": "y_true", "column": "y_true"}, {"role": "y_pred", "column": " y_pred"}],
                 "multiclass", ["M1", "M4", "M21"])
    assert r["M1"] == pytest.approx(0.75)
    assert r["M21"]["labels"] == ["bird", "cat", "dog"]
    assert _notation_warnings(r)


def test_truth_internal_space_variants_are_one_class():
    truth = ["cat", "cat", "cat", " cat", "dog", "dog", "dog", "dog"]
    pred = ["cat", "cat", "cat", "cat", "dog", "dog", "dog", "dog"]
    r = _eval(truth, pred, "multiclass", ["M1", "M4", "M11", "M22", "M23"])
    assert r["M1"] == 1.0
    assert r["M4"] == 1.0
    assert r["M11"]["f1_score"] == 1.0
    assert r["M22"]["cat"]["support"] == 4
    assert " cat" not in r["M22"]
    # 불균형 비율은 한 값이다(M23 = 분포 = 불균형 영향).
    assert r["M23"] == 1.0
    assert r["_metadata"]["class_distribution"] == {"cat": 4, "dog": 4}
    assert r["imbalance_effect"]["imbalance_ratio"] == 1.0
    assert _notation_warnings(r)


def test_clean_labels_have_no_notation_warning():
    r = _eval(["cat", "dog", "cat"], ["cat", "dog", "dog"], "multiclass", ["M1"])
    assert _notation_warnings(r) == []


# ── 숫자 표기·순서 ──

def test_float_truth_with_blank_names_classes_like_distribution():
    # 빈 칸 하나 때문에 정답이 float(1.0·2.0·10.0)이 된다. 빈 행은 결측으로 빠진다.
    truth = [1, 2, 10, None, 1, 2, 10]
    pred = [1, 2, 10, 1, 1, 10, 10]
    r = _eval(truth, pred, "multiclass", ["M1", "M21", "M22"])
    assert r["M21"]["labels"] == ["1", "2", "10"]
    assert [k for k in r["M22"] if k in {"1", "2", "10", "1.0", "2.0", "10.0"}] == ["1", "2", "10"]
    assert [row["label"] for row in r["confusion_counts"]["rows"]] == ["1", "2", "10"]
    assert list(r["_metadata"]["class_distribution"]) == ["1", "2", "10"]
    assert r["imbalance_effect"]["majority_class"] in {"1", "2", "10"}
    # float 표기만 다른 것(1.0 → '1')은 데이터가 합쳐진 것이 아니라 경고하지 않는다.
    assert _notation_warnings(r) == []


def test_string_numeric_classes_are_in_numeric_order_everywhere():
    truth = ["1", "10", "2", "1", "10", "2"]
    pred = ["1", "10", "2", "2", "10", "1"]
    r = _eval(truth, pred, "multiclass", ["M21", "M22"])
    assert r["M21"]["labels"] == ["1", "2", "10"]
    assert [row["label"] for row in r["confusion_counts"]["rows"]] == ["1", "2", "10"]
    assert [k for k in r["M22"] if k in {"1", "2", "10"}] == ["1", "2", "10"]


def test_binary_float_truth_and_int_prediction_agree():
    truth = [1, 0, None, 1, 0, 1]
    pred = [1, 0, 1, 0, 0, 1]
    r = _eval(truth, pred, "binary", ["M1", "M21"], positive_class="1")
    assert r["M1"] == pytest.approx(0.8)
    assert r["M21"]["labels"] == ["0", "1"]
    assert r["confusion_counts"]["positive_class"] == "1"


# ── 이진에서 클래스가 둘을 넘으면 막는다 ──

def test_binary_with_three_classes_after_normalization_is_an_error():
    r = _eval(["cat", "dog", "bird", "cat"], ["cat", "dog", "cat", "cat"], "binary", ["M1"],
              positive_class="cat")
    assert "error" in r
    assert "클래스" in r["error"] and "3" in r["error"]


def test_binary_ghost_prediction_class_is_an_error():
    r = _eval(["cat", "dog", "cat", "dog"], ["cat", "dog", "fox", "dog"], "binary", ["M1"],
              positive_class="cat")
    assert "error" in r
    assert "fox" in r["error"]


def test_binary_space_variants_collapse_to_two_classes_and_pass():
    # 표기를 맞추기 전에는 ' cat'·'cat'·'dog' 세 값이라 막혔을 데이터 — 맞춘 뒤 두 클래스다.
    r = _eval([" cat", "cat", "dog", "dog"], ["cat", "cat", "dog", "cat"], "binary", ["M1"],
              positive_class="cat")
    assert "error" not in r
    assert r["M1"] == pytest.approx(0.75)


# ── 확률에서 예측을 만드는 경로도 같은 표기 ──

def test_derived_multiclass_prediction_with_space_truth():
    df = pd.DataFrame({
        "y_true": [" cat", " dog", " cat", " dog"],
        "prob_cat": [0.9, 0.2, 0.6, 0.3],
        "prob_dog": [0.1, 0.8, 0.4, 0.7],
    })
    maps = [{"role": "y_true", "column": "y_true"},
            {"role": "prob_per_class", "column": "prob_cat"},
            {"role": "prob_per_class", "column": "prob_dog"}]
    r = evaluate(df, maps, "multiclass", ["M1", "M21"])
    assert r["M1"] == 1.0
    assert r["M21"]["labels"] == ["cat", "dog"]


# ── (P1) 예측에만 있는 클래스(유령) ──

def test_m11_macro_uses_truth_classes_like_m4():
    # 점검 재현: a/b/c 정답, 예측에 'z' 2건 — 종전 M4 0.7302 · M11 0.5476.
    truth = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "a"]
    pred = ["a", "a", "z", "b", "b", "c", "c", "c", "z", "a"]
    r = _eval(truth, pred, "multiclass", ["M4", "M11", "M22"])
    assert r["M11"]["f1_score"] == pytest.approx(r["M4"])
    assert r["M11"]["f1_score"] == pytest.approx(r["M22"]["macro avg"]["f1-score"])


def test_confusion_counts_have_no_ghost_rows():
    r = _eval(["a", "b", "a", "b"], ["a", "z", "a", "b"], "multiclass", ["M1"])
    rows = r["confusion_counts"]["rows"]
    assert [row["label"] for row in rows] == ["a", "b"]
    b = rows[1]
    assert (b["tp"], b["fn"]) == (1, 1)  # 'z' 로 예측한 b 는 b 의 FN 이다


def test_string_integer_decimal_matches_integer_prediction():
    # JSON 의 "1.0"·"0.0" 글자 정답과 정수 예측 — 같은 클래스 '1'·'0' 이다(숫자 열의 1.0 과 같은 규칙).
    r = _eval(["1.0", "0.0", "1.0", "0.0"], [1, 0, 1, 1], "binary", ["M1", "M21"], positive_class="1")
    assert r["M1"] == pytest.approx(0.75)
    assert r["M21"]["labels"] == ["0", "1"]
    # 글자 '2.50' 같은 정수가 아닌 소수는 그대로다.
    from app.evaluation.labels import normalize_label
    assert normalize_label("2.50") == "2.50" and normalize_label(" 1.00 ") == "1"
