"""경계·확신 오류 비율(서술 설계 D30) — success_metrics 의 error_ratios 키, 손검산 고정.

- 분모는 확률이 있는 오분류. 이진·다중은 표본, 멀티레이블은 틀린 레이블 칸.
- 경계: 이진·멀티레이블 |p − 임계값| < 0.1(엄격 미만), 다중 p(예측) − p(정답) < 0.1.
- 확신: 틀린 쪽 확률 ≥ 0.9.
- 임계값은 확률에서 예측을 파생했을 때만 안다. 예측을 직접 올렸으면 이진·멀티레이블 경계는 null.

부동소수 주의: `0.6 − 0.5` 는 0.0999…98, `0.35 − 0.25` 도 0.0999…98 이라 그대로 비교하면
"정확히 0.1 떨어진" 점이 경계로 셈해진다. 아래 손검산은 그런 점(0.6·0.4 / 임계값 0.5,
0.35 vs 0.25)을 일부러 넣어 "정확히 0.1 = 경계 아님"을 고정한다. `0.25 − 0.15` 는 부동소수로도
정확히 0.1 이라 엄격 미만(<) ↔ 이하(<=) 차이를 따로 잡는다.
"""
import pandas as pd
import pytest

from app.evaluation.engine import evaluate

BINARY_SCORE_ONLY = [
    {"role": "y_true", "column": "y_true"},
    {"role": "score_positive", "column": "score"},
]


def _binary_derived(rows, threshold=0.5):
    df = pd.DataFrame(rows, columns=["y_true", "score"])
    return evaluate(
        df, BINARY_SCORE_ONLY, "binary", ["M1"], positive_class="1",
        decision_threshold=threshold, metadata={"positive_class": "1", "negative_class": "0"},
    )


# (정답, 양성 확률) — 임계값 0.5 로 예측을 파생한다.
BINARY_ROWS = [
    (0, 0.55),  # FP, |0.05| 경계
    (0, 0.6),   # FP, 정확히 0.1 → 경계 아님(부동소수 0.0999…98)
    (0, 0.9),   # FP, 정확히 0.9 → 확신
    (1, 0.1),   # FN, 1 − p = 0.9 → 확신
    (1, 0.45),  # FN, |0.05| 경계
    (1, 0.89),  # 맞음
    (0, 0.2),   # 맞음
    (1, 0.4),   # FN, 정확히 0.1 → 경계 아님
    (0, 0.95),  # FP, 확신
]


def test_binary_derived_threshold_hand_computed():
    ratios = _binary_derived(BINARY_ROWS)["error_ratios"]

    assert ratios == {
        "unit": "sample",
        "misclassified": 7,
        "boundary": {"count": 2, "ratio": 0.2857},   # 0.55, 0.45
        "boundary_note": None,
        "confident": {"count": 3, "ratio": 0.4286},  # 0.9, 0.1(1−p), 0.95
        "boundary_margin": 0.1,
        "confident_level": 0.9,
        "threshold": 0.5,
    }


def test_binary_exact_float_distance_is_not_boundary():
    """임계값 0.25: 0.15 는 부동소수로도 정확히 0.1, 0.35 는 0.0999…98 — 둘 다 경계 아님."""
    ratios = _binary_derived([(1, 0.15), (0, 0.35), (1, 0.16)], threshold=0.25)["error_ratios"]

    assert ratios["misclassified"] == 3
    assert ratios["boundary"] == {"count": 1, "ratio": 0.3333}  # 0.16 만(|0.09|)
    assert ratios["threshold"] == 0.25


def test_binary_confident_level_is_inclusive():
    ratios = _binary_derived([(0, 0.9), (1, 0.1), (0, 0.8999), (1, 0.1001)])["error_ratios"]

    assert ratios["misclassified"] == 4
    assert ratios["confident"] == {"count": 2, "ratio": 0.5}


def test_binary_with_uploaded_prediction_has_no_boundary():
    """y_pred 를 직접 올렸으면 임계값을 모른다 — 경계는 null, 확신은 그대로 센다."""
    df = pd.DataFrame(
        [(0, 1, 0.95), (1, 0, 0.05), (0, 1, 0.55), (1, 1, 0.7), (0, 0, 0.2)],
        columns=["y_true", "y_pred", "score"],
    )
    mappings = BINARY_SCORE_ONLY + [{"role": "y_pred", "column": "y_pred"}]
    # decision_threshold 를 줘도 예측이 파생되지 않았으므로 그 값으로 경계를 재지 않는다.
    ratios = evaluate(df, mappings, "binary", ["M1"], positive_class="1", decision_threshold=0.5)["error_ratios"]

    assert ratios["unit"] == "sample"
    assert ratios["misclassified"] == 3
    assert ratios["boundary"] is None
    assert ratios["boundary_note"] == "threshold_unknown"
    assert ratios["threshold"] is None
    assert ratios["confident"] == {"count": 2, "ratio": 0.6667}  # 0.95(FP), 0.05(FN, 1−p=0.95)


def test_binary_uploaded_prediction_confident_uses_predicted_side_probability():
    """확신은 '틀린 쪽(예측한 쪽)' 확률로 잰다 — y_pred 를 직접 올리면 예측이 점수 쪽과 어긋날 수 있다.

    (0, 1, 0.05): 양성으로 예측했는데 양성 확률 0.05 → 틀린 쪽 확률 0.05, 확신 아님.
    (1, 0, 0.97): 음성으로 예측했는데 음성 확률 0.03 → 확신 아님.
    (0, 1, 0.95): 양성 예측, 양성 확률 0.95 → 확신.
    max(p, 1 − p) 로 재면 셋 다 확신(0.95·0.97·0.95)이 된다.
    """
    df = pd.DataFrame([(0, 1, 0.05), (1, 0, 0.97), (0, 1, 0.95)], columns=["y_true", "y_pred", "score"])
    mappings = BINARY_SCORE_ONLY + [{"role": "y_pred", "column": "y_pred"}]
    ratios = evaluate(df, mappings, "binary", ["M1"], positive_class="1")["error_ratios"]

    assert ratios["misclassified"] == 3
    assert ratios["confident"] == {"count": 1, "ratio": 0.3333}


def test_no_misclassification_gives_null_ratios():
    ratios = _binary_derived([(1, 0.8), (0, 0.2)])["error_ratios"]

    assert ratios["misclassified"] == 0
    assert ratios["boundary"] == {"count": 0, "ratio": None}
    assert ratios["confident"] == {"count": 0, "ratio": None}


def test_no_key_without_probabilities():
    df = pd.DataFrame([(0, 1), (1, 1)], columns=["y_true", "y_pred"])
    mappings = [{"role": "y_true", "column": "y_true"}, {"role": "y_pred", "column": "y_pred"}]
    assert "error_ratios" not in evaluate(df, mappings, "binary", ["M1"], positive_class="1")


MULTICLASS = [
    {"role": "y_true", "column": "y_true"},
    {"role": "y_pred", "column": "y_pred"},
    {"role": "prob_per_class", "column": "prob_a"},
    {"role": "prob_per_class", "column": "prob_b"},
    {"role": "prob_per_class", "column": "prob_c"},
]

# (정답, 예측, p(a), p(b), p(c)) — 예측은 직접 올린 값(argmax 와 다를 수 있다).
MULTICLASS_ROWS = [
    ("a", "b", 0.45, 0.5, 0.05),   # 0.5 − 0.45 = 0.05 경계
    ("a", "b", 0.3, 0.6, 0.1),     # 0.3 차이
    ("b", "c", 0.05, 0.05, 0.9),   # p(예측) 0.9 → 확신
    ("c", "a", 0.35, 0.4, 0.25),   # 0.35 − 0.25 = 정확히 0.1(부동소수 0.0999…98) → 경계 아님
    ("a", "c", 0.6, 0.0, 0.4),     # 0.4 − 0.6 = −0.2 → 0.1 미만이라 경계
    ("b", "b", 0.1, 0.8, 0.1),     # 맞음
    ("a", "a", 0.7, 0.2, 0.1),     # 맞음
]


def test_multiclass_always_has_boundary_even_with_uploaded_prediction():
    df = pd.DataFrame(MULTICLASS_ROWS, columns=["y_true", "y_pred", "prob_a", "prob_b", "prob_c"])
    ratios = evaluate(df, MULTICLASS, "multiclass", ["M1"])["error_ratios"]

    assert ratios == {
        "unit": "sample",
        "misclassified": 5,
        "boundary": {"count": 2, "ratio": 0.4},
        "boundary_note": None,
        "confident": {"count": 1, "ratio": 0.2},
        "boundary_margin": 0.1,
        "confident_level": 0.9,
        "threshold": None,
    }


def test_multiclass_derived_argmax_counts_the_same_way():
    rows = [("a", 0.45, 0.5, 0.05), ("b", 0.05, 0.05, 0.9), ("c", 0.1, 0.2, 0.7)]
    df = pd.DataFrame(rows, columns=["y_true", "prob_a", "prob_b", "prob_c"])
    mappings = [m for m in MULTICLASS if m["role"] != "y_pred"]
    ratios = evaluate(df, mappings, "multiclass", ["M1"])["error_ratios"]

    assert ratios["misclassified"] == 2
    assert ratios["boundary"] == {"count": 1, "ratio": 0.5}
    assert ratios["confident"] == {"count": 1, "ratio": 0.5}


def test_multiclass_unresolvable_probability_names_drop_the_key():
    """확률 컬럼명으로 클래스를 확정할 수 없으면 순서로 추측하지 않고 키를 두지 않는다."""
    df = pd.DataFrame(MULTICLASS_ROWS, columns=["y_true", "y_pred", "p1", "p2", "p3"])
    mappings = MULTICLASS[:2] + [{"role": "prob_per_class", "column": c} for c in ("p1", "p2", "p3")]
    results = evaluate(df, mappings, "multiclass", ["M1"])

    assert "error_ratios" not in results
    assert "M1" in results  # 지표는 그대로 계산된다


MULTILABEL_SCORE_ONLY = [
    {"role": "true_labels", "column": "true_labels"},
    {"role": "score_per_label", "column": "score_x"},
    {"role": "score_per_label", "column": "score_y"},
]

# (정답, score_x, score_y) — 임계값 x 0.5, y 0.25 로 예측을 파생한다.
MULTILABEL_ROWS = [
    ("x", 0.55, 0.15),    # 예측 [x] — 두 칸 다 맞음
    ("x|y", 0.45, 0.15),  # 예측 [] — x 칸 FN 경계(|0.05|), y 칸 FN 정확히 0.1 → 경계 아님
    ("", 0.95, 0.3),      # 예측 [x,y] — x 칸 FP 확신, y 칸 FP 경계(|0.05|)
    ("y", 0.1, 0.05),     # 예측 [] — x 칸 맞음, y 칸 FN 1 − p = 0.95 확신
]


def test_multilabel_counts_label_cells_with_per_label_thresholds():
    df = pd.DataFrame(MULTILABEL_ROWS, columns=["true_labels", "score_x", "score_y"])
    thresholds = {"score_x": 0.5, "score_y": 0.25}
    ratios = evaluate(
        df, MULTILABEL_SCORE_ONLY, "multilabel", ["M15"], decision_threshold=thresholds,
    )["error_ratios"]

    assert ratios == {
        "unit": "label_cell",
        "misclassified": 5,
        "boundary": {"count": 2, "ratio": 0.4},
        "boundary_note": None,
        "confident": {"count": 2, "ratio": 0.4},
        "boundary_margin": 0.1,
        "confident_level": 0.9,
        "threshold": thresholds,
    }


def test_multilabel_with_uploaded_prediction_has_no_boundary():
    df = pd.DataFrame(MULTILABEL_ROWS, columns=["true_labels", "score_x", "score_y"])
    df["pred_labels"] = ["x", "", "x|y", ""]
    mappings = MULTILABEL_SCORE_ONLY + [{"role": "pred_labels", "column": "pred_labels"}]
    ratios = evaluate(df, mappings, "multilabel", ["M15"])["error_ratios"]

    assert ratios["unit"] == "label_cell"
    assert ratios["misclassified"] == 5
    assert ratios["boundary"] is None
    assert ratios["boundary_note"] == "threshold_unknown"
    assert ratios["threshold"] is None
    assert ratios["confident"] == {"count": 2, "ratio": 0.4}


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_router_fixtures_with_probabilities_carry_error_ratios(task):
    """골든 픽스처는 모두 확률과 직접 올린 예측을 가진다 — 이진·멀티는 경계 null, 다중은 계산."""
    from router_cases import CASES

    c = CASES[task]
    df = pd.read_csv(c["csv"])
    mappings = [{"column": col, "role": role} for col, role in c["mappings"]]
    results = evaluate(df, mappings, task, c["selected_metric_ids"], positive_class="1")
    ratios = results["error_ratios"]

    assert ratios["misclassified"] > 0
    if task == "multiclass":
        assert ratios["boundary"] is not None
    else:
        assert ratios["boundary"] is None and ratios["boundary_note"] == "threshold_unknown"
