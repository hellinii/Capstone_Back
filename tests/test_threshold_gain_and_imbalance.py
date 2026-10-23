"""임계값 조정 여지(서술 설계 D40)·불균형 영향(D20) — success_metrics 의 두 키, 손검산 고정.

- f1_threshold_gain: 원본 PR 곡선의 최대 F1 − 지금 예측의 F1 (음수면 0). 이진·score 가 있을 때만.
- imbalance_effect: IR < 1.5 → mild. IR ≥ 1.5 이고 (정확도 − 매크로 F1 ≥ 0.10 또는
  다수 F1 − 소수 F1 ≥ 0.10) → affected. 그 외 → no_distortion.

부동소수 주의: 아래 "정확히 0.10" 예시는 분수로는 1/10 이지만 부동소수 계산은 0.0999…98 이다
(`0.7 − 0.6`, `0.6 − 0.5`). 그대로 비교하면 경계값이 "영향 없음"으로 빠진다.
"""
import random

import pandas as pd

from app.evaluation.engine import evaluate

BINARY_SCORE_ONLY = [
    {"role": "y_true", "column": "y_true"},
    {"role": "score_positive", "column": "score"},
]
BINARY_PRED = [
    {"role": "y_true", "column": "y_true"},
    {"role": "y_pred", "column": "y_pred"},
]
MULTICLASS_PRED = BINARY_PRED
MULTILABEL_PRED = [
    {"role": "true_labels", "column": "true_labels"},
    {"role": "pred_labels", "column": "pred_labels"},
]


def _binary_derived(rows, threshold=0.5):
    df = pd.DataFrame(rows, columns=["y_true", "score"])
    return evaluate(
        df, BINARY_SCORE_ONLY, "binary", ["M1"], positive_class="1",
        decision_threshold=threshold, metadata={"positive_class": "1", "negative_class": "0"},
    )


def _with_pred(rows, task="binary", positive_class=None):
    df = pd.DataFrame(rows, columns=["y_true", "y_pred"])
    return evaluate(df, BINARY_PRED, task, ["M1"], positive_class=positive_class)


def _repeat(*groups):
    """(정답, 예측, 개수) 묶음을 행으로 펼친다."""
    return [(t, p) for t, p, n in groups for _ in range(n)]


# ── f1_threshold_gain ─────────────────────────────────────────────

# (정답, 양성 확률). 임계값 0.5 → 0.9(TP)·0.6(FP)만 양성: TP1 FP1 FN3 → F1 = 2/6 = 0.3333.
# 원본 곡선의 최대는 임계값 0.35: TP4 FP1 FN0 → F1 = 8/9 = 0.8889. 이득 = 0.5556.
GAIN_ROWS = [
    (1, 0.9), (1, 0.45), (1, 0.4), (1, 0.35),
    (0, 0.6), (0, 0.3), (0, 0.2), (0, 0.1),
]


def test_gain_hand_computed_from_original_pr_curve():
    gain = _binary_derived(GAIN_ROWS)["f1_threshold_gain"]

    # 판 4 재료(S1): 0.35 점 TP4 FP1 FN0 → P 0.8·R 1.0, 지금(0.5) TP1 FP1 FN3 → P 0.5·R 0.25.
    # 표기: "0.350" 을 적용해도 0.35 와 예측이 같다(셋째 자리, 끝 0 유지).
    assert gain == {"max_f1": 0.8889, "current_f1": 0.3333, "gain": 0.5556, "best_threshold": 0.35,
                    "notable": True, "gain_text": "0.5556",
                    "best_recall": 1.0, "best_precision": 0.8, "current_recall": 0.25, "current_precision": 0.5,
                    "max_f1_raw": 0.888888888888889, "current_f1_raw": 1 / 3, "gain_raw": 0.5555555555555556,
                    "best_threshold_raw": 0.35, "best_recall_raw": 1.0, "best_precision_raw": 0.8,
                    "current_recall_raw": 0.25, "current_precision_raw": 0.5, "best_threshold_text": "0.350"}


def test_gain_is_zero_when_current_threshold_is_best():
    """임계값 0.35 로 파생하면 지금 예측이 곧 최대점이다 — 이득 0."""
    gain = _binary_derived(GAIN_ROWS, threshold=0.35)["f1_threshold_gain"]

    # gain_raw 는 두 F1 식(2PR/(P+R) 와 2TP/(2TP+FP+FN))의 마지막 자리 차이 그대로다 — 반올림 전 값이라 걷지 않는다.
    assert gain == {"max_f1": 0.8889, "current_f1": 0.8889, "gain": 0.0, "best_threshold": 0.35,
                    "notable": False, "gain_text": "0",
                    "best_recall": 1.0, "best_precision": 0.8, "current_recall": 1.0, "current_precision": 0.8,
                    "max_f1_raw": 0.888888888888889, "current_f1_raw": 8 / 9, "gain_raw": 1.1102230246251565e-16,
                    "best_threshold_raw": 0.35, "best_recall_raw": 1.0, "best_precision_raw": 0.8,
                    "current_recall_raw": 1.0, "current_precision_raw": 0.8, "best_threshold_text": "0.350"}


def test_gain_negative_difference_is_clamped_to_zero_with_uploaded_prediction():
    """y_pred 를 직접 올리면 곡선 밖의 점일 수 있다 — 지금 F1 이 최대보다 크면 이득은 0(음수 아님)."""
    df = pd.DataFrame(
        [(1, 1, 0.2), (1, 1, 0.3), (0, 0, 0.6), (0, 0, 0.1)],
        columns=["y_true", "y_pred", "score"],
    )
    mappings = BINARY_PRED + [{"role": "score_positive", "column": "score"}]
    gain = evaluate(df, mappings, "binary", ["M1"], positive_class="1")["f1_threshold_gain"]

    # 곡선: 임계값 0.2 → TP2 FP1 → F1 = 4/5. 지금 예측은 완벽 → F1 = 1.
    assert gain == {"max_f1": 0.8, "current_f1": 1.0, "gain": 0.0, "best_threshold": 0.2,
                    "notable": False, "gain_text": "0",
                    "best_recall": 1.0, "best_precision": 0.6667, "current_recall": 1.0, "current_precision": 1.0,
                    "max_f1_raw": 0.8, "current_f1_raw": 1.0, "gain_raw": 0.0,
                    "best_threshold_raw": 0.2, "best_recall_raw": 1.0, "best_precision_raw": 2 / 3,
                    "current_recall_raw": 1.0, "current_precision_raw": 1.0, "best_threshold_text": "0.200"}


def test_gain_uses_uploaded_prediction_for_current_f1():
    """y_pred 가 있으면 확률에서 파생하지 않는다 — 지금 F1 은 올린 예측으로 센다."""
    df = pd.DataFrame(
        [(t, int(s >= 0.5), s) for t, s in GAIN_ROWS], columns=["y_true", "y_pred", "score"],
    )
    df.loc[0, "y_pred"] = 0  # 0.9 를 음성으로 올렸다 → TP0 FP1 FN4 → F1 = 0
    mappings = BINARY_PRED + [{"role": "score_positive", "column": "score"}]
    gain = evaluate(df, mappings, "binary", ["M1"], positive_class="1")["f1_threshold_gain"]

    # 지금 예측 TP0 → Recall·Precision 모두 0(분모 FP1 이 있어 정의되지만 TP 가 0).
    assert gain == {"max_f1": 0.8889, "current_f1": 0.0, "gain": 0.8889, "best_threshold": 0.35,
                    "notable": True, "gain_text": "0.8889",
                    "best_recall": 1.0, "best_precision": 0.8, "current_recall": 0.0, "current_precision": 0.0,
                    "max_f1_raw": 0.888888888888889, "current_f1_raw": 0.0, "gain_raw": 0.888888888888889,
                    "best_threshold_raw": 0.35, "best_recall_raw": 1.0, "best_precision_raw": 0.8,
                    "current_recall_raw": 0.0, "current_precision_raw": 0.0, "best_threshold_text": "0.350"}


def test_gain_tie_with_float_noise_picks_lowest_threshold():
    """분수로는 같은 F1 이 부동소수로는 마지막 자리만 다른 동률 — 낮은 임계값을 고른다.

    임계값 0.6: TP3 FP2 FN1 → 6/9, 0.9: TP2 FP0 FN2 → 4/6. 둘 다 2/3 이지만 2PR/(P+R) 계산은
    0.66…65(0.6) vs 0.66…66(0.9) 이라 그대로 argmax 하면 0.9 가 이긴다.
    """
    rows = [
        (0, 0.3), (0, 0.4), (0, 0.5), (0, 0.7), (0, 0.8),
        (1, 0.2), (1, 0.6), (1, 0.9), (1, 0.9),
    ]
    gain = _binary_derived(rows)["f1_threshold_gain"]

    assert gain["max_f1"] == 0.6667
    assert gain["best_threshold"] == 0.6
    # 판 4 재료도 낮은 임계값(0.6) 점에서 온다: TP3 FP2 FN1 → P 0.6·R 0.75(0.9 점이면 P 1.0·R 0.5).
    assert (gain["best_threshold_raw"], gain["best_precision_raw"], gain["best_recall_raw"]) == (0.6, 0.6, 0.75)
    assert gain["best_threshold_text"] == "0.600"


def test_gain_absent_without_score_or_for_other_tasks():
    assert "f1_threshold_gain" not in _with_pred([(0, 0), (1, 1), (1, 0)])
    assert "f1_threshold_gain" not in _with_pred([("a", "a"), ("b", "b"), ("c", "b")], task="multiclass")


def test_gain_absent_without_prediction():
    """예측이 필요 없는 지표만 고르면 파생하지 않는다 — 지금 F1 이 없어 키도 없다."""
    df = pd.DataFrame(GAIN_ROWS, columns=["y_true", "score"])
    result = evaluate(df, BINARY_SCORE_ONLY, "binary", ["M9"], positive_class="1")

    assert "M9" in result and "f1_threshold_gain" not in result


# ── imbalance_effect ──────────────────────────────────────────────

def test_imbalance_ratio_exactly_1_5_is_not_mild():
    """3:2 = 1.5 정확히 — 경미(1.5 미만)가 아니다. 예측이 완벽하니 왜곡 미관찰."""
    effect = _with_pred(_repeat(("a", "a", 3), ("b", "b", 2)))["imbalance_effect"]

    assert effect == {
        "level": "no_distortion",
        "display": {"imbalance_ratio": "1.5", "accuracy_minus_macro_f1": "0",
                    "majority_minus_minority_f1": "0",
                    "accuracy": "1", "macro_f1": "1", "majority_f1": "1", "minority_f1": "1"},
        "imbalance_ratio": 1.5,
        "accuracy_minus_macro_f1": 0.0,
        "majority_minus_minority_f1": 0.0,
        "accuracy": 1.0, "macro_f1": 1.0, "majority_f1": 1.0, "minority_f1": 1.0,
        "majority_class": "a",
        "minority_class": "b",
    }


def test_imbalance_below_1_5_is_mild_even_with_large_gap():
    """4:3 = 1.3333 — F1 차가 커도 경미."""
    effect = _with_pred(_repeat(("a", "a", 4), ("b", "a", 3)))["imbalance_effect"]

    assert effect["level"] == "mild"
    assert effect["imbalance_ratio"] == 1.3333
    assert effect["majority_minus_minority_f1"] >= 0.10


def test_class_f1_gap_exactly_0_10_is_affected():
    """이진 neg 7 : pos 2 (IR 3.5). 양성 TP2·FP4 → pos F1 = 4/8 = 0.5, neg F1 = 6/10 = 0.6.
    차 = 1/10 (부동소수 0.0999…98). 정확도 − 매크로 F1 = 5/9 − 0.55 ≈ 0.0056 이라 둘째 조건만 성립."""
    rows = _repeat(("pos", "pos", 2), ("neg", "pos", 4), ("neg", "neg", 3))
    effect = _with_pred(rows, positive_class="pos")["imbalance_effect"]

    assert effect == {
        "level": "affected",
        "display": {"imbalance_ratio": "3.5", "accuracy_minus_macro_f1": "0.0056",
                    "majority_minus_minority_f1": "0.1",
                    "accuracy": "0.5556", "macro_f1": "0.55", "majority_f1": "0.6", "minority_f1": "0.5"},
        "imbalance_ratio": 3.5,
        "accuracy_minus_macro_f1": 0.0056,
        "majority_minus_minority_f1": 0.1,
        "accuracy": 0.5556, "macro_f1": 0.55, "majority_f1": 0.6, "minority_f1": 0.5,
        "majority_class": "neg",
        "minority_class": "pos",
    }


def test_accuracy_minus_macro_f1_exactly_0_10_is_affected():
    """다중 a 1 · b 2 · c 7 (IR 7). F1: a 1, b 0, c 12/15 = 0.8 → 매크로 0.6, 정확도 0.7.
    차 = 1/10 (부동소수 0.0999…98). 다수(c) − 소수(a) F1 = −0.2 라 첫째 조건만 성립."""
    rows = _repeat(("a", "a", 1), ("b", "c", 2), ("c", "b", 1), ("c", "c", 6))
    effect = _with_pred(rows, task="multiclass")["imbalance_effect"]

    assert effect == {
        "level": "affected",
        "display": {"imbalance_ratio": "7", "accuracy_minus_macro_f1": "0.1",
                    "majority_minus_minority_f1": "-0.2",
                    "accuracy": "0.7", "macro_f1": "0.6", "majority_f1": "0.8", "minority_f1": "1"},
        "imbalance_ratio": 7.0,
        "accuracy_minus_macro_f1": 0.1,
        "majority_minus_minority_f1": -0.2,
        "accuracy": 0.7, "macro_f1": 0.6, "majority_f1": 0.8, "minority_f1": 1.0,
        "majority_class": "c",
        "minority_class": "a",
    }


def test_gaps_just_below_0_10_are_no_distortion():
    """IR 3 이고 두 차가 모두 0.10 미만 — 불균형은 있으나 왜곡 미관찰."""
    rows = _repeat(("neg", "neg", 9), ("pos", "pos", 3))
    rows[0] = ("neg", "pos")  # neg F1 = 16/17, pos F1 = 6/7 → 차 0.0840
    effect = _with_pred(rows, positive_class="pos")["imbalance_effect"]

    assert effect["level"] == "no_distortion"
    assert effect["imbalance_ratio"] == 3.0
    assert 0 < effect["majority_minus_minority_f1"] < 0.10
    assert 0 < effect["accuracy_minus_macro_f1"] < 0.10


def test_minority_better_than_majority_is_no_distortion():
    """IR ≥ 1.5, 정확도 − 매크로 F1 < 0.10, 다수 F1 − 소수 F1 ≤ −0.10 — 소수 클래스가 더 잘 맞는다.
    차이는 부호 그대로 비교하므로 왜곡 미관찰이다(절댓값으로 비교하면 affected 로 뒤집힌다).

    neg 2 (둘 다 → pos), pos 1 (맞음), IR 2. pos: TP1 FP2 → F1 2/4 = 0.5, neg F1 = 0.
    다수 − 소수 = −0.5. 정확도 1/3 − 매크로 0.25 = 0.0833.
    """
    rows = _repeat(("neg", "pos", 2), ("pos", "pos", 1))
    effect = _with_pred(rows, positive_class="pos")["imbalance_effect"]

    assert effect == {
        "level": "no_distortion",
        "display": {"imbalance_ratio": "2", "accuracy_minus_macro_f1": "0.0833",
                    "majority_minus_minority_f1": "-0.5",
                    "accuracy": "0.3333", "macro_f1": "0.25", "majority_f1": "0", "minority_f1": "0.5"},
        "imbalance_ratio": 2.0,
        "accuracy_minus_macro_f1": 0.0833,
        "majority_minus_minority_f1": -0.5,
        "accuracy": 0.3333, "macro_f1": 0.25, "majority_f1": 0.0, "minority_f1": 0.5,
        "majority_class": "neg",
        "minority_class": "pos",
    }


def test_multilabel_minority_better_than_majority_is_no_distortion():
    """멀티레이블도 같다. 빈도 x 3 · y 1 (IR 3). x: TP1 FN2 → F1 2/4 = 0.5, y: TP1 FP1 → F1 2/3.
    다수 − 소수 = −0.1667 → 왜곡 미관찰."""
    df = pd.DataFrame(
        [("x|y", "x|y"), ("x", "y"), ("x", "")],
        columns=["true_labels", "pred_labels"],
    )
    effect = evaluate(df, MULTILABEL_PRED, "multilabel", ["M15"])["imbalance_effect"]

    assert effect["imbalance_ratio"] == 3.0
    assert effect["majority_minus_minority_f1"] == -0.1667
    assert effect["level"] == "no_distortion"


def test_imbalance_effect_ignores_ghost_predicted_class():
    """예측에만 있는 클래스는 F1 집합에 들어가지 않는다(M22 처럼 y_true 고정)."""
    rows = _repeat(("a", "a", 3), ("b", "b", 1), ("b", "z", 1))
    effect = _with_pred(rows, task="multiclass")["imbalance_effect"]

    # a F1 1, b F1 = 2/3 → 매크로 5/6, 정확도 4/5 → 차 −0.0333. z 를 넣었다면 매크로가 5/9 로 떨어진다.
    assert effect["accuracy_minus_macro_f1"] == -0.0333
    assert effect["majority_minus_minority_f1"] == 0.3333
    assert effect["level"] == "affected"


def test_imbalance_multilabel_uses_label_frequency_and_class_gap_only():
    df = pd.DataFrame(
        [("x|y", "x"), ("x", "x"), ("x", "x"), ("", "y")],
        columns=["true_labels", "pred_labels"],
    )
    effect = evaluate(df, MULTILABEL_PRED, "multilabel", ["M15"])["imbalance_effect"]

    # 빈도 x 3 · y 1 → IR 3. x F1 = 1, y F1 = 0 (TP0·FN1·FP1).
    assert effect == {
        "level": "affected",
        "display": {"imbalance_ratio": "3", "accuracy_minus_macro_f1": None,
                    "majority_minus_minority_f1": "1",
                    "accuracy": None, "macro_f1": None, "majority_f1": "1", "minority_f1": "0"},
        "imbalance_ratio": 3.0,
        "accuracy_minus_macro_f1": None,
        "majority_minus_minority_f1": 1.0,
        "accuracy": None, "macro_f1": None, "majority_f1": 1.0, "minority_f1": 0.0,
        "majority_class": "x",
        "minority_class": "y",
    }


def test_imbalance_multilabel_perfect_prediction_is_no_distortion():
    df = pd.DataFrame(
        [("x|y", "x|y"), ("x", "x"), ("x", "x")],
        columns=["true_labels", "pred_labels"],
    )
    effect = evaluate(df, MULTILABEL_PRED, "multilabel", ["M15"])["imbalance_effect"]

    assert effect["level"] == "no_distortion"
    assert effect["imbalance_ratio"] == 3.0


def test_imbalance_tie_break_is_deterministic():
    """개수가 같으면 sort_labels 순서의 앞쪽 — 숫자 라벨은 수치 순서(2 가 10 보다 앞)."""
    rows = _repeat((10, 10, 2), (2, 2, 2), (7, 7, 1), (5, 5, 1))
    shuffled = rows[:]
    random.Random(3).shuffle(shuffled)

    first = _with_pred(rows, task="multiclass")["imbalance_effect"]
    second = _with_pred(shuffled, task="multiclass")["imbalance_effect"]

    assert first == second
    assert (first["majority_class"], first["minority_class"]) == ("2", "5")


def test_imbalance_all_equal_picks_two_different_classes():
    effect = _with_pred(_repeat(("b", "b", 2), ("a", "a", 2)))["imbalance_effect"]

    assert effect["level"] == "mild"
    assert (effect["majority_class"], effect["minority_class"]) == ("a", "b")


def test_imbalance_absent_when_single_class():
    """정답 클래스가 하나뿐이면 IR 을 정의할 수 없다(M23 과 같다) — 키 없음."""
    assert "imbalance_effect" not in _with_pred([("a", "a"), ("a", "b")])


def test_imbalance_absent_without_prediction():
    df = pd.DataFrame([(0, 0.1), (1, 0.9), (1, 0.8)], columns=["y_true", "score"])
    result = evaluate(df, BINARY_SCORE_ONLY, "binary", ["M23"], positive_class="1")

    assert "M23" in result and "imbalance_effect" not in result
