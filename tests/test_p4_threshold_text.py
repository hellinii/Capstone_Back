"""test_p4_threshold_text.py — 판 4 라운드 S1 ②: F1 최대 임계값의 표기(best_threshold_text)와 그 점의 Recall·Precision.

정해진 것 3: F1 최대 임계값은 셋째 자리로 찍고, 그 표기를 적용해 예측(판정)이 바뀌면 자릿수를 늘린다.
RPT-2026-0001(bc_strong)은 0.617 을 적용하면 Recall 이 0.984 → 0.969 로 바뀌어 "0.6166" 이다(실데이터 대조는
test_p4_bc_strong.py).
"""
import random

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from app.evaluation.engine import evaluate
from app.evaluation.metrics.binary import threshold_text

BINARY_SCORE_ONLY = [
    {"role": "y_true", "column": "y_true"},
    {"role": "score_positive", "column": "score"},
]


def _gain(rows, threshold=0.5):
    df = pd.DataFrame(rows, columns=["y_true", "score"])
    return evaluate(
        df, BINARY_SCORE_ONLY, "binary", ["M1"], positive_class="1",
        decision_threshold=threshold, metadata={"positive_class": "1", "negative_class": "0"},
    )["f1_threshold_gain"]


def test_third_digit_when_it_keeps_predictions():
    scores = np.array([0.1, 0.35, 0.9])
    assert threshold_text(0.35, scores) == "0.350"  # 끝 0 유지
    assert threshold_text(0.5, np.array([0.2, 0.5, 0.7])) == "0.500"


def test_rounding_up_drops_the_threshold_sample_so_digits_grow():
    """0.6166 → "0.617" 이면 0.6166 인 표본이 음성이 된다 ✗ → "0.6166"(RPT-2026-0001 모양)."""
    scores = np.array([0.2, 0.6166, 0.8])
    assert threshold_text(0.6166, scores) == "0.6166"


def test_rounding_down_pulls_in_a_lower_sample_so_digits_grow():
    """0.61249 → "0.612" 이면 0.6124 가 양성이 된다 ✗, "0.6125" 는 0.61249 가 음성 ✗ → "0.61249"."""
    scores = np.array([0.6124, 0.61249, 0.7])
    assert threshold_text(0.61249, scores) == "0.61249"


def test_rounding_down_is_fine_when_no_sample_in_between():
    """0.6124 → "0.612" 사이에 표본이 없으면 셋째 자리 그대로."""
    scores = np.array([0.3, 0.6124, 0.7])
    assert threshold_text(0.6124, scores) == "0.612"


def test_falls_back_to_repr_when_scores_are_denser_than_12_digits():
    t = 0.1 + 1e-14
    scores = np.array([0.1, t, 0.5])
    assert threshold_text(t, scores) == repr(t)


def test_random_scores_text_is_shortest_and_keeps_predictions():
    """무작위 점수(넷째 자리 반올림 — 실데이터 모양)에서 표기를 적용한 예측 = 그 점의 예측, 한 자리 줄이면 다르다."""
    rng = random.Random(4213)
    for _ in range(200):
        n = rng.randint(5, 60)
        scores = np.array([round(rng.random(), rng.choice([2, 3, 4, 6])) for _ in range(n)])
        t = float(rng.choice(list(scores)))
        text = threshold_text(t, scores)
        assert np.array_equal(scores >= float(text), scores >= t), (t, text)
        digits = len(text.split(".")[1])
        assert digits >= 3
        if digits > 3:
            shorter = f"{t:.{digits - 1}f}"
            assert not np.array_equal(scores >= float(shorter), scores >= t), (t, text)


def test_best_and_current_recall_precision_match_sklearn():
    rng = random.Random(7)
    # 점수는 여섯째 자리 — 원값 임계값이 넷째 자리 칸(best_threshold)과 달라야 반올림을 시험할 수 있다.
    rows = [(int(rng.random() < 0.4), round(rng.random(), 6)) for _ in range(120)]
    gain = _gain(rows)
    assert gain["best_threshold_raw"] != gain["best_threshold"]
    y = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows])
    t = gain["best_threshold_raw"]
    best_pred = (s >= t).astype(int)
    assert gain["best_recall_raw"] == recall_score(y, best_pred)
    assert gain["best_precision_raw"] == precision_score(y, best_pred)
    # 표기를 적용한 예측도 같은 Recall·Precision 이다.
    text_pred = (s >= float(gain["best_threshold_text"])).astype(int)
    assert np.array_equal(text_pred, best_pred)
    current_pred = (s >= 0.5).astype(int)
    assert gain["current_recall_raw"] == recall_score(y, current_pred)
    assert gain["current_precision_raw"] == precision_score(y, current_pred, zero_division=0)
    # 넷째 자리 칸은 원값의 반올림 그대로.
    for key in ("best_recall", "best_precision", "current_recall", "current_precision",
                "max_f1", "current_f1", "gain", "best_threshold"):
        assert gain[key] == round(gain[f"{key}_raw"], 4), key


def test_current_precision_is_zero_when_nothing_predicted_positive():
    """임계값 0.95 → 양성 예측 0건: Precision 분모 0 → 0(M2 의 zero_division=0 과 같다)."""
    gain = _gain([(1, 0.9), (1, 0.4), (0, 0.2), (0, 0.1)], threshold=0.95)
    assert gain["current_precision_raw"] == 0.0
    assert gain["current_recall_raw"] == 0.0
    assert gain["best_threshold_text"] == "0.400"
