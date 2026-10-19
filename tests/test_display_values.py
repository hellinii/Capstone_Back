"""판정과 맞게 자릿수를 늘린 표시값(S2) — 불균형 영향·임계값 조정 여지.

엔진은 반올림 전 값(9자리로 잡음만 걷어 냄)으로 기준(IR 1.5, 차이 0.10)을 판정하고, 넷째 자리로 반올림한
값을 싣는다. 그래서 IR 1.49996 은 '경미'(1.5 미만)인데 실린 값은 1.5, 임계값 조정 여지 0.09996 은 '부적합
아님'인데 실린 값은 0.1 이라, 숫자를 그대로 찍으면 판정과 어긋나 보였다(계획 검토 P2-2·P1-9).
- imbalance_effect.display: 세 값의 표시 문자열(판정과 같은 쪽으로 읽히게 자릿수를 늘림).
- f1_threshold_gain.notable: 반올림 전 gain 으로 정한 '임계값 부적합'(≥ 0.10). gain_text: 표시 문자열.
"""
import pandas as pd
import pytest

from app.evaluation.engine import evaluate
from app.evaluation.metrics.binary import threshold_gain_notable
from app.evaluation.metrics.display import judged_text


@pytest.mark.parametrize("value, threshold, meets, expected", [
    (1.49996, 1.5, False, "1.49996"),        # 넷째 자리로는 1.5 — 경미 판정과 어긋남
    (1.4999999, 1.5, False, "1.4999999"),
    (2.9683, 1.5, True, "2.9683"),
    (1.5, 1.5, True, "1.5"),
    (-0.0213, 0.1, False, "-0.0213"),        # 음수 차이는 그대로
    (0.09996, 0.1, False, "0.09996"),
    (0.1533, 0.1, True, "0.1533"),
    (0.0999999999999998, 0.1, True, "0.1"),  # 뺄셈 잡음 — 판정은 '기준 이상'이고 0.1 로 찍는다
    (0.1000049, 0.1, True, "0.1"),
    (0.0, 0.1, False, "0"),
    (-0.0, 0.1, False, "0"),
])
def test_judged_text(value, threshold, meets, expected):
    assert judged_text(value, threshold, meets) == expected


def test_judged_text_never_reads_on_the_wrong_side():
    # 9자리에서도 기준과 같게 읽히면 판정 쪽으로 자른다(숫자를 지어내지 않되 반대쪽으로 읽히지 않게).
    text = judged_text(1.4999999999, 1.5, False)
    assert float(text) < 1.5


def _binary(rows):
    df = pd.DataFrame(rows, columns=["y_true", "y_pred", "score"])
    maps = [{"role": "y_true", "column": "y_true"}, {"role": "y_pred", "column": "y_pred"},
            {"role": "score_positive", "column": "score"}]
    return evaluate(df, maps, "binary", ["M1"], positive_class="1")


def test_imbalance_effect_carries_display_strings():
    rows = [(1, 1, 0.9)] * 3 + [(1, 0, 0.4)] * 2 + [(0, 0, 0.1)] * 10 + [(0, 1, 0.6)] * 1
    ie = _binary(rows)["imbalance_effect"]
    d = ie["display"]
    assert set(d) == {"imbalance_ratio", "accuracy_minus_macro_f1", "majority_minus_minority_f1"}
    assert float(d["imbalance_ratio"]) == pytest.approx(ie["imbalance_ratio"])
    assert float(d["majority_minus_minority_f1"]) == pytest.approx(ie["majority_minus_minority_f1"])


def test_multilabel_imbalance_display_has_no_accuracy_gap():
    df = pd.DataFrame({"t": ["a|b", "a", "a", "b", "a"], "p": ["a", "a", "a|b", "b", "a"]})
    maps = [{"role": "true_labels", "column": "t"}, {"role": "pred_labels", "column": "p"}]
    ie = evaluate(df, maps, "multilabel", ["M15"])["imbalance_effect"]
    assert ie["display"]["accuracy_minus_macro_f1"] is None


def test_threshold_gain_carries_notable_and_text():
    rows = [(1, 0, 0.45)] * 4 + [(1, 1, 0.9)] * 2 + [(0, 0, 0.1)] * 6
    g = _binary(rows)["f1_threshold_gain"]
    assert isinstance(g["notable"], bool)
    assert g["notable"] == (g["gain"] >= 0.1)
    assert float(g["gain_text"]) == pytest.approx(g["gain"])


@pytest.mark.parametrize("gain, notable", [
    (0.09996, False),               # 넷째 자리로는 0.1 — 반올림 전 값으로 판정하면 부적합이 아니다
    (0.0999999999999998, True),     # 0.9 − 0.8 같은 뺄셈 잡음 — 꼭 0.10 은 부적합
    (0.1, True), (0.1409, True), (0.0, False),
])
def test_threshold_gain_notable_uses_unrounded_gain(gain, notable):
    assert threshold_gain_notable(gain) is notable
