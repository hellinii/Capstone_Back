"""늘 계산하는 혼동행렬·클래스별 표(S2) — success_metrics 의 confusion_matrix·class_report 키.

서술 설계는 7.1 혼동행렬 해석·최약 클래스·주요 오류를 M21·M22 선택과 무관하게 쓰기로 했다(사용자 결정
2026-10-19 "최약 클래스·주요 오류"). 종전에는 M21·M22 를 골랐을 때만 행렬·표가 응답에 있어, 고르지 않은
평가의 사실 시트에는 둘이 없었다. 성적서 6절의 M21·M22 표 인쇄는 지금처럼 선택을 따른다(프론트 몫).

- confusion_matrix: M21 과 같은 모양 + n_samples. 멀티레이블은 레이블별 2×2 — 각 2×2 의 합은 표본 수이고,
  모든 칸의 합은 '레이블 칸' 수다(200행·레이블 4개면 800칸, 점검 BE-09 의 '전체 800건' 오해).
- class_report: M22 와 같은 모양(sklearn classification_report). 이진·다중은 'accuracy' 를 늘 싣고 'micro avg' 는
  싣지 않는다(다중의 마이크로 평균 = Accuracy — 프론트가 이 값으로 마이크로 줄을 만든다). 멀티레이블은
  매크로·가중·마이크로·샘플 평균.
"""
import random

import pandas as pd
import pytest

from app.evaluation.engine import evaluate
from app.evaluation.report import generate_report

PRED = [{"role": "y_true", "column": "y_true"}, {"role": "y_pred", "column": "y_pred"}]
ML = [{"role": "true_labels", "column": "t"}, {"role": "pred_labels", "column": "p"}]


def _eval(truth, pred, task, metrics, **kw):
    return evaluate(pd.DataFrame({"y_true": truth, "y_pred": pred}), PRED, task, metrics, **kw)


def test_binary_tables_without_choosing_m21_m22():
    r = _eval([1, 0, 1, 0, 1], [1, 0, 0, 0, 1], "binary", ["M1"], positive_class="1")
    assert "M21" not in r and "M22" not in r
    cm = r["confusion_matrix"]
    assert cm["type"] == "multiclass_or_binary"
    assert cm["labels"] == ["0", "1"]
    assert cm["matrix"] == [[2, 0], [1, 2]]
    assert cm["n_samples"] == 5
    report = r["class_report"]
    assert report["1"]["support"] == 3
    assert report["accuracy"] == pytest.approx(0.8)
    assert "micro avg" not in report
    assert set(report) >= {"0", "1", "macro avg", "weighted avg"}


def test_tables_match_m21_m22_when_chosen():
    truth = ["a", "b", "c", "a", "b", "c", "a"]
    pred = ["a", "b", "b", "a", "c", "c", "b"]
    r = _eval(truth, pred, "multiclass", ["M21", "M22"])
    assert r["confusion_matrix"]["labels"] == r["M21"]["labels"]
    assert r["confusion_matrix"]["matrix"] == r["M21"]["matrix"]
    for key in ("a", "b", "c", "macro avg", "weighted avg"):
        assert r["class_report"][key] == r["M22"][key]


def test_multiclass_ghost_prediction_keeps_accuracy_row():
    # 예측에만 있는 'z' 가 있으면 sklearn 은 accuracy 대신 'micro avg'(정답 클래스로 좁힌 값)를 낸다.
    # 새 칸은 늘 accuracy 를 싣는다 — 마이크로 평균 줄은 Accuracy 하나로 찍힌다.
    r = _eval(["a", "b", "a", "b"], ["a", "z", "a", "b"], "multiclass", ["M1"])
    report = r["class_report"]
    assert report["accuracy"] == pytest.approx(r["M1"])
    assert "micro avg" not in report and "z" not in report
    assert r["confusion_matrix"]["labels"] == ["a", "b", "z"]


def test_multilabel_800_cells_are_not_800_samples():
    rng = random.Random(7)
    labels = ["a", "b", "c", "d"]
    t = ["|".join(l for l in labels if rng.random() < 0.4) for _ in range(200)]
    p = ["|".join(l for l in labels if rng.random() < 0.4) for _ in range(200)]
    r = evaluate(pd.DataFrame({"t": t, "p": p}), ML, "multilabel", ["M15"])
    cm = r["confusion_matrix"]
    assert cm["type"] == "multilabel"
    assert cm["labels"] == labels
    assert cm["n_samples"] == 200
    # 레이블마다 2×2 는 표본 200개를 나눈다 — 합치면 800 '칸'이지 800 '건'이 아니다.
    for block in cm["matrix"]:
        assert sum(sum(row) for row in block) == 200
    assert cm["unit"] == "label_cell"
    assert cm["cell_total"] == 800
    report = r["class_report"]
    assert set(report) >= {"a", "b", "c", "d", "macro avg", "weighted avg", "micro avg", "samples avg"}


def test_tables_are_absent_without_prediction():
    # 예측이 없으면(불균형 비율만 고른 경우) 표를 만들 수 없다 — 키를 두지 않는다.
    df = pd.DataFrame({"y_true": ["a", "b", "a"]})
    r = evaluate(df, [{"role": "y_true", "column": "y_true"}], "multiclass", ["M23"])
    assert "confusion_matrix" not in r and "class_report" not in r


def test_tables_travel_in_success_metrics():
    r = _eval([1, 0, 1], [1, 0, 0], "binary", ["M1"], positive_class="1")
    formatted = generate_report(r)
    assert "confusion_matrix" in formatted["success_metrics"]
    assert "class_report" in formatted["success_metrics"]
    assert "confusion_matrix" not in formatted["failed_metrics"]
