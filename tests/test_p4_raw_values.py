"""test_p4_raw_values.py — 판 4 라운드 S1 ③: 신뢰구간 경계·불균형 영향의 반올림 전 값.

- confidence_intervals[*].low_raw·high_raw: 반올림 전 경계. low·high(넷째 자리)는 그 반올림 그대로다(판 3 값 불변).
- imbalance_effect.*_raw: 일곱 숫자 칸의 반올림 전 값. 불균형 비율은 M23 원값과 같다(계획 2.7: 1.671875 → 1.672).
"""
import json

import pandas as pd
import pytest

from app.evaluation.engine import evaluate
from router_cases import CASES, request_json

IMBALANCE_NUMBERS = (
    "imbalance_ratio", "accuracy_minus_macro_f1", "majority_minus_minority_f1",
    "accuracy", "macro_f1", "majority_f1", "minority_f1",
)


def _engine(task: str):
    c = CASES[task]
    req = json.loads(request_json(task))
    df = pd.read_csv(c["csv"])
    mappings = [{"column": m["column"], "role": m["role"]} for m in req["column_mappings"]]
    return evaluate(
        df, mappings, task, req["selected_metric_ids"],
        positive_class=req["metadata"].get("positive_class"), metadata=req["metadata"],
    )


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_interval_raw_bounds_round_to_the_published_bounds(task):
    cis = _engine(task)["confidence_intervals"]
    assert cis
    for metric_id, ci in cis.items():
        assert {"low_raw", "high_raw"} <= set(ci), metric_id
        if ci["method"] == "skipped":
            assert ci["low_raw"] is None and ci["high_raw"] is None
            continue
        assert ci["low"] == round(ci["low_raw"], 4), metric_id
        assert ci["high"] == round(ci["high_raw"], 4), metric_id
        assert ci["low_raw"] <= ci["high_raw"], metric_id
        assert isinstance(ci["low_raw"], float) and isinstance(ci["high_raw"], float)


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_imbalance_raw_values_round_to_the_published_values(task):
    result = _engine(task)
    effect = result["imbalance_effect"]
    for key in IMBALANCE_NUMBERS:
        raw = effect[f"{key}_raw"]
        if effect[key] is None:
            assert raw is None, key
        else:
            assert effect[key] == round(raw, 4), key
    # 불균형 비율 원값 = M23(같은 정의, 반올림 없음).
    assert effect["imbalance_ratio_raw"] == result["M23"]


def test_skipped_interval_carries_null_raw_bounds():
    rows = [(1, 0), (0, 0), (1, 0), (0, 0)]
    df = pd.DataFrame(rows, columns=["y_true", "y_pred"])
    ci = evaluate(
        df, [{"role": "y_true", "column": "y_true"}, {"role": "y_pred", "column": "y_pred"}],
        "binary", ["M2"], positive_class="1",
    )["confidence_intervals"]["M2"]
    assert ci["method"] == "skipped"
    assert (ci["low"], ci["high"], ci["low_raw"], ci["high_raw"]) == (None, None, None, None)
