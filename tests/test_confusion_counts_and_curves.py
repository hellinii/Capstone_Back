"""TP·TN·FP·FN 건수와 ROC·PR 곡선은 지표 선택과 무관하게 나온다(재료 목록 9절).

- 건수: 항상. 이진은 양성 기준 하나, 다중·멀티는 클래스(레이블)마다. 7.1 에 인쇄하고 AI 에 넘긴다.
- 곡선: 이진이고 score 가 있을 때. 종전에는 AUROC(M9)·AUPRC(M10)를 골랐을 때만 만들었다.
"""
from pathlib import Path

import pandas as pd
import pytest

from app.evaluation.engine import evaluate

DATA = Path(__file__).parent / "data"
BINARY = [
    {"role": "y_true", "column": "y_true"},
    {"role": "y_pred", "column": "y_pred"},
    {"role": "score_positive", "column": "score_positive"},
]


@pytest.fixture
def binary_df():
    return pd.read_csv(DATA / "binary" / "binary_test_data_200.csv")


def test_binary_counts_without_confusion_matrix_metric(binary_df):
    results = evaluate(binary_df, BINARY, "binary", ["M1"], positive_class="1")
    counts = results["confusion_counts"]

    assert counts["type"] == "binary"
    assert counts["positive_class"] == "1"
    assert counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"] == len(binary_df)
    assert "M21" not in results


def test_binary_counts_follow_the_positive_class(binary_df):
    one = evaluate(binary_df, BINARY, "binary", ["M1"], positive_class="1")["confusion_counts"]
    zero = evaluate(binary_df, BINARY, "binary", ["M1"], positive_class="0")["confusion_counts"]

    # 양성을 뒤집으면 TP↔TN, FP↔FN 이 바뀐다.
    assert (zero["tp"], zero["tn"], zero["fp"], zero["fn"]) == (one["tn"], one["tp"], one["fn"], one["fp"])


def test_multiclass_counts_per_class():
    df = pd.read_csv(DATA / "multiclass" / "multiclass_200.csv")
    mappings = [{"role": "y_true", "column": "y_true"}, {"role": "y_pred", "column": "y_pred"}]
    counts = evaluate(df, mappings, "multiclass", ["M1"])["confusion_counts"]

    assert counts["type"] == "per_class"
    assert {row["label"] for row in counts["rows"]} == {"bird", "cat", "dog"}
    for row in counts["rows"]:
        assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == len(df)


def test_multilabel_counts_per_label():
    df = pd.read_csv(DATA / "multilabel" / "multilabel_200.csv")
    mappings = [
        {"role": "true_labels", "column": "true_labels"},
        {"role": "pred_labels", "column": "pred_labels"},
    ]
    counts = evaluate(df, mappings, "multilabel", ["M15"])["confusion_counts"]

    assert counts["type"] == "per_class"
    assert len(counts["rows"]) >= 2
    for row in counts["rows"]:
        assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == len(df)


def test_binary_curves_without_auc_metrics(binary_df):
    results = evaluate(binary_df, BINARY, "binary", ["M1"], positive_class="1")

    assert "fpr" in results["roc_curve"] and "tpr" in results["roc_curve"]
    assert "recall" in results["pr_curve"] and "precision" in results["pr_curve"]


def test_no_curves_without_a_score_column(binary_df):
    mappings = [m for m in BINARY if m["role"] != "score_positive"]
    results = evaluate(binary_df, mappings, "binary", ["M1", "M9"], positive_class="1")

    assert "roc_curve" not in results
    assert "pr_curve" not in results
