"""지표 95% 신뢰구간(서술 설계 D29) — success_metrics 의 confidence_intervals 키, 손검산 고정.

- 비율 지표는 Wilson. 8/10 → [0.4902, 0.9433] 은 손으로 푼 값이다:
  z = 1.96, z²/n = 0.38415, 중심 = (0.8 + 0.19207)/1.38415 = 0.71674,
  반폭 = 1.95996/1.38415 × √(0.016 + 0.009604) = 0.22658.
- 혼동행렬 지표는 칸 다항 재표집, 점수·멀티레이블 지표는 행 재표집(2,000회, 시드 4213).
  작업량 상한을 넘으면 해석적 방법(DeLong·로짓·정규근사), 그래도 안 되면 skipped + 사유.
- 점추정이 엔진 지표 값과 1e-9 넘게 다르면 skipped("점추정 불일치").

이벤트 루프: 평가 파이프라인 전체가 `run_cpu_bound` 로 스레드에 오프로드된다
(tests/test_event_loop_blocking.py::test_evaluate_does_not_block_event_loop). 여기서는 그 안에서
도는 신뢰구간 계산이 최악 크기(작업량 상한 바로 아래)에서도 수 초 안에 끝나는지를 잰다.
"""
import time

import numpy as np
import pandas as pd
import pytest

from app.evaluation.engine import evaluate
from app.evaluation.metrics import intervals
from router_cases import CASES

BINARY = [
    {"role": "y_true", "column": "y_true"},
    {"role": "y_pred", "column": "y_pred"},
]
BINARY_SCORE = BINARY + [{"role": "score_positive", "column": "score"}]


def _binary(rows, metrics, mappings=BINARY, columns=("y_true", "y_pred")):
    df = pd.DataFrame(rows, columns=list(columns))
    return evaluate(
        df, mappings, "binary", metrics, positive_class="1",
        metadata={"positive_class": "1", "negative_class": "0"},
    )


def _synthetic_scores(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    score = np.round(np.clip(0.35 * y + 0.65 * rng.random(n), 0, 1), 3)
    pred = (score >= 0.5).astype(int)
    return pd.DataFrame({"y_true": y, "y_pred": pred, "score": score})


def _csv_case(task):
    from app.core.parsing import parse_file_content

    c = CASES[task]
    _, df = parse_file_content(c["csv"].read_bytes(), c["csv"].name)
    mappings = [{"column": col, "role": role} for col, role in c["mappings"]]
    meta = {"positive_class": "1", "negative_class": "0"} if task == "binary" else None
    return df, mappings, c["selected_metric_ids"], meta


# ── Wilson 손검산 ──────────────────────────────────────────────────────

@pytest.mark.parametrize("k, n, expected", [
    (8, 10, (0.4902, 0.9433)),
    (0, 10, (0.0, 0.2775)),
    (10, 10, (0.7225, 1.0)),
    (1, 2, (0.0945, 0.9055)),
])
def test_wilson_hand_computed(k, n, expected):
    low, high = intervals.wilson_interval(k, n)
    assert (round(low, 4), round(high, 4)) == expected


def test_wilson_with_zero_denominator_is_skipped():
    with pytest.raises(intervals._Skip):
        intervals.wilson_interval(0, 0)


def test_engine_accuracy_eight_of_ten_is_wilson():
    rows = [(1, 1)] * 4 + [(0, 0)] * 4 + [(1, 0), (0, 1)]
    cis = _binary(rows, ["M1"])["confidence_intervals"]

    assert cis == {"M1": {"low": 0.4902, "high": 0.9433, "method": "wilson", "level": 0.95}}


def test_engine_binary_wilson_denominators_follow_the_metric():
    """M2 분모 = 예측 양성, M3 = 정답 양성, M7·M8 = 정답 음성."""
    # 정답 양성 5(TP 4, FN 1), 정답 음성 5(TN 3, FP 2) → 예측 양성 6.
    rows = [(1, 1)] * 4 + [(1, 0)] + [(0, 0)] * 3 + [(0, 1)] * 2
    cis = _binary(rows, ["M2", "M3", "M7", "M8"])["confidence_intervals"]

    def wilson(k, n):
        low, high = intervals.wilson_interval(k, n)
        return {"low": round(low, 4), "high": round(high, 4), "method": "wilson", "level": 0.95}

    assert cis == {"M2": wilson(4, 6), "M3": wilson(4, 5), "M7": wilson(3, 5), "M8": wilson(2, 5)}


def test_precision_without_predicted_positives_is_skipped():
    rows = [(1, 0), (0, 0), (1, 0), (0, 0)]
    ci = _binary(rows, ["M2"])["confidence_intervals"]["M2"]

    assert ci["method"] == "skipped"
    assert ci["low"] is None and ci["high"] is None
    assert "분모" in ci["reason"]


# ── 재표집 ─────────────────────────────────────────────────────────────

def test_resampling_is_deterministic():
    for task in ("binary", "multiclass", "multilabel"):
        df, mappings, metrics, meta = _csv_case(task)
        pos = "1" if task == "binary" else None
        first = evaluate(df.copy(), mappings, task, metrics, positive_class=pos, metadata=meta)
        second = evaluate(df.copy(), mappings, task, list(reversed(metrics)), positive_class=pos, metadata=meta)
        assert first["confidence_intervals"] == second["confidence_intervals"], task


def test_bootstrap_intervals_contain_point_estimate():
    df = _synthetic_scores(400)
    metrics = ["M4", "M5", "M6", "M9", "M10", "M19", "M20", "M23"]
    result = evaluate(df, BINARY_SCORE, "binary", metrics, positive_class="1",
                      metadata={"positive_class": "1", "negative_class": "0"})
    cis = result["confidence_intervals"]

    for metric_id in metrics:
        ci = cis[metric_id]
        assert ci["method"] in ("bootstrap", "bootstrap_multinomial"), metric_id
        assert ci["n_resamples"] == 2000 and ci["seed"] == 4213
        assert ci["low"] <= round(result[metric_id], 4) <= ci["high"], metric_id
    assert cis["M9"]["method"] == "bootstrap"
    assert cis["M4"]["method"] == "bootstrap_multinomial"


def test_selected_metrics_except_m21_m22_all_get_intervals():
    for task in ("binary", "multiclass", "multilabel"):
        df, mappings, metrics, meta = _csv_case(task)
        result = evaluate(df, mappings, task, metrics,
                          positive_class="1" if task == "binary" else None, metadata=meta)
        cis = result["confidence_intervals"]
        assert set(cis) == set(metrics) - {"M21", "M22"}, task
        # 고정 픽스처에서는 모든 지표의 점추정이 맞아야 한다 — skipped 가 없다.
        assert all(ci["method"] != "skipped" for ci in cis.values()), task


def test_no_interval_for_failed_or_matrix_metrics():
    rows = [(1, 1), (0, 0), (1, 0), (0, 1)]
    result = _binary(rows, ["M21", "M22", "M9"])  # M9 는 score 가 없어 실패
    assert "confidence_intervals" not in result


def test_multiclass_ghost_class_definitions_match_engine():
    """예측에만 있는 클래스(유령)가 있으면 M2~M5 와 M11 의 클래스 집합이 갈린다 — 둘 다 점추정이 맞아야 한다."""
    rows = [("a", "a")] * 6 + [("b", "b")] * 5 + [("c", "c")] * 4 + [("a", "z"), ("b", "a"), ("c", "z"), ("b", "c")]
    df = pd.DataFrame(rows, columns=["y_true", "y_pred"])
    metrics = ["M2", "M3", "M4", "M5", "M6", "M11", "M12", "M13", "M14"]
    cis = evaluate(df, BINARY, "multiclass", metrics, beta=2.0)["confidence_intervals"]

    for metric_id in metrics:
        assert cis[metric_id]["method"] in ("wilson", "bootstrap_multinomial"), (metric_id, cis[metric_id])


def test_imbalance_ratio_with_vanishing_minority_is_skipped():
    """소수 클래스가 1건이면 재표집의 약 37%에서 사라져 비가 무한대가 된다 — 구간을 내지 않는다."""
    rows = [(0, 0)] * 60 + [(1, 1)]
    ci = _binary(rows, ["M23"])["confidence_intervals"]["M23"]

    assert ci["method"] == "skipped"
    assert ci["reason"] == "재표집에서 정답 클래스가 사라지는 경우가 있어 구간 끝을 정할 수 없음"


def test_point_estimate_mismatch_is_skipped():
    df = _synthetic_scores(120)
    mapping = {"y_true": "y_true", "y_pred": "y_pred", "_positive_class": 1, "_beta": 1.0, "_task_type": "binary"}
    results = {"M1": 0.5, "M4": 0.1234}  # 일부러 틀린 지표 값
    cis = intervals.calculate_confidence_intervals(df, mapping, ["M1", "M4"], results)

    assert cis["M1"] == {"low": None, "high": None, "method": "skipped", "level": 0.95, "reason": "점추정 불일치"}
    assert cis["M4"]["reason"] == "점추정 불일치"


# ── 작업량 상한 ─────────────────────────────────────────────────────────

def test_work_cap_switches_score_metrics_to_analytic(monkeypatch):
    df = _synthetic_scores(400)
    metrics = ["M9", "M10", "M19"]
    meta = {"positive_class": "1", "negative_class": "0"}
    boot = evaluate(df, BINARY_SCORE, "binary", metrics, positive_class="1", metadata=meta)["confidence_intervals"]

    monkeypatch.setattr(intervals, "ROW_BOOTSTRAP_WORK_CAP", 1000)
    result = evaluate(df, BINARY_SCORE, "binary", metrics, positive_class="1", metadata=meta)
    analytic = result["confidence_intervals"]

    assert [analytic[m]["method"] for m in metrics] == ["delong", "logit", "normal"]
    for metric_id in metrics:
        a, b = analytic[metric_id], boot[metric_id]
        assert "n_resamples" not in a
        assert a["low"] < result[metric_id] < a["high"]
        # 표본 400 에서 해석적 구간과 재표집 구간은 끝이 0.03 안에서 맞는다.
        assert abs(a["low"] - b["low"]) < 0.03 and abs(a["high"] - b["high"]) < 0.03, (metric_id, a, b)


def test_delong_hand_computed():
    """양성 [0.9, 0.6], 음성 [0.7, 0.2]: AUC = 3/4.

    V10 = [1, 0.5], V01 = [0.5, 1] → 분산 = 0.125/2 + 0.125/2 = 0.125, se = 0.35355.
    """
    auc, ci = intervals._delong(np.array([0.9, 0.6, 0.7, 0.2]), np.array([1.0, 1.0, 0.0, 0.0]))
    assert auc == 0.75
    assert (ci["low"], ci["high"]) == (round(0.75 - 1.959964 * 0.353553, 4), 1.0)


def test_work_cap_multilabel_normal_or_skipped(monkeypatch):
    df, mappings, metrics, meta = _csv_case("multilabel")
    monkeypatch.setattr(intervals, "ROW_BOOTSTRAP_WORK_CAP", 1000)
    cis = evaluate(df, mappings, "multilabel", metrics, metadata=meta)["confidence_intervals"]

    assert cis["M15"]["method"] == "normal"
    assert cis["M17"]["method"] == "normal"
    assert cis["M16"]["method"] == "wilson"
    assert cis["M23"]["method"] == "bootstrap_multinomial"  # 클래스 수 재표집 — 행 상한과 무관
    for metric_id in ("M2", "M3", "M4", "M5", "M18"):
        assert cis[metric_id]["method"] == "skipped"
        assert "상한" in cis[metric_id]["reason"]


def test_cell_cap_skips_confusion_bootstrap(monkeypatch):
    monkeypatch.setattr(intervals, "CELL_BOOTSTRAP_WORK_CAP", 10)
    rows = [(1, 1), (0, 0), (1, 0), (0, 1)] * 5
    cis = _binary(rows, ["M1", "M4"])["confidence_intervals"]

    assert cis["M1"]["method"] == "wilson"
    assert cis["M4"]["method"] == "skipped"
    assert "칸" in cis["M4"]["reason"]


def test_worst_case_at_work_cap_stays_within_budget():
    """작업량 상한 바로 아래(행 = 상한 / 2,000)에서 점수 지표 셋을 다 골라도 수 초 안.

    상한을 한 행 넘기면 해석적 방법으로 바뀌어 더 빨라진다 — 최악은 상한 바로 아래다.
    """
    n = intervals.ROW_BOOTSTRAP_WORK_CAP // intervals.N_RESAMPLES
    df = _synthetic_scores(n, seed=11)
    mapping = {"y_true": "y_true", "y_pred": "y_pred", "score_positive": "score",
               "_positive_class": 1, "_beta": 1.0, "_task_type": "binary"}
    from app.evaluation.metrics import binary

    results = {"M9": binary.calculate_auroc(df, mapping), "M10": binary.calculate_auprc(df, mapping),
               "M19": binary.calculate_log_loss(df, mapping)}

    started = time.perf_counter()
    cis = intervals.calculate_confidence_intervals(df, mapping, ["M9", "M10", "M19"], results)
    elapsed = time.perf_counter() - started

    assert {ci["method"] for ci in cis.values()} == {"bootstrap"}
    assert elapsed < 6.0, f"신뢰구간 계산이 {elapsed:.2f}초 — 작업량 상한을 낮춰야 한다"

    over = _synthetic_scores(n + 1, seed=11)
    cis_over = intervals.calculate_confidence_intervals(over, mapping, ["M9"], {
        "M9": binary.calculate_auroc(over, mapping)})
    assert cis_over["M9"]["method"] == "delong"
