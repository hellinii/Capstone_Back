"""
benchmark_baselines.py — 지표별 "내부 참조 기준치" 정적 테이블.

⚠️ 이 값들은 "공개 벤치마크 평균"이 아니라 사내 합의 보수 기준(내부 참조 기준치)이다.
   허위 권위(authoritative benchmark claim)를 피하기 위해 반드시 SOURCE_NOTE 와 함께 표기한다.

향후 실제 도메인 벤치마크(RAG/CSV)로 교체할 경우 get_baseline() 인터페이스만 유지하면 된다.
키는 지표 display_name(프론트 METRICS 기준: "Accuracy", "F1 Score", "AUROC" 등) 사용.
"""
from typing import Optional

SOURCE_NOTE = "내부 참조 기준치 v1 (공개 벤치마크 평균이 아닌 사내 합의 보수 기준이며, 도메인별로 상이할 수 있음)"

# task_type → {지표 display_name: (range_low, range_high)}
BASELINES: dict[str, dict[str, tuple[float, float]]] = {
    "binary": {
        "Accuracy":  (0.85, 0.92),
        "Precision": (0.80, 0.92),
        "Recall":    (0.80, 0.92),
        "F1 Score":  (0.83, 0.91),
        "AUROC":     (0.85, 0.95),
        "AUPRC":     (0.80, 0.93),
        "MCC":       (0.60, 0.85),
    },
    "multiclass": {
        "Accuracy":  (0.80, 0.90),
        "Precision": (0.78, 0.89),
        "Recall":    (0.78, 0.89),
        "F1 Score":  (0.78, 0.89),
    },
    "multilabel": {
        "F1 Score":      (0.70, 0.85),
        "Jaccard Index": (0.65, 0.82),
        "Hamming Loss":  (0.05, 0.20),  # 낮을수록 좋음(해석은 narrator/fallback에서 처리)
    },
}


def get_baseline(task_type: str, metric_name: str) -> Optional[tuple[float, float]]:
    """해당 task_type·지표의 참조 범위 (range_low, range_high). 없으면 None."""
    return BASELINES.get(task_type, {}).get(metric_name)


def benchmark_position(value: float, low: float, high: float) -> str:
    """value 가 참조 범위 대비 어디에 위치하는지: below | within | above"""
    if value < low:
        return "below"
    if value > high:
        return "above"
    return "within"


def build_benchmark_refs(task_type: str, metrics: list) -> list[dict]:
    """
    fact_sheet.metrics 각 지표를 참조 테이블과 대조해 position 을 미리 계산한다.
    LLM 은 이 position 을 산문화만 하고 숫자 비교를 직접 하지 않는다.
    기준표에 없는 지표는 제외(→ LLM 에 '비교 데이터 없음'으로 전달).

    metrics: MetricFact 리스트 (또는 .display_name/.value 속성을 가진 객체)
    """
    refs = []
    for m in metrics:
        name = getattr(m, "display_name", None)
        value = getattr(m, "value", None)
        if name is None or value is None:
            continue
        rng = get_baseline(task_type, name)
        if rng is None:
            continue
        low, high = rng
        refs.append({
            "metric": name,
            "model_value": round(float(value), 4),
            "ref_low": low,
            "ref_high": high,
            "position": benchmark_position(float(value), low, high),
            "source_note": SOURCE_NOTE,
        })
    return refs
