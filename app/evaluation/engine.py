"""app/evaluation/engine.py — 평가 지표 계산 오케스트레이션(디스패치)

전처리(preprocess_data) 후 METRIC_REQUIREMENTS 로부터 계산 가능 지표 를 정하고, METRIC_REGISTRY
로 각 지표 를 실제 계산 함수(metrics/*)에 디스패치한다. binary ROC/PR 곡선·latency 통계도 부착.

상호작용
- 의존(import): pandas, app.core.schemas(METRIC_REQUIREMENTS), .preprocessor, .metrics(common/binary/multiclass/multilabel)
- 사용처: app.evaluation.service.run_evaluation_pipeline
"""

import pandas as pd
from typing import Dict, Any, List

from .metrics import common, binary, multiclass, multilabel
from .preprocessor import preprocess_data
from .errors import metric_error
from app.core.schemas import METRIC_REQUIREMENTS

# Task Type 별로 허용되는 지표 정의 (core.schemas 의 단일 출처에서 동적 생성)
VALID_METRICS_BY_TASK = {
    task_type.value: set(requirements.keys())
    for task_type, requirements in METRIC_REQUIREMENTS.items()
}

# 지표 ID 와 실제 계산 함수 매핑 (Registry)
METRIC_REGISTRY = {
    "M1": common.calculate_accuracy,
    "M2": common.calculate_precision,
    "M3": common.calculate_recall,
    "M4": common.calculate_f1_score,
    "M5": common.calculate_fbeta_score,
    "M6": common.calculate_kl_divergence,
    "M7": binary.calculate_specificity,
    "M8": binary.calculate_fpr,
    "M9": binary.calculate_auroc,
    "M10": binary.calculate_auprc,
    "M11": multiclass.calculate_macro_average,
    "M12": multiclass.calculate_micro_average,
    "M13": multiclass.calculate_weighted_average,
    "M14": multiclass.calculate_distribution_diff_mc,
    "M15": multilabel.calculate_hamming_loss,
    "M16": multilabel.calculate_exact_match_ratio,
    "M17": multilabel.calculate_jaccard_index,
    "M18": multilabel.calculate_distribution_diff_ml,
    "M19": binary.calculate_log_loss,
    "M20": binary.calculate_mcc,
    "M21": common.calculate_confusion_matrix,
    "M22": common.calculate_class_metrics,
    "M23": common.calculate_imbalance_ratio,
}


def evaluate(
    df: pd.DataFrame,
    mappings: List[Dict[str, str]],
    task_type: str,
    selected_metric_ids: List[str],
    positive_class: str | None = None,
    beta: float = 1.0,
    decision_threshold=None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    메인 평가 엔진 진입점.
    Task Type에 검증된 지표들만 필터링하여 동적으로 실행합니다.
    
    Args:
        df: 전처리/검증이 완료된 DataFrame
        mappings: 프론트에서 확정하여 전달한 역할 매핑 리스트 [{"column": "col_A", "role": "true_class"}, ...]
        task_type: "binary" | "multiclass" | "multilabel"
        selected_metric_ids: 클라이언트가 요청한 평가 지표 리스트 ["M1", "M7"]
        positive_class: Binary 평가 시 양성(Positive)으로 간주할 값
        beta: F-beta score 계산용 가중치 beta 값
        
    Returns:
        평가 결과 (최종 리포트 딕셔너리 형태)
    """
    results = {}
    valid_metric_ids = VALID_METRICS_BY_TASK.get(task_type, set())
    
    # ── [전처리 단계 추가] ──
    try:
        df, pre_logs = preprocess_data(
            df, mappings, task_type,
            decision_threshold=decision_threshold, metadata=metadata,
        )
        results["_metadata"] = pre_logs
    except Exception as e:
        # 전처리 단계에서 에러(예: 필수 컬럼 누락, 확률 범위 초과)가 나면 즉시 반환
        return {"error": str(e)}

    # mappings 리스트를 딕셔너리로 변환 ({"y_true": "col_A", ...} 형태)
    mapping_dict = {item['role']: item['column'] for item in mappings}

    # 전처리가 확률에서 예측을 파생했다면 그 컬럼을 예측 역할로 등록한다.
    # 이 mapping_dict 는 preprocess **이후에** 다시 만들어지므로, df 에 컬럼만 붙여서는
    # metrics 가 그것을 찾지 못한다(ISSUES.md A-01).
    derived = pre_logs.get("derived_prediction")
    if derived:
        mapping_dict[derived["target_role"]] = derived["column"]
        results["derived_prediction"] = {
            k: v for k, v in derived.items() if k != "column"
        }

    # 파라미터 전달용 내부 변수 설정
    mapping_dict['_positive_class'] = positive_class
    mapping_dict['_beta'] = beta
    mapping_dict['_task_type'] = task_type
    
    for metric_id in selected_metric_ids:
        if metric_id not in valid_metric_ids:
            results[metric_id] = metric_error(f"{task_type}에서는 지원하지 않는 지표입니다.")
            continue
            
        if metric_id in METRIC_REGISTRY:
            func = METRIC_REGISTRY[metric_id]
            try:
                # 매핑된 계산 함수 실행
                results[metric_id] = func(df, mapping_dict)
            except Exception as e:
                # 에러가 나더라도 다른 지표 계산에 영향을 주지 않도록 격리
                results[metric_id] = metric_error(str(e))
        else:
            results[metric_id] = metric_error("구현되지 않은 지표입니다.")

    # ── 차트용 곡선 좌표 (binary, 스칼라 AUROC/AUPRC 산출 성공 시 함께 제공) ──
    # 별도 지표가 아니라 success_metrics 의 roc_curve/pr_curve 키로 내려보낸다.
    if task_type == "binary":
        if "M9" in selected_metric_ids and isinstance(results.get("M9"), (int, float)):
            try:
                results["roc_curve"] = binary.calculate_roc_curve(df, mapping_dict)
            except Exception:
                pass
        if "M10" in selected_metric_ids and isinstance(results.get("M10"), (int, float)):
            try:
                results["pr_curve"] = binary.calculate_pr_curve(df, mapping_dict)
            except Exception:
                pass

    # ── 지연시간(Latency) 통계 (선택 컬럼, 모든 task_type, 단위 ms 가정) ──
    # 별도 지표가 아니라 success_metrics 의 latency_stats 키로 내려보낸다.
    latency_col = mapping_dict.get("latency")
    if latency_col and latency_col in df.columns:
        try:
            lat = pd.to_numeric(df[latency_col], errors="coerce").dropna()
            if len(lat) > 0:
                results["latency_stats"] = {
                    "count": int(len(lat)),
                    "mean": float(lat.mean()),
                    "min": float(lat.min()),
                    "p50": float(lat.quantile(0.50)),
                    "p95": float(lat.quantile(0.95)),
                    "p99": float(lat.quantile(0.99)),
                    "max": float(lat.max()),
                    "unit": "ms",
                }
        except Exception:
            pass

    return results
