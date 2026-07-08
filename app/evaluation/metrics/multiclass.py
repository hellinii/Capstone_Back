"""app/evaluation/metrics/multiclass.py — 다중클래스 전용 지표(sklearn 기반)

macro/micro/weighted 평균과 분포 차이(distribution_diff) 계산. common 의 헬퍼를 일부 재사용.

상호작용
- 의존(import): pandas, sklearn, .common
- 사용처: app.evaluation.engine(METRIC_REGISTRY)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import precision_recall_fscore_support

from .common import _get_true_pred

def calculate_macro_average(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, float]:
    """TC11: Macro Average"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1_score": float(f1)}

def calculate_micro_average(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, float]:
    """TC12: Micro Average"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1_score": float(f1)}

def calculate_weighted_average(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, float]:
    """TC13: Weighted Average"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1_score": float(f1)}

def calculate_distribution_diff_mc(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC14: Distribution Diff (MC) - Total Variation Distance (TVD) 적용"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    p_counts = pd.Series(y_true).value_counts(normalize=True)
    q_counts = pd.Series(y_pred).value_counts(normalize=True)
    
    # 클래스 분포 간의 절대 차이 합의 절반 (TVD)
    tvd = 0.5 * sum(abs(p_counts.get(c, 0.0) - q_counts.get(c, 0.0)) for c in classes)
    return float(tvd)

