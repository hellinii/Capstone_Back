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

from .common import _get_true_pred, _label_kwargs

def calculate_macro_average(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, float]:
    """M11: Macro Average

    클래스 집합을 정답(y_true)으로 고정한다 — M2~M5·M22 와 같은 규칙(C-03). 종전에는 labels 인자가 없어 예측에만
    있는 클래스가 분모에 들어가, 한 성적서의 M4 와 M11 매크로 F1 이 0.7302 / 0.5476 처럼 갈렸다(점검 D-V1).
    """
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0, **_label_kwargs(df, mapping_dict, y_true),
    )
    return {"precision": float(p), "recall": float(r), "f1_score": float(f1)}

def calculate_micro_average(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, float]:
    """M12: Micro Average"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1_score": float(f1)}

def calculate_weighted_average(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, float]:
    """M13: Weighted Average"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1_score": float(f1)}

def calculate_distribution_diff_mc(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M14: Distribution Diff (MC) - Total Variation Distance (TVD) 적용"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    p_counts = pd.Series(y_true).value_counts(normalize=True)
    q_counts = pd.Series(y_pred).value_counts(normalize=True)
    
    # 클래스 분포 간의 절대 차이 합의 절반 (TVD)
    tvd = 0.5 * sum(abs(p_counts.get(c, 0.0) - q_counts.get(c, 0.0)) for c in classes)
    return float(tvd)

