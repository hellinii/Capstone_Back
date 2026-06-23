import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import precision_recall_fscore_support

def _get_true_pred(df: pd.DataFrame, mapping_dict: dict):
    true_col = mapping_dict.get('y_true')
    pred_col = mapping_dict.get('y_pred')
    if not true_col or not pred_col:
        raise ValueError("y_true 및 y_pred 컬럼 매핑이 필요합니다.")
    return df[true_col], df[pred_col]

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

