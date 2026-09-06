"""app/evaluation/metrics/multilabel.py — 다중레이블 전용 지표(sklearn 기반)

Hamming Loss/Exact Match Ratio/Jaccard/분포 차이 계산. 파이프(|) 구분 라벨을 이진화해 계산.

상호작용
- 의존(import): pandas, sklearn
- 사용처: app.evaluation.engine(METRIC_REGISTRY)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, accuracy_score, jaccard_score

from app.evaluation.labels import parse_label_cell, sort_labels

def _parse_multilabel_col(series: pd.Series):
    """멀티레이블 컬럼 → 라벨 리스트의 리스트.

    파싱 규칙은 `evaluation.labels.parse_label_cell` 하나뿐이다(ISSUES.md D-04).
    종전에는 여기에 같은 규칙의 사본이 있었고, 구분자 처리가 조금씩 달라 같은 셀이
    계층마다 다른 라벨 집합이 됐다.
    """
    return series.apply(parse_label_cell).tolist()


def _get_binarized_true_pred(df: pd.DataFrame, mapping_dict: dict):
    """
    MultiLabelBinarizer를 사용해 One-Hot Vector 형태(2D Array)로 변환
    """
    true_col = mapping_dict.get('true_labels')
    pred_col = mapping_dict.get('pred_labels')
    if not true_col or not pred_col:
        raise ValueError("true_labels 및 pred_labels 컬럼 매핑이 필요합니다.")
        
    y_true_list = _parse_multilabel_col(df[true_col])
    y_pred_list = _parse_multilabel_col(df[pred_col])
    
    mlb = MultiLabelBinarizer()
    # 전체 등장 가능한 클래스들을 수집하여 피팅
    mlb.fit(y_true_list + y_pred_list)
    
    return mlb.transform(y_true_list), mlb.transform(y_pred_list)

def calculate_hamming_loss(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M15: Hamming Loss"""
    y_true_bin, y_pred_bin = _get_binarized_true_pred(df, mapping_dict)
    return float(hamming_loss(y_true_bin, y_pred_bin))

def calculate_exact_match_ratio(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M16: Exact Match Ratio (Subset Accuracy)"""
    y_true_bin, y_pred_bin = _get_binarized_true_pred(df, mapping_dict)
    return float(accuracy_score(y_true_bin, y_pred_bin))

def calculate_jaccard_index(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M17: Jaccard Index (Samples Average)

    zero_division=1: 정답·예측이 모두 빈 레이블 집합인 샘플(0/0)은 '일치'로 센다.
    전처리가 결측을 ''로 채워 살려두므로 빈 레이블 행은 이 시스템의 정상 입력이고,
    zero_division=0 이면 완벽 예측인데도 M17만 깎이는 모순이 생긴다(M16 은 일치로 셈).
    """
    y_true_bin, y_pred_bin = _get_binarized_true_pred(df, mapping_dict)
    return float(jaccard_score(y_true_bin, y_pred_bin, average='samples', zero_division=1))

def calculate_distribution_diff_ml(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M18: Distribution Diff (ML) - 레이블 빈도수 벡터 간의 코사인 거리 사용"""
    y_true_bin, y_pred_bin = _get_binarized_true_pred(df, mapping_dict)
    
    p_freq = np.sum(y_true_bin, axis=0)
    q_freq = np.sum(y_pred_bin, axis=0)
    
    if np.sum(p_freq) == 0 or np.sum(q_freq) == 0:
        return 0.0
        
    dot = np.dot(p_freq, q_freq)
    norm_p = np.linalg.norm(p_freq)
    norm_q = np.linalg.norm(q_freq)
    
    if norm_p == 0 or norm_q == 0:
        return 0.0
        
    cos_sim = dot / (norm_p * norm_q)
    return float(1.0 - cos_sim)

