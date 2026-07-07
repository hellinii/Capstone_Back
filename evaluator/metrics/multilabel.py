import pandas as pd
import numpy as np
import ast
from typing import Dict, Any
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, accuracy_score, jaccard_score

def _parse_multilabel_col(series: pd.Series):
    """
    CSV 등에 저장될 때 '["A", "B"]' 형태의 문자열로 들어오는 경우
    실제 파이썬 리스트로 안전하게 파싱합니다.
    """
    def parse_item(item):
        if isinstance(item, list): return item
        if isinstance(item, str):
            try:
                parsed = ast.literal_eval(item)
                if isinstance(parsed, list): return parsed
            except:
                return [x.strip() for x in item.split(',') if x.strip()]
        return [item]
    
    return series.apply(parse_item).tolist()

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
    """TC15: Hamming Loss"""
    y_true_bin, y_pred_bin = _get_binarized_true_pred(df, mapping_dict)
    return float(hamming_loss(y_true_bin, y_pred_bin))

def calculate_exact_match_ratio(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC16: Exact Match Ratio (Subset Accuracy)"""
    y_true_bin, y_pred_bin = _get_binarized_true_pred(df, mapping_dict)
    return float(accuracy_score(y_true_bin, y_pred_bin))

def calculate_jaccard_index(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC17: Jaccard Index (Samples Average)"""
    y_true_bin, y_pred_bin = _get_binarized_true_pred(df, mapping_dict)
    return float(jaccard_score(y_true_bin, y_pred_bin, average='samples', zero_division=0))

def calculate_distribution_diff_ml(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC18: Distribution Diff (ML) - 레이블 빈도수 벡터 간의 코사인 거리 사용"""
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

