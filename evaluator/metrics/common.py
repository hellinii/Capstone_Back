import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    fbeta_score,
    confusion_matrix,
    classification_report
)

def _get_true_pred(df: pd.DataFrame, mapping_dict: dict):
    true_col = mapping_dict.get('y_true') or mapping_dict.get('true_labels')
    pred_col = mapping_dict.get('y_pred') or mapping_dict.get('pred_labels')
    if not true_col or not pred_col:
        raise ValueError("y_true(또는 true_labels) 및 y_pred(또는 pred_labels) 컬럼 매핑이 필요합니다.")
        
    y_true = df[true_col]
    y_pred = df[pred_col]
    
    # Multilabel (리스트)일 경우 2D 이진 배열로 변환
    if not y_true.empty and isinstance(y_true.iloc[0], list):
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        # 정답과 예측의 모든 클래스를 수집하여 fit
        mlb.fit(y_true.tolist() + y_pred.tolist())
        return mlb.transform(y_true), mlb.transform(y_pred)
        
    return y_true, y_pred

def calculate_accuracy(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC1: Accuracy"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    return float(accuracy_score(y_true, y_pred))

def calculate_precision(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC2: Precision (Macro Average)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    # 클래스 이름이 문자열일 경우를 대비해 가장 안전한 macro average 사용
    return float(precision_score(y_true, y_pred, average='macro', zero_division=0))

def calculate_recall(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC3: Recall (Macro Average)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    return float(recall_score(y_true, y_pred, average='macro', zero_division=0))

def calculate_f1_score(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC4: F1 Score (Macro Average)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))

def calculate_fbeta_score(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC5: F-beta Score (Macro Average)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    beta = mapping_dict.get('_beta', 1.0)
    return float(fbeta_score(y_true, y_pred, beta=beta, average='macro', zero_division=0))

def calculate_kl_divergence(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC6: KL Divergence"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    # 확률 분포 계산 (normalize=True)
    p_counts = pd.Series(y_true).value_counts(normalize=True)
    q_counts = pd.Series(y_pred).value_counts(normalize=True)
    
    # 0으로 나누는 것을 방지하기 위해 아주 작은 값(epsilon) 추가
    epsilon = 1e-9
    p = np.array([p_counts.get(c, epsilon) for c in classes])
    q = np.array([q_counts.get(c, epsilon) for c in classes])
    
    kl_div = np.sum(p * np.log(p / q))
    return float(kl_div)

def calculate_confusion_matrix(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, Any]:
    """TC21: Confusion Matrix"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    
    # Multilabel의 경우 (2D 배열)
    if isinstance(y_true, np.ndarray) and y_true.ndim > 1:
        from sklearn.metrics import multilabel_confusion_matrix
        cm = multilabel_confusion_matrix(y_true, y_pred)
        return {
            "type": "multilabel",
            "matrix": cm.tolist() # 클래스별 2x2 행렬의 리스트
        }
        
    labels = sorted(np.unique(np.concatenate((y_true, y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "type": "multiclass_or_binary",
        "labels": [str(l) for l in labels],
        "matrix": cm.tolist()
    }

def calculate_class_metrics(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, Any]:
    """TC22: Class별 Metric"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    # output_dict=True를 통해 JSON 형태로 바로 내려주기 좋음
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report

def calculate_imbalance_ratio(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC23: Imbalance Ratio (가장 많은 클래스 / 가장 적은 클래스 비율)"""
    true_col = mapping_dict.get('y_true') or mapping_dict.get('true_labels')
    if not true_col:
        raise ValueError("y_true 또는 true_labels 컬럼 매핑이 필요합니다.")
    
    y_true = df[true_col]
    if not y_true.empty and isinstance(y_true.iloc[0], list):
        # 다중 레이블: 리스트를 평탄화하여 클래스별 빈도수 계산
        all_labels = [label for sublist in y_true for label in sublist]
        counts = pd.Series(all_labels).value_counts()
    else:
        counts = y_true.value_counts()
        
    if len(counts) == 0:
        return 0.0
    
    majority = counts.max()
    minority = counts.min()
    
    if minority == 0:
        return float('inf')
        
    return float(majority / minority)

