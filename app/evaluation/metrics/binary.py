"""app/evaluation/metrics/binary.py — 이진 분류 전용 지표(sklearn 기반)

특이도/FPR/AUROC/AUPRC/LogLoss/MCC 및 ROC·PR 곡선 좌표 계산.

상호작용
- 의존(import): pandas, sklearn
- 사용처: app.evaluation.engine(METRIC_REGISTRY, 곡선 부착)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    log_loss,
    matthews_corrcoef
)

def _get_true_pred(df: pd.DataFrame, mapping_dict: dict):
    true_col = mapping_dict.get('y_true')
    pred_col = mapping_dict.get('y_pred')
    if not true_col or not pred_col:
        raise ValueError("y_true 및 y_pred 컬럼 매핑이 필요합니다.")
    return df[true_col], df[pred_col]

def _get_true_score(df: pd.DataFrame, mapping_dict: dict):
    true_col = mapping_dict.get('y_true')
    score_col = mapping_dict.get('score_positive')
    if not true_col or not score_col:
        raise ValueError("y_true 및 score_positive 컬럼 매핑이 필요합니다.")
    return df[true_col], df[score_col]

def _binarize_true_labels(y_true, positive_class=None):
    """문자열 등 비숫자형 라벨을 0, 1로 안전하게 변환"""
    if positive_class is not None:
        # positive_class가 문자열 형태일 수 있으므로 원래 데이터 타입과 맞춤 비교
        try:
            if np.issubdtype(y_true.dtype, np.number):
                pos_val = type(y_true.iloc[0])(positive_class)
            else:
                pos_val = str(positive_class)
        except Exception:
            pos_val = positive_class
        return (y_true == pos_val).astype(int)

    classes = np.sort(np.unique(y_true))
    if len(classes) <= 2:
        pos_label = classes[-1]  # 정렬 시 마지막 값을 positive(1)로 간주 (e.g. 'Yes', 1)
        return (y_true == pos_label).astype(int)
    return y_true

def calculate_specificity(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC7: Specificity (True Negative Rate)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return 0.0

def calculate_fpr(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC8: False Positive Rate (FPR)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return 0.0

def calculate_auroc(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC9: Area Under the Receiver Operating Characteristic Curve (AUROC)"""
    y_true, y_score = _get_true_score(df, mapping_dict)
    pos_class = mapping_dict.get('_positive_class')
    y_true_bin = _binarize_true_labels(y_true, pos_class)
    return float(roc_auc_score(y_true_bin, y_score))

def calculate_auprc(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC10: Area Under the Precision-Recall Curve (AUPRC)"""
    y_true, y_score = _get_true_score(df, mapping_dict)
    pos_class = mapping_dict.get('_positive_class')
    y_true_bin = _binarize_true_labels(y_true, pos_class)
    return float(average_precision_score(y_true_bin, y_score))

def calculate_log_loss(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC19: Log Loss"""
    y_true, y_score = _get_true_score(df, mapping_dict)
    pos_class = mapping_dict.get('_positive_class')
    y_true_bin = _binarize_true_labels(y_true, pos_class)
    return float(log_loss(y_true_bin, y_score))

def calculate_mcc(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC20: Matthews Correlation Coefficient (MCC)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    return float(matthews_corrcoef(y_true, y_pred))


def _downsample_pair(x, y, n: int = 60):
    """
    쌍을 이루는 두 좌표 배열(x, y)을 동일 인덱스로 균등 다운샘플링.
    차트 렌더링/응답 페이로드 경량화를 위해 점 개수를 n개 이하로 줄인다.
    """
    x = list(x)
    y = list(y)
    L = len(x)
    if L <= n:
        return [float(v) for v in x], [float(v) for v in y]
    idxs = sorted({round(i * (L - 1) / (n - 1)) for i in range(n)})
    return [float(x[i]) for i in idxs], [float(y[i]) for i in idxs]


def calculate_roc_curve(df: pd.DataFrame, mapping_dict: dict) -> dict:
    """ROC 곡선 좌표 (차트용). AUROC(TC9)와 동일 입력을 사용한다."""
    y_true, y_score = _get_true_score(df, mapping_dict)
    pos_class = mapping_dict.get('_positive_class')
    y_true_bin = _binarize_true_labels(y_true, pos_class)
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    fpr, tpr = _downsample_pair(fpr, tpr)
    return {"fpr": fpr, "tpr": tpr}


def calculate_pr_curve(df: pd.DataFrame, mapping_dict: dict) -> dict:
    """Precision-Recall 곡선 좌표 (차트용). AUPRC(TC10)와 동일 입력을 사용한다."""
    y_true, y_score = _get_true_score(df, mapping_dict)
    pos_class = mapping_dict.get('_positive_class')
    y_true_bin = _binarize_true_labels(y_true, pos_class)
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    # recall 오름차순으로 정렬(차트 x축 기준)
    recall = recall[::-1]
    precision = precision[::-1]
    recall, precision = _downsample_pair(recall, precision)
    return {"recall": recall, "precision": precision}

