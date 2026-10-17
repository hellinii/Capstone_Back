"""app/evaluation/metrics/binary.py — 이진 분류 전용 지표(sklearn 기반)

특이도/FPR/AUROC/AUPRC/LogLoss/MCC 및 ROC·PR 곡선 좌표, 임계값 조정 여지(f1_threshold_gain) 계산.

상호작용
- 의존(import): pandas, sklearn
- 사용처: app.evaluation.engine(METRIC_REGISTRY, 곡선·임계값 조정 여지 부착)
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

def _resolve_positive_value(y_true, positive_class=None):
    """양성으로 간주할 실제 값을 결정한다.

    positive_class 가 주어지면 원래 데이터 타입에 맞춰 변환해 쓰고,
    없으면 정렬상 마지막 값을 양성으로 본다(e.g. 'Yes', 1).
    """
    if positive_class is not None:
        # positive_class가 문자열 형태일 수 있으므로 원래 데이터 타입과 맞춤 비교
        try:
            if np.issubdtype(y_true.dtype, np.number):
                return type(y_true.iloc[0])(positive_class)
            return str(positive_class)
        except Exception:
            return positive_class

    classes = np.sort(np.unique(y_true))
    return classes[-1] if len(classes) > 0 else 1

def _binarize_true_labels(y_true, positive_class=None):
    """문자열 등 비숫자형 라벨을 0, 1로 안전하게 변환"""
    if positive_class is not None:
        return (y_true == _resolve_positive_value(y_true, positive_class)).astype(int)

    classes = np.sort(np.unique(y_true))
    if len(classes) <= 2:
        return (y_true == _resolve_positive_value(y_true)).astype(int)
    return y_true

def _binary_confusion_counts(df: pd.DataFrame, mapping_dict: dict):
    """M7/M8 공용. positive_class 기준으로 0/1 이진화 후 labels=[0,1] 고정 혼동행렬.

    labels 를 고정하지 않으면 관측 클래스가 1종일 때(희귀 양성 필터링 데이터 등) 행렬이
    1x1 이 되어 tn/fp/fn/tp 분해가 불가능하다. 종전에는 그 경우 지표를 0.0 으로 반환했는데,
    M8(FPR)은 낮을수록 좋은 지표라 "오탐률 0%"라는 허위 우수 판정이 됐다.
    y_pred 에 미지 라벨이 섞이는 경우도 여기서 '양성 아님(=음성)'으로 일관 처리된다.
    """
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    pos_val = _resolve_positive_value(y_true, mapping_dict.get('_positive_class'))
    y_true_bin = (y_true == pos_val).astype(int)
    y_pred_bin = (y_pred == pos_val).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)

def calculate_specificity(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M7: Specificity (True Negative Rate)"""
    tn, fp, _fn, _tp = _binary_confusion_counts(df, mapping_dict)
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

def calculate_fpr(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M8: False Positive Rate (FPR)"""
    tn, fp, _fn, _tp = _binary_confusion_counts(df, mapping_dict)
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

def calculate_auroc(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M9: Area Under the Receiver Operating Characteristic Curve (AUROC)"""
    y_true, y_score = _get_true_score(df, mapping_dict)
    pos_class = mapping_dict.get('_positive_class')
    y_true_bin = _binarize_true_labels(y_true, pos_class)
    return float(roc_auc_score(y_true_bin, y_score))

def calculate_auprc(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M10: Area Under the Precision-Recall Curve (AUPRC)"""
    y_true, y_score = _get_true_score(df, mapping_dict)
    pos_class = mapping_dict.get('_positive_class')
    y_true_bin = _binarize_true_labels(y_true, pos_class)
    return float(average_precision_score(y_true_bin, y_score))

def calculate_log_loss(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M19: Log Loss"""
    y_true, y_score = _get_true_score(df, mapping_dict)
    pos_class = mapping_dict.get('_positive_class')
    y_true_bin = _binarize_true_labels(y_true, pos_class)
    return float(log_loss(y_true_bin, y_score))

def calculate_mcc(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M20: Matthews Correlation Coefficient (MCC)"""
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
    """ROC 곡선 좌표 (차트용). AUROC(M9)와 동일 입력을 사용한다."""
    y_true, y_score = _get_true_score(df, mapping_dict)
    pos_class = mapping_dict.get('_positive_class')
    y_true_bin = _binarize_true_labels(y_true, pos_class)
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    fpr, tpr = _downsample_pair(fpr, tpr)
    return {"fpr": fpr, "tpr": tpr}


def calculate_pr_curve(df: pd.DataFrame, mapping_dict: dict) -> dict:
    """Precision-Recall 곡선 좌표 (차트용). AUPRC(M10)와 동일 입력을 사용한다."""
    y_true, y_score = _get_true_score(df, mapping_dict)
    pos_class = mapping_dict.get('_positive_class')
    y_true_bin = _binarize_true_labels(y_true, pos_class)
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    # recall 오름차순으로 정렬(차트 x축 기준)
    recall = recall[::-1]
    precision = precision[::-1]
    recall, precision = _downsample_pair(recall, precision)
    return {"recall": recall, "precision": precision}



def calculate_f1_threshold_gain(df: pd.DataFrame, mapping_dict: dict) -> dict:
    """임계값 조정 여지(서술 설계 D40) — 원본 PR 곡선의 최대 F1 과 지금 예측의 F1 차이.

    지표가 아니라 success_metrics 의 f1_threshold_gain 키로 내려보낸다(이진·score 가 있을 때).
    곡선은 **다운샘플 전** `precision_recall_curve` 를 쓴다 — 차트용 `pr_curve` 는 60점으로
    줄여 최댓값을 놓칠 수 있다.

    - max_f1·best_threshold: 곡선의 점마다 F1 = 2PR/(P+R)(P+R=0 이면 0). 동률은 낮은 임계값.
    - current_f1: 평가에 실제로 쓴 예측(직접 올린 y_pred 또는 파생 예측)의 양성 F1.
      양성 기준은 TP·FP 건수(counts)·M7·M8 과 같다(`_binary_confusion_counts`).
    - gain = max(0, max_f1 − current_f1). y_pred 를 직접 올리면 그 예측이 곡선 위의 점이
      아닐 수 있어 차이가 음수가 될 수 있다 — 그때는 "올릴 여지 없음"이므로 0.
    값은 4자리로 반올림해 싣는다. "≥ 0.10 이면 임계값 부적합" 판정은 프론트가 실린 gain 으로 한다.
    """
    y_true, y_score = _get_true_score(df, mapping_dict)
    positive = _resolve_positive_value(y_true, mapping_dict.get('_positive_class'))
    y_true_bin = (y_true == positive).astype(int)
    if y_true_bin.sum() == 0:
        raise ValueError("정답에 양성이 없어 PR 곡선의 F1 을 정의할 수 없습니다.")

    precision, recall, thresholds = precision_recall_curve(y_true_bin, y_score)
    # 마지막 점(P=1, R=0)은 임계값이 없는 곡선 끝이라 뺀다(F1 도 0).
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    best = int(np.argmax(f1))  # 동률이면 첫 점 = 가장 낮은 임계값

    _tn, fp, fn, tp = _binary_confusion_counts(df, mapping_dict)
    current = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    max_f1 = float(f1[best])
    return {
        "max_f1": round(max_f1, 4),
        "current_f1": round(current, 4),
        "gain": round(max(0.0, max_f1 - current), 4),
        "best_threshold": round(float(thresholds[best]), 4),
    }
