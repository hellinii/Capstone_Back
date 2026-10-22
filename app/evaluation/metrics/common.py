"""app/evaluation/metrics/common.py — task 공통 지표 계산(sklearn 기반)

정확도/정밀도/재현율/F1/Fbeta/혼동행렬/클래스별 지표/불균형비 등 세 task 가 공유하는 지표.
engine 의 METRIC_REGISTRY 가 지표 별로 호출한다.

상호작용
- 의존(import): pandas, sklearn
- 사용처: app.evaluation.engine, 그리고 metrics.multiclass/multilabel 이 일부 헬퍼 재사용
"""

import warnings

import pandas as pd
import numpy as np
from typing import Dict, Any

from app.evaluation.labels import normalize_label, parse_label_cell, sort_labels
from app.evaluation.metrics.binary import _resolve_positive_value
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    fbeta_score,
    confusion_matrix,
    classification_report
)

def evaluation_classes(df: pd.DataFrame, mapping_dict: dict) -> list:
    """평가에 쓰이는 클래스 집합 — **`y_true` 기준으로 고정한다**(ISSUES.md C-03).

    예측에만 등장한 클래스(유령 클래스)를 집합에 넣으면 macro 평균의 분모가 늘어 값이
    희석되고, 클래스별 표에 support 0 인 행이 생긴다. 무엇보다 M23(불균형비)과
    `class_distribution` 은 원래 `y_true` 만 보므로 **한 성적서 안에 클래스 집합이
    두 벌**이 된다.

    유령 클래스로 예측한 행은 사라지지 않는다 — 정답 클래스의 오분류(FN)로 남는다.
    """
    true_col = mapping_dict.get('y_true') or mapping_dict.get('true_labels')
    if not true_col:
        return []
    return sort_labels(pd.Series(df[true_col]).dropna().unique())


def ghost_classes(df: pd.DataFrame, mapping_dict: dict) -> list:
    """예측에만 등장하고 정답에는 없는 클래스."""
    pred_col = mapping_dict.get('y_pred') or mapping_dict.get('pred_labels')
    true_col = mapping_dict.get('y_true') or mapping_dict.get('true_labels')
    if not pred_col or not true_col or pred_col not in df.columns:
        return []
    series = pd.Series(df[pred_col]).dropna()
    if series.empty:
        return []
    if isinstance(series.iloc[0], list):
        predicted = {l for cell in series for l in cell}
        truths = {l for cell in pd.Series(df[true_col]).dropna() for l in (cell or [])}
    else:
        predicted = {normalize_label(v) for v in series.unique()}
        truths = set(evaluation_classes(df, mapping_dict))
    return sort_labels(predicted - truths)


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
        # **정답 라벨만으로 fit 한다**(C-03). 예측에만 있는 라벨은 transform 에서 무시되어
        # 해당 샘플의 정답 라벨이 FN 으로 남는다 — 정확히 결정 6-① 이 요구하는 셈법이다.
        # multiclass 는 sklearn 의 labels= 인자로 같은 일을 하지만 multilabel 은 fit 대상을
        # 바꿔야 한다(메커니즘이 다르다).
        mlb.fit([sort_labels({l for cell in y_true for l in cell})])
        mapping_dict['_mlb_classes'] = list(mlb.classes_)
        with warnings.catch_warnings():
            # "unknown class(es) will be ignored" — 의도된 동작이라 삼킨다.
            # 사용자 안내는 preprocess 가 만든 유령 클래스 경고가 담당한다.
            warnings.simplefilter("ignore", UserWarning)
            return mlb.transform(y_true), mlb.transform(y_pred)

    return y_true, y_pred


def class_values(values, names: list) -> list:
    """표준 이름 목록(names) → 데이터에 든 값 목록(같은 순서). 데이터에 없는 이름은 이름 그대로.

    전처리(_normalize_hard_labels)가 정답·예측을 한 표기로 맞췄으므로 이름 하나에 값은 하나다 — 문자열 라벨은
    이름이 곧 값이고, 정수 값 숫자 라벨은 int64 로 남아 이름 '1' 의 값이 1 이다(sklearn 이 숫자 배열을 빠르게
    다루도록 문자열로 바꾸지 않는다). 종전에는 원래 값으로 되찾았는데, ' cat'·'cat' 처럼 같은 이름으로 모이는
    값이 둘이면 하나만 남아 다른 쪽 클래스가 매크로 지표와 클래스별 표에서 통째로 빠졌다(점검 BE-13) — 이제는
    표기가 먼저 합쳐져 그런 값이 없다.
    """
    by_name = {normalize_label(v): v for v in pd.Series(values).dropna().unique()}
    return [by_name.get(n, n) for n in names]


def _label_kwargs(df: pd.DataFrame, mapping_dict: dict, y_true) -> dict:
    """클래스 집합을 y_true 로 고정하는 sklearn 인자(2-D 멀티레이블에는 넘기지 않는다).

    labels 는 데이터 값(class_values)이다 — 이름이 필요하면 normalize_label(값)으로 낸다.
    """
    if isinstance(y_true, np.ndarray) and y_true.ndim > 1:
        return {}
    classes = evaluation_classes(df, mapping_dict)
    if not classes:
        return {}
    return {"labels": class_values(y_true, classes)}

def calculate_accuracy(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M1: Accuracy"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    return float(accuracy_score(y_true, y_pred))

def _get_binary_kwargs(y_true, mapping_dict):
    """이진 분류를 위한 kwargs 반환 (pos_label 결정).

    양성은 binary._resolve_positive_value 한 함수로 정한다 — 지정된 표기('1'·'cat')면 데이터의 원래 값
    (1.0·' cat')을 찾고, 미지정이면 labels.default_positive_class(M5)를 쓴다. 종전에는 미지정일 때
    정렬 뒤쪽(classes[-1])을 써서 fraud/normal 이면 precision·recall·F1 은 normal, binary.py 지표는
    fraud 를 양성으로 봤다.
    """
    pos_label = _resolve_positive_value(y_true, mapping_dict.get('_positive_class'))
    return {"pos_label": pos_label, "average": "binary", "zero_division": 0}

def calculate_precision(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M2: Precision"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    task_type = mapping_dict.get('_task_type', 'multiclass')
    if task_type == 'binary':
        return float(precision_score(y_true, y_pred, **_get_binary_kwargs(y_true, mapping_dict)))
    return float(precision_score(y_true, y_pred, average='macro', zero_division=0,
                        **_label_kwargs(df, mapping_dict, y_true)))

def calculate_recall(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M3: Recall"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    task_type = mapping_dict.get('_task_type', 'multiclass')
    if task_type == 'binary':
        return float(recall_score(y_true, y_pred, **_get_binary_kwargs(y_true, mapping_dict)))
    return float(recall_score(y_true, y_pred, average='macro', zero_division=0,
                        **_label_kwargs(df, mapping_dict, y_true)))

def calculate_f1_score(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M4: F1 Score"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    task_type = mapping_dict.get('_task_type', 'multiclass')
    if task_type == 'binary':
        return float(f1_score(y_true, y_pred, **_get_binary_kwargs(y_true, mapping_dict)))
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0,
                        **_label_kwargs(df, mapping_dict, y_true)))

def calculate_fbeta_score(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M5: F-beta Score"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    task_type = mapping_dict.get('_task_type', 'multiclass')
    beta = mapping_dict.get('_beta', 1.0)
    if task_type == 'binary':
        kwargs = _get_binary_kwargs(y_true, mapping_dict)
        kwargs['beta'] = beta
        return float(fbeta_score(y_true, y_pred, **kwargs))
    return float(fbeta_score(y_true, y_pred, beta=beta, average='macro', zero_division=0,
                             **_label_kwargs(df, mapping_dict, y_true)))

def calculate_kl_divergence(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M6: KL Divergence"""
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
    """M21: Confusion Matrix"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    
    # Multilabel의 경우 (2D 배열)
    if isinstance(y_true, np.ndarray) and y_true.ndim > 1:
        from sklearn.metrics import multilabel_confusion_matrix
        cm = multilabel_confusion_matrix(y_true, y_pred)
        classes = mapping_dict.get('_mlb_classes', [])
        return {
            "type": "multilabel",
            "labels": [str(l) for l in classes] if classes else [f"Class {i}" for i in range(len(cm))],
            "matrix": cm.tolist() # 클래스별 2x2 행렬의 리스트
        }
        
    # 클래스 이름·순서는 분포·M22·건수와 같은 한 벌(sort_labels — 숫자면 수치 순서)이다. 종전 sorted(np.unique)
    # 는 문자열 숫자를 ['1','10','2'] 로 늘어놓아 한 성적서에 순서가 두 벌이었다(점검 BE-12). 예측에만 있는
    # 클래스(유령)도 줄·열로 남는다(줄은 모두 0) — 정답 행의 오분류가 어디로 갔는지 보여 준다. 클래스 수를 셀 때는
    # 정답 클래스(evaluation_classes)를 쓸 것 — confusion_counts 는 유령 줄을 뺀다.
    # 고유값만 이름으로 바꾼다 — 행마다 normalize_label 을 부르면 80만 행에서 160만 번이다.
    t, p = np.asarray(y_true), np.asarray(y_pred)
    if t.dtype != p.dtype:
        t, p = t.astype(object), p.astype(object)
    present = pd.unique(np.concatenate((t, p)))
    labels = sort_labels(present)
    cm = confusion_matrix(y_true, y_pred, labels=class_values(present, labels))

    return {
        "type": "multiclass_or_binary",
        "labels": list(labels),
        "matrix": cm.tolist()
    }

def calculate_class_metrics(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, Any]:
    """M22: Class별 Metric"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    target_names = mapping_dict.get('_mlb_classes')
    # output_dict=True를 통해 JSON 형태로 바로 내려주기 좋음
    if target_names:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0, target_names=[str(x) for x in target_names])
    else:
        # 클래스 집합을 y_true 로 고정한다(C-03) — 정답에 없는 클래스가 support 0 인
        # 행을 만들면 독자는 그것을 평가 대상 클래스로 읽는다.
        kwargs = _label_kwargs(df, mapping_dict, y_true)
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0,
            target_names=[normalize_label(l) for l in kwargs["labels"]] if kwargs else None,
            **kwargs,
        )
    return report

def calculate_imbalance_ratio(df: pd.DataFrame, mapping_dict: dict) -> float:
    """M23: Imbalance Ratio (가장 많은 클래스 / 가장 적은 클래스 비율)"""
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
        raise ValueError("정답 컬럼에 값이 없어 불균형비를 계산할 수 없습니다.")

    if len(counts) == 1:
        # 클래스가 1종이면 '가장 많은 클래스 / 가장 적은 클래스' 가 자기 자신이라 항상
        # 1.0 이 된다. 그 값을 그대로 내보내면 성적서가 '완벽한 균형 상태'라고 서술한다 —
        # 실제로는 평가가 성립하지 않는 데이터인데 정반대 결론이 인쇄됐다(ISSUES.md C-04).
        raise ValueError(
            f"정답 클래스가 '{list(counts.index)[0]}' 하나뿐이라 불균형비를 정의할 수 "
            "없습니다(측정 불가). 두 개 이상의 클래스가 있는 데이터가 필요합니다."
        )

    majority = counts.max()
    minority = counts.min()

    if minority == 0:
        return float('inf')

    return float(majority / minority)

