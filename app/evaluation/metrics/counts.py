"""app/evaluation/metrics/counts.py — TP·TN·FP·FN 건수(지표 선택과 무관하게 늘 계산)

재료 목록 9절 "TP, TN, FP, FN 건수 — 항상, 7.1, AI 예"와 재료 선택 점검 4.2 #6("대표 작동 지점의
TP·FP 건수 — 시스템이 계산, 늘 채워짐")을 채운다. 종전에는 혼동행렬(M21)을 골랐을 때만
행렬로 나왔다.

- 이진: 양성 클래스 기준의 tp·tn·fp·fn 하나. 양성 기준은 M7·M8 과 같다(`_binary_confusion_counts`).
- 다중: 클래스마다 일대다(one-vs-rest) 건수.
- 멀티레이블: 레이블마다 2x2 건수.

같은 건수에서 나오는 불균형 영향 판정(서술 설계 D20, `calculate_imbalance_effect`)도 여기에 둔다.

상호작용
- 의존(import): numpy, sklearn, ..labels, ..preprocessor(class_distribution_of), .binary, .common
- 사용처: app.evaluation.engine.evaluate (success_metrics 의 confusion_counts·imbalance_effect 키)
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from app.evaluation.labels import normalize_label
from app.evaluation.preprocessor import class_distribution_of

from . import binary, common


def calculate_confusion_counts(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, Any]:
    if mapping_dict.get("_task_type") == "binary":
        tn, fp, fn, tp = binary._binary_confusion_counts(df, mapping_dict)
        y_true, _ = binary._get_true_pred(df, mapping_dict)
        positive = binary._resolve_positive_value(y_true, mapping_dict.get("_positive_class"))
        return {
            "type": "binary",
            "positive_class": normalize_label(positive),  # 표기는 metadata 와 같게('1', '1.0' 아님)
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    cm = common.calculate_confusion_matrix(df, mapping_dict)
    rows = []
    if cm["type"] == "multilabel":
        for label, ((tn, fp), (fn, tp)) in zip(cm["labels"], cm["matrix"]):
            rows.append({"label": label, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})
    else:
        matrix = np.array(cm["matrix"])
        total = int(matrix.sum())
        # 줄은 정답 클래스만 둔다. 예측에만 있는 클래스(유령)는 평가 클래스가 아니다 — 그 예측은 정답 클래스 줄의
        # FN 으로 이미 세어진다. 줄을 두면 tp 0·fn 0·fp k 인 '클래스'가 사실 시트에 실렸다.
        classes = set(common.evaluation_classes(df, mapping_dict))
        for i, label in enumerate(cm["labels"]):
            if label not in classes:
                continue
            tp = int(matrix[i, i])
            fp = int(matrix[:, i].sum()) - tp
            fn = int(matrix[i, :].sum()) - tp
            rows.append({"label": label, "tp": tp, "fp": fp, "fn": fn, "tn": total - tp - fp - fn})
    return {"type": "per_class", "rows": rows}


# ── 불균형 영향(서술 설계 D20) ──
# IR 이 이 값 **미만**이면 경미. 1.5 정확히는 경미가 아니다(설계 D20 "1.5 미만 = 경미").
IMBALANCE_MILD_BELOW = 1.5
# 정확도 − 매크로 F1, 또는 다수 F1 − 소수 F1 이 이 값 **이상**이면 "영향 있음".
IMBALANCE_F1_GAP = 0.10
# 부동소수 표현 오차를 걷어 내는 자릿수 — `0.9 − 0.8` 은 0.0999…98 이라 그대로 비교하면
# 사용자가 읽는 "정확히 0.10 차이"가 영향 없음이 된다(misclassification._clean 과 같은 이유).
_CLEAN_DIGITS = 9


def _clean(x: float) -> float:
    return round(float(x), _CLEAN_DIGITS)


def _majority_minority(distribution: Dict[str, int]):
    """다수·소수 클래스를 결정적으로 고른다 — 개수가 같으면 `sort_labels` 순서의 앞쪽.

    분포가 모두 같으면(IR = 1) 소수는 다수를 뺀 나머지에서 앞쪽을 고른다(같은 클래스를 두 번 고르지 않게).
    `class_distribution_of` 가 이미 `sort_labels` 순서로 돌려준다.
    """
    labels = list(distribution)
    majority = max(labels, key=lambda l: distribution[l])  # max/min 은 동률이면 앞의 것
    minority = min((l for l in labels if l != majority), key=lambda l: distribution[l])
    return majority, minority


def calculate_imbalance_effect(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, Any]:
    """불균형이 성능 수치를 왜곡했는지(서술 설계 D20) — success_metrics 의 imbalance_effect 키.

    - IR(다수/소수, M23 과 같은 정의 — M23 을 고르지 않아도 계산) < 1.5 → "mild"(경미).
    - IR ≥ 1.5 이고 (정확도 − 매크로 F1 ≥ 0.10 또는 다수 F1 − 소수 F1 ≥ 0.10) → "affected".
    - 그 외 → "no_distortion"(불균형은 있으나 성능 왜곡은 관찰되지 않음).

    F1 은 M22(classification_report)와 같은 계산이다 — 클래스 집합은 y_true 로 고정, zero_division=0.
    다수 F1 − 소수 F1 은 부호 그대로 싣는다(소수가 더 잘 맞으면 음수, 그때 둘째 조건은 성립하지 않는다).

    멀티레이블(자체 설정): 레이블 등장 빈도로 IR 을 내고, 다수·소수 레이블의 F1 차만 본다.
    한 표본에 레이블이 여럿이라 "정확도 − 매크로 F1" 은 뜻이 서지 않아 null 로 둔다.

    정답이나 예측이 없거나 정답 클래스가 하나뿐이면 예외 — 엔진이 키를 두지 않는다.
    """
    task_type = mapping_dict.get("_task_type")
    # 분포의 단일 출처 — 3절 IR·검증 화면 IR 과 같은 값(키는 표준 표현형, sort_labels 순서).
    distribution = class_distribution_of(df, mapping_dict, task_type)
    if len(distribution) < 2:
        raise ValueError("정답 클래스가 둘 이상이어야 불균형 영향을 판단할 수 있습니다.")
    majority, minority = _majority_minority(distribution)
    ratio = distribution[majority] / distribution[minority]

    y_true, y_pred = common._get_true_pred(df, mapping_dict)
    if task_type == "multilabel":
        labels = [normalize_label(l) for l in mapping_dict.get("_mlb_classes", [])]
        per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
        f1_by = dict(zip(labels, (float(v) for v in per_label)))
        accuracy_gap = None
    else:
        kwargs = common._label_kwargs(df, mapping_dict, y_true)
        per_class = f1_score(y_true, y_pred, average=None, zero_division=0, **kwargs)
        f1_by = dict(zip((normalize_label(l) for l in kwargs["labels"]), (float(v) for v in per_class)))
        accuracy_gap = float(accuracy_score(y_true, y_pred)) - float(np.mean(per_class))
    class_gap = f1_by[majority] - f1_by[minority]

    if _clean(ratio) < IMBALANCE_MILD_BELOW:
        level = "mild"
    elif (accuracy_gap is not None and _clean(accuracy_gap) >= IMBALANCE_F1_GAP) \
            or _clean(class_gap) >= IMBALANCE_F1_GAP:
        level = "affected"
    else:
        level = "no_distortion"

    return {
        "level": level,
        "imbalance_ratio": round(ratio, 4),
        "accuracy_minus_macro_f1": None if accuracy_gap is None else round(accuracy_gap, 4),
        "majority_minus_minority_f1": round(class_gap, 4),
        "majority_class": majority,
        "minority_class": minority,
    }
