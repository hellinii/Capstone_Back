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
from .display import judged_text, plain_text


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


# ── 늘 계산하는 혼동행렬·클래스별 표(S2) ──
# 서술 7.1·8.3 과 9절 최약 클래스는 M21·M22 를 고르지 않아도 행렬·표를 쓴다(사용자 결정 2026-10-19). 성적서
# 6절의 M21·M22 표 인쇄는 지금처럼 선택을 따른다 — 이 두 키는 서술·사실 시트·문제 목록의 재료다.

def calculate_confusion_matrix_always(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, Any]:
    """success_metrics.confusion_matrix — M21 과 같은 모양 + n_samples.

    멀티레이블은 레이블마다 2×2 라 각 2×2 의 합이 표본 수이고, 모든 칸의 합은 '레이블 칸' 수다. 그 단위를
    unit = "label_cell"·cell_total 로 밝힌다(200행·레이블 4개면 800칸 — 800건으로 읽으면 표본 수와 어긋난다).
    """
    cm = common.calculate_confusion_matrix(df, mapping_dict)
    out = {**cm, "n_samples": int(len(df))}
    if cm["type"] == "multilabel":
        out["unit"] = "label_cell"
        out["cell_total"] = int(sum(sum(sum(row) for row in block) for block in cm["matrix"]))
    return out


def calculate_class_report_always(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, Any]:
    """success_metrics.class_report — M22 와 같은 모양(sklearn classification_report).

    이진·다중은 'accuracy' 를 늘 싣고 'micro avg' 는 싣지 않는다. sklearn 은 예측에만 있는 클래스가 있으면
    accuracy 대신 정답 클래스로 좁힌 'micro avg' 를 내는데, 다중 클래스의 마이크로 평균은 Accuracy 다
    (프론트 classReportTable 이 이 값으로 마이크로 평균 줄을 만든다). 멀티레이블은 sklearn 그대로
    (매크로·가중·마이크로·샘플 평균).
    """
    report = common.calculate_class_metrics(df, mapping_dict)
    if mapping_dict.get("_task_type") in ("binary", "multiclass"):
        y_true, y_pred = common._get_true_pred(df, mapping_dict)
        report.pop("micro avg", None)
        report["accuracy"] = float(accuracy_score(y_true, y_pred))
    return report


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

    차이를 만든 네 값도 싣는다(3절 불균형 진단 문장, 사용자 결정 6): accuracy·macro_f1(판정에 쓴 것과 같은 정의 —
    정확도와 클래스별 F1 의 단순 평균; 멀티레이블은 판정에 쓰지 않아 둘 다 null)·majority_f1·minority_f1. 판정과 무관한
    값이라 display 의 같은 이름 칸은 엔진의 평소 표기(plain_text — 넷째 자리)다.

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
        accuracy = macro_f1 = accuracy_gap = None
    else:
        kwargs = common._label_kwargs(df, mapping_dict, y_true)
        per_class = f1_score(y_true, y_pred, average=None, zero_division=0, **kwargs)
        f1_by = dict(zip((normalize_label(l) for l in kwargs["labels"]), (float(v) for v in per_class)))
        accuracy, macro_f1 = float(accuracy_score(y_true, y_pred)), float(np.mean(per_class))
        accuracy_gap = accuracy - macro_f1
    class_gap = f1_by[majority] - f1_by[minority]

    if _clean(ratio) < IMBALANCE_MILD_BELOW:
        level = "mild"
    elif (accuracy_gap is not None and _clean(accuracy_gap) >= IMBALANCE_F1_GAP) \
            or _clean(class_gap) >= IMBALANCE_F1_GAP:
        level = "affected"
    else:
        level = "no_distortion"

    over_gap = lambda gap: _clean(gap) >= IMBALANCE_F1_GAP  # noqa: E731 — 판정과 같은 식
    return {
        "level": level,
        # 판정과 같은 쪽으로 읽히게 자릿수를 늘린 표시 문자열(S2) — 1.49996 은 넷째 자리로 1.5 지만 '경미'다.
        # 성적서·사실 시트·7.2 문장이 숫자를 찍을 때 이 값을 쓴다. 아래 숫자 칸은 종전과 같다.
        "display": {
            "imbalance_ratio": judged_text(ratio, IMBALANCE_MILD_BELOW, _clean(ratio) >= IMBALANCE_MILD_BELOW),
            "accuracy_minus_macro_f1": None if accuracy_gap is None
            else judged_text(accuracy_gap, IMBALANCE_F1_GAP, over_gap(accuracy_gap)),
            "majority_minus_minority_f1": judged_text(class_gap, IMBALANCE_F1_GAP, over_gap(class_gap)),
            "accuracy": None if accuracy is None else plain_text(accuracy),
            "macro_f1": None if macro_f1 is None else plain_text(macro_f1),
            "majority_f1": plain_text(f1_by[majority]),
            "minority_f1": plain_text(f1_by[minority]),
        },
        "imbalance_ratio": round(ratio, 4),
        "accuracy_minus_macro_f1": None if accuracy_gap is None else round(accuracy_gap, 4),
        "majority_minus_minority_f1": round(class_gap, 4),
        "accuracy": None if accuracy is None else round(accuracy, 4),
        "macro_f1": None if macro_f1 is None else round(macro_f1, 4),
        "majority_f1": round(f1_by[majority], 4),
        "minority_f1": round(f1_by[minority], 4),
        "majority_class": majority,
        "minority_class": minority,
    }
