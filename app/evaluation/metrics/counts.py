"""app/evaluation/metrics/counts.py — TP·TN·FP·FN 건수(지표 선택과 무관하게 늘 계산)

재료 목록 9절 "TP, TN, FP, FN 건수 — 항상, 7.1, AI 예"와 재료 선택 점검 4.2 #6("대표 작동 지점의
TP·FP 건수 — 시스템이 계산, 늘 채워짐")을 채운다. 종전에는 혼동행렬(M21)을 골랐을 때만
행렬로 나왔다.

- 이진: 양성 클래스 기준의 tp·tn·fp·fn 하나. 양성 기준은 M7·M8 과 같다(`_binary_confusion_counts`).
- 다중: 클래스마다 일대다(one-vs-rest) 건수.
- 멀티레이블: 레이블마다 2x2 건수.

상호작용
- 의존(import): numpy, .binary, .common
- 사용처: app.evaluation.engine.evaluate (success_metrics 의 confusion_counts 키)
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from . import binary, common


def calculate_confusion_counts(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, Any]:
    if mapping_dict.get("_task_type") == "binary":
        tn, fp, fn, tp = binary._binary_confusion_counts(df, mapping_dict)
        y_true, _ = binary._get_true_pred(df, mapping_dict)
        positive = binary._resolve_positive_value(y_true, mapping_dict.get("_positive_class"))
        return {
            "type": "binary",
            "positive_class": str(positive),
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
        for i, label in enumerate(cm["labels"]):
            tp = int(matrix[i, i])
            fp = int(matrix[:, i].sum()) - tp
            fn = int(matrix[i, :].sum()) - tp
            rows.append({"label": label, "tp": tp, "fp": fp, "fn": fn, "tn": total - tp - fp - fn})
    return {"type": "per_class", "rows": rows}
