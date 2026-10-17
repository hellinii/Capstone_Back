"""app/evaluation/metrics/misclassification.py — 경계 오류·확신 오류 비율(서술 설계 D30)

7.1 "오분류 중 결정 경계 근처(애매한 것)가 몇 %, 높은 확신으로 틀린 것이 몇 %"를 채운다.
지표가 아니라 success_metrics 의 error_ratios 키로 내려보낸다(지표 선택과 무관, 확률이 있을 때만).

- 분모: **확률이 있는** 오분류 수. 이진·다중은 표본 단위, 멀티레이블은 "틀린 레이블 칸" 단위.
- 경계 오류: 이진·멀티레이블은 `|p − 임계값| < 0.1`(엄격 미만). 다중은 `p(예측) − p(정답) < 0.1`.
- 확신 오류: 틀린 쪽 확률 ≥ 0.9. 예측 양성이 틀렸으면 p, 예측 음성이 틀렸으면 1 − p.
  다중은 p(예측 클래스).
- 임계값은 예측을 확률에서 파생했을 때만 안다(`derived_prediction`). 예측을 직접 올렸으면 이진·
  멀티레이블의 경계는 계산하지 않고 `boundary_note = "threshold_unknown"` 으로 그 사실을 남긴다.
  다중은 정의에 임계값이 없어 늘 계산한다.

비교는 4자리 반올림 전 값으로 하되, 이진 표현 오차만 9자리에서 걷어 낸다(`_clean`).
`0.6 − 0.5` 는 부동소수로 0.0999…98 이라 그대로 비교하면 "임계값에서 정확히 0.1 떨어진"
점이 경계 오류로 셈해진다 — 사용자가 적은 소수 그대로의 뜻(정확히 0.1 = 경계 아님)을 따른다.

상호작용
- 의존(import): pandas, ..labels, ..preprocessor(_resolve_probability_labels), .binary
- 사용처: app.evaluation.engine.evaluate (success_metrics 의 error_ratios 키)
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from app.evaluation.labels import normalize_label, parse_label_cell

from . import binary

# 서술 설계 D30 의 두 기준. 응답에도 그대로 실어 서술·인쇄가 같은 값을 읽게 한다.
BOUNDARY_MARGIN = 0.1
CONFIDENT_LEVEL = 0.9

# 부동소수 표현 오차를 걷어 내는 자릿수. 입력 확률은 많아야 소수 몇 자리라 9자리면 충분하다.
_CLEAN_DIGITS = 9


def _clean(x: float) -> float:
    return round(float(x), _CLEAN_DIGITS)


def _is_boundary(distance: float) -> bool:
    """임계값(다중은 정답 클래스 확률)과의 거리가 0.1 **미만**이면 경계 오류."""
    return _clean(distance) < BOUNDARY_MARGIN


def _is_confident(wrong_side_prob: float) -> bool:
    """틀린 쪽 확률이 0.9 **이상**이면 확신 오류."""
    return _clean(wrong_side_prob) >= CONFIDENT_LEVEL


def _ratio(count: int, total: int) -> Optional[float]:
    return round(count / total, 4) if total else None


def _result(unit: str, misclassified: int, boundary: Optional[int], confident: int, threshold) -> Dict[str, Any]:
    return {
        "unit": unit,
        "misclassified": misclassified,
        "boundary": None if boundary is None else {
            "count": boundary, "ratio": _ratio(boundary, misclassified),
        },
        "boundary_note": "threshold_unknown" if boundary is None else None,
        "confident": {"count": confident, "ratio": _ratio(confident, misclassified)},
        "boundary_margin": BOUNDARY_MARGIN,
        "confident_level": CONFIDENT_LEVEL,
        "threshold": threshold,
    }


def _binary(df: pd.DataFrame, mapping_dict: dict, threshold: Optional[float]) -> Dict[str, Any]:
    # 양성 기준은 TP·FP 건수(counts)·M7·M8 과 같다 — 같은 행이 한쪽에선 FP, 다른 쪽에선 맞음이 되지 않게.
    y_true, y_pred = binary._get_true_pred(df, mapping_dict)
    score = pd.to_numeric(df[mapping_dict["score_positive"]], errors="coerce")
    positive = binary._resolve_positive_value(y_true, mapping_dict.get("_positive_class"))
    true_pos = (y_true == positive).to_numpy()
    pred_pos = (y_pred == positive).to_numpy()

    misclassified = boundary = confident = 0
    for t_pos, p_pos, p in zip(true_pos, pred_pos, score.to_numpy()):
        if t_pos == p_pos or pd.isna(p):
            continue
        misclassified += 1
        if threshold is not None and _is_boundary(abs(p - threshold)):
            boundary += 1
        if _is_confident(p if p_pos else 1 - p):
            confident += 1
    return _result("sample", misclassified, None if threshold is None else boundary, confident, threshold)


def _known_labels(cells) -> List[str]:
    return sorted({normalize_label(v) for v in cells if not pd.isna(v)})


def _multiclass(df: pd.DataFrame, mapping_dict: dict, prob_cols: List[str]) -> Dict[str, Any]:
    from app.evaluation.preprocessor import _resolve_probability_labels

    true_col, pred_col = mapping_dict["y_true"], mapping_dict["y_pred"]
    truths = [normalize_label(v) for v in df[true_col]]
    preds = [normalize_label(v) for v in df[pred_col]]
    # 확률 컬럼 ↔ 클래스는 파생 예측과 같은 규칙(이름으로만)으로 짝짓는다. 대응이 안 되면
    # ValueError 가 나고, 호출부가 키를 두지 않는다 — 순서로 추측해 엉뚱한 비율을 내지 않는다.
    labels = _resolve_probability_labels(prob_cols, _known_labels(list(df[true_col]) + list(df[pred_col])), "클래스")
    col_of = dict(zip(labels, prob_cols))
    probs = df[prob_cols].astype(float)

    misclassified = boundary = confident = 0
    for i, (t, p) in enumerate(zip(truths, preds)):
        if t == p or t not in col_of or p not in col_of:
            continue  # 맞았거나, 예측·정답 클래스 중 하나의 확률이 없다(분모 밖).
        p_pred = probs[col_of[p]].iloc[i]
        p_true = probs[col_of[t]].iloc[i]
        if pd.isna(p_pred) or pd.isna(p_true):
            continue
        misclassified += 1
        # 예측을 직접 올렸으면 p(예측) < p(정답) 일 수 있다 — 음수 차이도 "0.1 미만"이라 경계로 센다.
        if _is_boundary(p_pred - p_true):
            boundary += 1
        if _is_confident(p_pred):
            confident += 1
    return _result("sample", misclassified, boundary, confident, None)


def _multilabel(df: pd.DataFrame, mapping_dict: dict, score_cols: List[str], threshold) -> Dict[str, Any]:
    from app.evaluation.preprocessor import _resolve_probability_labels

    true_cells = [set(parse_label_cell(v)) for v in df[mapping_dict["true_labels"]]]
    pred_cells = [set(parse_label_cell(v)) for v in df[mapping_dict["pred_labels"]]]
    truth_labels = set().union(*true_cells) if true_cells else set()
    known = sorted(truth_labels.union(*pred_cells) if pred_cells else truth_labels)
    labels = _resolve_probability_labels(score_cols, known, "레이블")
    scores = df[score_cols].astype(float)

    def threshold_of(col: str) -> Optional[float]:
        if threshold is None:
            return None
        if isinstance(threshold, dict):
            return float(threshold.get(col, 0.5))
        return float(threshold)

    misclassified = boundary = confident = 0
    for label, col in zip(labels, score_cols):
        # 레이블 집합은 정답 기준으로 고정한다(common._get_true_pred 의 MLB 와 같다) — 예측에만
        # 있는 레이블의 칸은 지표 어디에도 셈해지지 않으므로 여기서도 세지 않는다.
        if label not in truth_labels:
            continue
        t_col = threshold_of(col)
        for i, p in enumerate(scores[col].to_numpy()):
            in_true, in_pred = label in true_cells[i], label in pred_cells[i]
            if in_true == in_pred or pd.isna(p):
                continue
            misclassified += 1
            if t_col is not None and _is_boundary(abs(p - t_col)):
                boundary += 1
            if _is_confident(p if in_pred else 1 - p):
                confident += 1
    return _result("label_cell", misclassified, None if threshold is None else boundary, confident, threshold)


def calculate_error_ratios(
    df: pd.DataFrame,
    mapping_dict: dict,
    role_columns: Dict[str, List[str]],
    derived: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """경계·확신 오류 비율. 확률 컬럼이나 예측이 없으면 None(호출부가 키를 두지 않는다).

    role_columns 는 역할 → 컬럼 **목록**이다(`preprocessor.collect_role_columns`). engine 의
    mapping_dict 는 역할당 컬럼 하나만 남겨 다중·멀티레이블 확률 컬럼을 잃는다.
    derived 는 전처리가 남긴 `derived_prediction`(없으면 None) — 임계값을 아는 유일한 출처다.
    """
    task_type = mapping_dict.get("_task_type")
    pred_role = "pred_labels" if task_type == "multilabel" else "y_pred"
    pred_col = mapping_dict.get(pred_role)
    if not pred_col or pred_col not in df.columns:
        return None

    if task_type == "binary":
        if not mapping_dict.get("score_positive"):
            return None
        threshold = derived.get("threshold") if derived and derived.get("method") == "threshold" else None
        return _binary(df, mapping_dict, None if threshold is None else float(threshold))

    if task_type == "multiclass":
        prob_cols = role_columns.get("prob_per_class", [])
        if not prob_cols or not mapping_dict.get("y_true"):
            return None
        return _multiclass(df, mapping_dict, prob_cols)

    if task_type == "multilabel":
        score_cols = role_columns.get("score_per_label", [])
        if not score_cols or not mapping_dict.get("true_labels"):
            return None
        threshold = (
            derived.get("threshold")
            if derived and derived.get("method") == "threshold_per_label" else None
        )
        return _multilabel(df, mapping_dict, score_cols, threshold)

    return None
