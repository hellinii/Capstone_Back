"""app/evaluation/preprocessor.py — 지표 계산 전 데이터 정리/검증 전처리

preprocess_data 는 단계별 헬퍼(_guard/_coerce/_validate/_check/_extract)를 순서대로
호출하는 얇은 파이프라인이다. 각 헬퍼는 한 가지 정리·검증 책임만 가진다.

"어떤 행이 평가에 들어가는가"(필수 컬럼 확인 · 멀티레이블 빈 셀 보존 · 결측 행 제거)는
`frame.build_evaluation_frame` 이 정본이며, **검증(validation_service)도 같은 함수를
호출한다** — 종전에는 두 계층이 서로 다른 결측 기준을 써서 성적서에 인쇄되는 표본 수가
지표를 만든 표본 수와 달랐다(ISSUES.md D-01).

상호작용
- 의존(import): pandas, .frame
- 사용처: app.evaluation.engine.evaluate (지표 계산 직전 호출)
"""
import ast
from typing import Any, Dict, List, Tuple

import pandas as pd

from app.core.schemas import METRIC_REQUIREMENTS, PREDICTION_ROLES_BY_TASK, TaskType
from app.evaluation.labels import (
    default_positive_class, find_label, normalize_distribution, normalize_label, parse_label_cell, sort_labels,
)

from .frame import build_evaluation_frame


def _guard_identical_true_pred(mapping_dict: dict) -> None:
    """0. 정답=예측 동일 컬럼 방어선 (가짜 100% 차단). 라우터/검증 우회 호출의 최종 백스톱."""
    for true_role, pred_role in [
        ('y_true', 'y_pred'), ('true_class', 'predicted_class'), ('true_labels', 'pred_labels'),
    ]:
        t_col, p_col = mapping_dict.get(true_role), mapping_dict.get(pred_role)
        if t_col and p_col and t_col == p_col:
            raise ValueError(
                f"정답('{true_role}')과 예측('{pred_role}')에 동일한 컬럼 '{t_col}'이 매핑되어 "
                "모든 지표가 100%로 산출됩니다(평가 무의미). 서로 다른 컬럼을 지정해주세요."
            )


def _coerce_label_types(df: pd.DataFrame, mapping_dict: dict) -> pd.DataFrame:
    """3. y_true 기준으로 y_pred dtype 강제 캐스팅."""
    y_true_col = mapping_dict.get('y_true') or mapping_dict.get('true_class')
    y_pred_col = mapping_dict.get('y_pred') or mapping_dict.get('predicted_class')
    if y_true_col and y_pred_col:
        true_type = df[y_true_col].dtype
        pred_series = df[y_pred_col]

        # 정수형으로 캐스팅할 때 소수값은 astype 이 예외 없이 잘라낸다(0.6 → 0).
        # 확률 컬럼을 y_pred 로 잘못 매핑한 경우가 대표적이며, 그대로 두면 모든 예측이
        # 0 으로 뭉개진 채 겉보기 정상인 성적서가 발급된다. 조용한 절단 대신 차단한다.
        if pd.api.types.is_integer_dtype(true_type) and pd.api.types.is_float_dtype(pred_series):
            fractional = pred_series.dropna() % 1 != 0
            if fractional.any():
                sample = pred_series[fractional].iloc[0]
                raise ValueError(
                    f"예측 라벨 '{y_pred_col}'에 소수값({sample})이 있어 정답 라벨 '{y_true_col}'의 "
                    f"정수 타입으로 변환하면 값이 잘립니다. 확률 컬럼을 예측(y_pred) 컬럼으로 "
                    f"매핑하지 않았는지 확인해주세요."
                )

        try:
            df[y_pred_col] = pred_series.astype(true_type)
        except Exception:
            raise ValueError(f"예측 라벨 '{y_pred_col}'을 정답 라벨 '{y_true_col}'의 타입({true_type})으로 강제 변환할 수 없습니다.")
    return df


# 파서는 labels.parse_label_cell 하나뿐이다(ISSUES.md D-04). 종전에는 라벨을 만드는
# 코드가 넷이었고 구분자 처리와 산출 타입이 서로 달랐다.
_parse_multilabel_value = parse_label_cell


def _parse_multilabel_columns(df: pd.DataFrame, mapping_dict: dict) -> pd.DataFrame:
    """3. 멀티레이블 true/pred 컬럼을 리스트로 파싱."""
    for role in ['true_labels', 'pred_labels']:
        col = mapping_dict.get(role)
        if col and col in df.columns:
            df[col] = df[col].apply(_parse_multilabel_value)
    return df


def score_columns(role_columns: Dict[str, List[str]]) -> List[str]:
    """확률·점수 역할에 매핑된 **모든** 컬럼(매핑 순서 보존).

    종전에는 `{role: column}` 축약에서 뽑아 역할당 마지막 컬럼 하나만 남았다. 그래서
    score_per_label 을 4개 매핑하면 앞 3개는 범위 검사를 받지 못했고, 매핑 순서만 바꿔도
    같은 파일이 200/400 으로 갈렸다(ISSUES.md D-08).
    """
    cols: List[str] = []
    for role in ('score_positive', 'prob_per_class', 'score_per_label'):
        for col in role_columns.get(role, []):
            if col not in cols:
                cols.append(col)
    return cols


def _validate_score_columns(df: pd.DataFrame, role_columns: Dict[str, List[str]]) -> pd.DataFrame:
    """4. 확률/점수 컬럼 float 강제 변환 + [0,1] 범위 유효성 검사.

    역할당 1컬럼이 아니라 매핑된 전 컬럼을 검사한다(D-08).
    """
    for col in score_columns(role_columns):
        if col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except Exception:
                raise ValueError(f"확률 컬럼 '{col}'에 숫자로 변환할 수 없는 문자가 포함되어 있습니다.")

            invalid = df[(df[col] < 0.0) | (df[col] > 1.0)]
            if not invalid.empty:
                # 첫 번째 에러가 발생한 위치를 구체적으로 안내.
                # 인덱스는 0-based 라 그대로 인쇄하면 사용자가 세는 행 번호와 한 칸 어긋난다
                # (ISSUES.md D-11). 헤더를 뺀 데이터 행 번호(1-based)로 환산해 말한다.
                first_idx = invalid.index[0]
                first_val = invalid.loc[first_idx, col]
                raise ValueError(
                    f"'{col}' 컬럼의 {int(first_idx) + 1}번째 행: 값 {first_val} "
                    "(허용범위 0.0~1.0 초과). Logit 등이 아닌 0~1 사이의 확률값으로 "
                    "변환 후 다시 업로드해 주세요."
                )
    return df


# 확률 컬럼 이름의 접두어 — '접두어 + 클래스·레이블 이름'(prob_cat·score_sports). 이름에서 클래스를 확정할 때
# 쓰는 단일 출처다. 프론트 업로드·매핑 화면의 이름 안내가 이 튜플을 대조한다(프론트 테스트가 이 줄을 읽는다).
PROB_COLUMN_PREFIXES = ("prob_", "probability_", "score_", "p_")

# 파생 예측이 들어갈 컬럼명. 원본 컬럼과 충돌하지 않도록 접두어를 붙인다.
DERIVED_PREDICTION_COLUMN = "__derived_prediction__"


def _label_from_column(col: str) -> str:
    """확률 컬럼명에서 클래스·레이블 이름을 뽑는다('prob_cat' → 'cat')."""
    for prefix in PROB_COLUMN_PREFIXES:
        if col.startswith(prefix) and len(col) > len(prefix):
            return col[len(prefix):]
    return col


def _resolve_probability_labels(cols: List[str], known: List[str], kind: str) -> List[str]:
    """확률 컬럼 ↔ 클래스/레이블 대응을 **컬럼명으로만** 확정한다.

    컬럼 순서로 클래스를 정하는 것은 검증할 수 없는 가정이라(정렬 순서가 컬럼 순서와
    같다는 보장이 없다) 쓰지 않는다. 이름으로 확정되지 않으면 조용히 추측하는 대신
    막는다 — 잘못 짝지으면 전 행의 예측이 뒤바뀐 채 겉보기 정상인 성적서가 나온다.
    (validation_checks 의 argmax 검사가 D-07 에서 택한 것과 같은 원칙.)
    """
    labels = [_label_from_column(c) for c in cols]
    known_set = {normalize_label(k) for k in known}
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"확률 컬럼명에서 뽑은 {kind} 이름이 중복됩니다({labels}). "
            f"컬럼명을 '접두어+{kind}명' 형태로 서로 다르게 지어 주세요."
        )
    unresolved = [c for c, l in zip(cols, labels) if l not in known_set]
    if unresolved:
        raise ValueError(
            f"확률 컬럼 {unresolved} 의 이름에서 {kind}를 확정할 수 없습니다"
            f"(데이터의 {kind}: {sorted(known_set)}). "
            f"컬럼명을 'prob_<{kind}명>' 처럼 짓거나 예측 컬럼을 직접 매핑해 주세요."
        )
    return labels


def _threshold_for(decision_threshold, col: str) -> float:
    """컬럼별 임계값 — dict 면 컬럼명으로 조회, 스칼라면 공통, 없으면 SPEC §6 기본값 0.5."""
    if isinstance(decision_threshold, dict):
        return float(decision_threshold.get(col, 0.5))
    if decision_threshold is None:
        return 0.5
    return float(decision_threshold)


def prediction_is_needed(task_type: str, selected_metric_ids: List[str] | None) -> bool:
    """선택한 지표 중 예측 역할을 쓰는 것이 하나라도 있는가.

    M23(불균형비)처럼 정답 분포만으로 계산되는 지표만 골랐다면 예측이 필요 없다 —
    그때는 확률이 매핑돼 있어도 파생하지 않는다. 파생은 공짜가 아니다: 확률 컬럼명에서
    클래스를 확정할 수 없으면 400 으로 막으므로, 쓰지도 않을 예측을 만들려다 정답
    컬럼만으로 진행 가능한 요청을 거절하게 된다.

    지표 ID 를 하드코딩하지 않고 METRIC_REQUIREMENTS 에서 유도한다.
    """
    if not selected_metric_ids:
        return True  # 지표를 모르면 종전대로 보수적으로 파생한다.
    try:
        task = TaskType(task_type)
    except ValueError:
        return True
    primary, _ = PREDICTION_ROLES_BY_TASK[task]
    requirements = METRIC_REQUIREMENTS[task]
    return any(
        primary in requirements[m] for m in selected_metric_ids if m in requirements
    )


def derivation_source(
    role_columns: Dict[str, List[str]], task_type: str, selected_metric_ids: List[str] | None,
) -> List[str]:
    """확률에서 예측을 파생할 때 쓸 확률 컬럼 — 파생하지 않으면 빈 목록.

    파생하는 조건은 이 함수 하나에서 정한다. 평가(_derive_predictions)와 검증 단계의 이름 검사
    (probability_names_problem)가 같이 쓴다 — 둘이 다르게 판정하면 검증을 통과한 데이터가 평가에서
    400 으로 막힌다(M3).
    - 하드 예측이 매핑돼 있으면 파생하지 않는다(SPEC §1 규칙 1 — 둘 다 있으면 y_pred 우선).
    - 고른 지표가 예측을 쓰지 않으면 파생하지 않는다(prediction_is_needed).
    - 파생할 확률이 없으면 예측 없이 계산 가능한 지표(M23 등)만 남는다.
    """
    primary = 'pred_labels' if task_type == 'multilabel' else 'y_pred'
    if role_columns.get(primary):
        return []
    if not prediction_is_needed(task_type, selected_metric_ids):
        return []
    return list(role_columns.get(
        {'binary': 'score_positive', 'multiclass': 'prob_per_class'}.get(task_type, 'score_per_label'),
        [],
    ))


def _known_targets(task_type: str, df: pd.DataFrame, truth_col: str, metadata: Dict[str, Any]) -> List[str]:
    """확률 컬럼 이름과 맞춰 볼 클래스(다중)·레이블(멀티) 이름 — 메타데이터가 있으면 그것, 없으면 정답 컬럼에서."""
    if task_type == 'multilabel':
        known = metadata.get('detected_labels') or sorted(
            {l for cell in df[truth_col] for l in parse_label_cell(cell)}
        )
    else:
        known = metadata.get('detected_classes') or list(pd.Series(df[truth_col]).dropna().unique())
    # 표기는 normalize_label — 빈 칸 섞인 정수 정답(1.0)·앞뒤 공백(' cat')도 prob_1·prob_cat 과 맞는다.
    return [normalize_label(k) for k in known]


def probability_names_problem(
    df: pd.DataFrame,
    role_columns: Dict[str, List[str]],
    task_type: str,
    metadata: Dict[str, Any] | None,
    selected_metric_ids: List[str] | None,
) -> str | None:
    """평가가 확률에서 예측을 파생할 때 컬럼 이름으로 클래스·레이블을 확정할 수 있는가(M3).

    확정할 수 없으면 평가가 400 으로 낼 문구를, 되거나 파생하지 않으면 None 을 돌려준다. 검증 단계가
    평가 전에 같은 판정을 미리 본다 — 이름 규칙은 '접두어(PROB_COLUMN_PREFIXES) + 클래스·레이블 이름'.
    이진은 확률 컬럼이 하나라 이름을 보지 않는다.
    """
    if task_type not in ('multiclass', 'multilabel'):
        return None
    source = derivation_source(role_columns, task_type, selected_metric_ids)
    if not source:
        return None
    truth_col = role_columns.get('y_true', role_columns.get('true_labels', [None]))[0]
    if truth_col is None or truth_col not in df.columns:
        return None
    kind = "클래스" if task_type == 'multiclass' else "레이블"
    try:
        _resolve_probability_labels(source, _known_targets(task_type, df, truth_col, metadata or {}), kind)
    except ValueError as e:
        return str(e)
    return None


def _derive_predictions(
    df: pd.DataFrame,
    role_columns: Dict[str, List[str]],
    task_type: str,
    decision_threshold,
    metadata: Dict[str, Any],
    logs: Dict[str, Any],
    selected_metric_ids: List[str] | None = None,
) -> pd.DataFrame:
    """4-2. 하드 예측이 없을 때 확률에서 예측을 파생한다(ISSUES.md A-01·A-02, 결정 1).

    파생은 **모델의 실제 출력이 아니다.** SPEC §0 이 요구하는 대로 파생 사실·임계값·출처
    컬럼을 `logs["derived_prediction"]` 에 남겨 응답과 성적서가 인쇄하게 한다.

    이미 하드 예측이 매핑돼 있으면 아무것도 하지 않는다(SPEC §1 규칙 1 — 둘 다 있으면
    y_pred 우선). 그래서 기존 요청의 결과는 한 값도 바뀌지 않는다.
    """
    source = derivation_source(role_columns, task_type, selected_metric_ids)
    if not source:
        return df

    truth_col = role_columns.get('y_true', role_columns.get('true_labels', [None]))[0]
    out = DERIVED_PREDICTION_COLUMN

    if task_type == 'binary':
        col = source[0]
        threshold = _threshold_for(decision_threshold, col)
        classes = list(pd.Series(df[truth_col]).dropna().unique())
        # metadata 의 양성·음성은 표기('1'·'cat')다. 정답 라벨의 실제 값(dtype·공백 포함)으로 되찾아 쓴다 —
        # 0/1 로 만들면 문자열 정답과 섞여 sklearn 이 'mix of binary and unknown targets' 로 죽고, str() 로
        # 맞춰 보면 빈 칸 섞인 정수 정답(1.0)에서 사용자가 고른 양성을 못 찾아 기본값으로 바꿔 버린다.
        positive = find_label(classes, metadata.get('positive_class'))
        if positive is None:
            # 기본 양성은 자동 판단·지표 계산과 같은 함수로 정한다(M5).
            positive = default_positive_class(classes)[0] if classes else 1
        negative = find_label(classes, metadata.get('negative_class'))
        if negative is None:
            remaining = [c for c in classes if normalize_label(c) != normalize_label(positive)]
            if not remaining:
                raise ValueError(
                    "정답 컬럼에 클래스가 하나뿐이라 확률에서 예측을 파생할 수 없습니다."
                )
            negative = remaining[0]
        df[out] = [positive if v >= threshold else negative for v in df[col]]
        logs["derived_prediction"] = {
            "method": "threshold", "threshold": threshold,
            "source_columns": [col], "target_role": "y_pred",
            "positive_class": normalize_label(positive), "negative_class": normalize_label(negative),
        }

    elif task_type == 'multiclass':
        if decision_threshold is not None:
            logs["warnings"].append(
                "다중 클래스는 확률이 가장 큰 클래스를 예측으로 쓰므로 결정 임계값은 쓰지 않았습니다."
            )
        labels = _resolve_probability_labels(source, _known_targets(task_type, df, truth_col, metadata), "클래스")
        classes = list(pd.Series(df[truth_col]).dropna().unique())
        by_name = {normalize_label(c): c for c in classes}  # 표기 → 원래 값(1.0·' cat')
        picks = df[source].astype(float).values.argmax(axis=1)
        df[out] = [by_name.get(labels[i], labels[i]) for i in picks]
        logs["derived_prediction"] = {
            "method": "argmax", "threshold": None,
            "source_columns": list(source), "target_role": "y_pred",
            "class_order": labels,
        }

    else:  # multilabel
        labels = _resolve_probability_labels(source, _known_targets(task_type, df, truth_col, metadata), "레이블")
        thresholds = {c: _threshold_for(decision_threshold, c) for c in source}
        values = df[source].astype(float)
        df[out] = [
            [labels[j] for j, col in enumerate(source) if row[j] >= thresholds[col]]
            for row in values.values
        ]
        logs["derived_prediction"] = {
            "method": "threshold_per_label",
            "threshold": thresholds if isinstance(decision_threshold, dict)
                         else _threshold_for(decision_threshold, source[0]),
            "source_columns": list(source), "target_role": "pred_labels",
            "label_order": labels,
        }

    logs["derived_prediction"]["column"] = out
    logs["warnings"].append(
        f"예측 컬럼이 없어 {_DERIVATION_TEXT[logs['derived_prediction']['method']]}. "
        "이렇게 만든 예측은 모델이 직접 낸 값이 아닙니다."
    )
    return df


# 평가 경고(화면 [평가] 안내·성적서)에 찍는 파생 방식 — 내부 이름(argmax 등) 대신 쉬운 말.
# 매핑 확인 단계의 안내(analysis/validator.py _WARNING_CONDITIONS)와 같은 설명이다.
_DERIVATION_TEXT = {
    "threshold": "양성 클래스 점수가 결정 임계값 이상이면 양성으로 예측했습니다",
    "argmax": "클래스별 확률이 가장 큰 클래스를 예측으로 썼습니다",
    "threshold_per_label": "레이블별 점수가 그 레이블의 결정 임계값 이상이면 그 레이블을 예측으로 썼습니다",
}


def _coerce_latency(df: pd.DataFrame, mapping_dict: dict, logs: Dict[str, Any]) -> None:
    """4-1. 지연시간 컬럼 숫자 변환(best-effort). 비숫자→NaN(통계만 제외), 음수 경고."""
    latency_col = mapping_dict.get('latency')
    if latency_col and latency_col in df.columns:
        coerced = pd.to_numeric(df[latency_col], errors="coerce")
        bad = int((coerced.isna() & df[latency_col].notna()).sum())
        if bad > 0:
            logs["warnings"].append(f"지연시간(latency) 컬럼에 숫자가 아닌 값 {bad}개가 있어 해당 행의 지연시간은 측정에서 제외됩니다.")
        df[latency_col] = coerced
        if (df[latency_col] < 0).any():
            neg = int((df[latency_col] < 0).sum())
            logs["warnings"].append(f"지연시간(latency) 컬럼에 음수 값 {neg}개가 있습니다(측정 오류 가능).")


def _check_prob_sum(df: pd.DataFrame, task_type: str, prob_cols: List[str], logs: Dict[str, Any]) -> None:
    """5. 확률합 검증 (Multiclass 한정)."""
    if task_type == 'multiclass' and len(prob_cols) > 1:
        row_sums = df[prob_cols].sum(axis=1)
        invalid_sums = row_sums[(row_sums < 0.99) | (row_sums > 1.01)]
        if not invalid_sums.empty:
            logs["warnings"].append(f"Multiclass 확률합 경고: {len(invalid_sums)}개 행에서 확률의 합이 1.0(±0.01) 범위를 벗어났습니다. 결과의 신뢰도가 낮을 수 있습니다.")


def _warn_ghost_classes(df: pd.DataFrame, mapping_dict: dict, logs: Dict[str, Any]) -> None:
    """5-1. 예측에만 등장한 클래스를 알린다 (ISSUES.md C-03).

    클래스 집합은 `y_true` 기준으로 고정되므로 이 클래스들은 어떤 지표의 분모도 되지
    않고, 그렇게 예측한 행은 정답 클래스의 오분류(FN)로 남는다. **표본 수는 변하지
    않는다.** 조용히 처리하면 사용자는 자기 예측의 일부가 어떻게 셈해졌는지 알 수 없다.
    """
    from app.evaluation.metrics.common import ghost_classes

    ghosts = ghost_classes(df, mapping_dict)
    if ghosts:
        logs["warnings"].append(
            f"정답에 없는 클래스가 예측에만 등장했습니다: {', '.join(ghosts)}. "
            "평가 클래스 집합은 정답(y_true) 기준으로 고정되며, 해당 예측은 정답 클래스의 "
            "오분류로 계산됩니다(표본 수는 변하지 않습니다)."
        )


def class_distribution_of(df: pd.DataFrame, mapping_dict: dict, task_type: str) -> Dict[str, int]:
    """평가 프레임의 정답 컬럼에서 클래스(레이블)별 등장 횟수를 센다 — 분포의 단일 출처.

    평가(`preprocess_data`)와 데이터 검증(`validation_service`)이 **같은 프레임**에 이 함수를
    불러, 검증 화면이 미리 보는 불균형 비율(IR)과 성적서 3절의 IR 이 같은 값이 된다.
    멀티레이블 셀은 파싱 전 원문이어도 되고 이미 리스트여도 된다(`parse_label_cell` 은 둘 다 받는다).
    """
    class_dist: Dict[str, int] = {}
    col = mapping_dict.get('y_true') or mapping_dict.get('true_class') or mapping_dict.get('true_labels')
    if col and col in df.columns:
        if task_type == 'multilabel':
            for cell in df[col]:
                for label in parse_label_cell(cell):
                    class_dist[label] = class_dist.get(label, 0) + 1
        else:
            class_dist = df[col].value_counts().to_dict()
    # 키 표현형과 순서를 표준화한다 — 종전 `{str(k): v}` 는 int 1 과 str '1' 을 뭉개며
    # 카운트를 **덮어썼고**, 사전순 정렬이 ['1','10','2'] 로 클래스 순서를 뒤집었다(D-04).
    return normalize_distribution(class_dist)


def _extract_class_distribution(df: pd.DataFrame, mapping_dict: dict, task_type: str, logs: Dict[str, Any]) -> None:
    """6. 클래스 분포 추출."""
    logs["class_distribution"] = class_distribution_of(df, mapping_dict, task_type)


def collect_role_columns(mappings: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """역할 → 매핑된 컬럼 **목록**(입력 순서 보존).

    `{role: column}` 축약은 확률 역할처럼 여러 컬럼을 갖는 역할에서 마지막 하나만
    남겨 조용히 정보를 잃는다(ISSUES.md D-08). 목록으로 모아 두고, 단일 컬럼 역할은
    호출부가 [0] 을 쓴다.
    """
    collected: Dict[str, List[str]] = {}
    for m in mappings:
        role = m['role']
        if role == 'ignore':
            continue
        collected.setdefault(role, []).append(m['column'])
    return collected


def preprocess_data(
    df: pd.DataFrame,
    mappings: List[Dict[str, str]],
    task_type: str,
    decision_threshold=None,
    metadata: Dict[str, Any] | None = None,
    selected_metric_ids: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """평가(Metric) 계산 전 데이터를 정리·검증하는 전처리 파이프라인(단계 헬퍼 순차 호출)."""
    if df.empty:
        raise ValueError("데이터셋이 비어 있습니다.")

    logs = {"dropped_rows": 0, "warnings": [], "errors": []}
    role_columns = collect_role_columns(mappings)
    mapping_dict = {role: cols[0] for role, cols in role_columns.items()}

    _guard_identical_true_pred(mapping_dict)

    prob_cols = role_columns.get('prob_per_class', [])

    # 필수 컬럼 확인 + 멀티레이블 빈 셀 보존 + 결측 행 제거를 공용 헬퍼에 위임한다.
    # 검증(validation_service)이 같은 함수를 호출해 **같은 프레임**을 쓴다(ISSUES.md D-01).
    df, dropped, notes = build_evaluation_frame(df, mappings, task_type, selected_metric_ids)
    if dropped > 0:
        logs["dropped_rows"] = dropped
        logs["warnings"].extend(notes)
    if len(df) == 0:
        raise ValueError("결측치를 제외하고 나니 평가할 데이터가 하나도 남지 않았습니다.")
    df = _coerce_label_types(df, mapping_dict)
    df = _parse_multilabel_columns(df, mapping_dict)
    df = _validate_score_columns(df, role_columns)
    # 확률이 float 로 강제되고 [0,1] 로 검증된 **뒤에** 파생한다 — 그 전에는 비교 연산이
    # object dtype 위에서 돌고 범위 이탈값이 그대로 예측이 된다.
    df = _derive_predictions(
        df, role_columns, task_type, decision_threshold, metadata or {}, logs,
        selected_metric_ids,
    )
    _coerce_latency(df, mapping_dict, logs)
    _check_prob_sum(df, task_type, prob_cols, logs)
    _warn_ghost_classes(df, mapping_dict, logs)
    _extract_class_distribution(df, mapping_dict, task_type, logs)

    # 실제로 지표를 계산한 행 수. 프론트가 분포 합계로 추측하던 값을 서버가 확정한다
    # (ISSUES.md B-02). 멀티레이블에서 class_distribution 은 '레이블 등장 횟수'라
    # 합계가 행 수를 넘는다 — 두 값은 같은 것이 아니다.
    logs["n_samples"] = len(df)

    return df, logs
