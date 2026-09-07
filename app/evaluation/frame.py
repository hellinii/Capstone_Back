"""app/evaluation/frame.py — 평가 대상 프레임 확정의 단일 출처.

"어떤 행이 평가에 들어가는가"를 정하는 규칙은 **한 곳에만** 있어야 한다.

종전에는 같은 데이터에 대해 결측 판정 기준이 셋이었다(ISSUES.md D-01):

  (A) 표시용   `validation_checks.check_missing_values` — latency 만 제외하고 NaN 행을 센다.
  (B) 표본수용 `validation_service` — `df_work.dropna()`. latency **포함** 전 컬럼 대상.
                이 값이 성적서 6절 "유효 예측 건수"로 인쇄된다.
  (C) 평가용   `preprocessor` — multilabel 결측은 ''로 채워 **살리고** latency 는 dropna 에서
                **제외**한다. 이 값으로 실제 지표가 계산된다.

(B)≠(C) 라서 **성적서에 인쇄되는 표본 수가 지표를 만든 표본 수와 달랐다.** 그리고
후속 검사(중복 ID·클래스 불일치·task 별 검사)가 과다 축소된 (B) 위에서 돌아 허위 경고를
만들었다(D-02).

이 모듈이 (C)의 규칙을 정본으로 삼고 검증·평가가 함께 호출한다.

상호작용
- 의존(import): pandas
- 사용처: app.evaluation.preprocessor, app.analysis.validation_service
"""
from typing import Any, Dict, List, Tuple

import pandas as pd

# 결측이어도 행을 버리지 않는 역할.
# - latency: 부가 측정이다. 응답시간을 못 잰 샘플이라고 지표 계산에서 뺄 이유가 없다.
# - true_labels/pred_labels: 빈 셀은 '해당 레이블 없음'이라는 **정상 입력**이다(''로 채운다).
_LATENCY_ROLE = "latency"
_MULTILABEL_ROLES = ("true_labels", "pred_labels")


def required_columns(mappings: List[dict]) -> List[str]:
    """존재 확인 대상 — ignore 가 아닌 모든 매핑 컬럼.

    입력 순서를 보존한다(`set()` 을 쓰면 순서가 비결정적이 되어 오류 메시지와
    프레임 컬럼 순서가 실행마다 달라진다 — ISSUES.md D-17).
    """
    seen: Dict[str, None] = {}
    for m in mappings:
        if m.get("role") != "ignore":
            seen.setdefault(m["column"], None)
    return list(seen)


def dropna_roles(
    task_type: str, selected_metric_ids: List[str] | None, mapped_roles: set
) -> set:
    """결측 시 행을 버리는 대상 **역할**.

    "선택한 지표가 실제로 읽는 컬럼"으로 좁힌다(ISSUES.md D-06). 종전에는 ignore 가
    아닌 모든 매핑 컬럼이 대상이라, 고른 지표와 무관한 컬럼(샘플 ID, 쓰지 않는 확률
    컬럼) 하나의 빈 칸이 행을 통째로 버렸다.

    좁히되 **정답·예측 역할은 언제나 대상으로 둔다.** M23 처럼 예측을 읽지 않는 지표만
    골라도 y_pred 의 NaN 이 남으면 `_coerce_label_types` 의 `astype` 이
    `IntCastingNaNError` 로 죽는다 — 그 경로를 원천 차단한다.

    확률 역할은 **파생에 쓰일 때만** 대상이다. 하드 예측이 있으면 확률은 읽히지 않으므로
    그 결측이 표본을 깎을 이유가 없다.

    `selected_metric_ids` 가 없으면 종전대로(전 컬럼) 동작한다 — 무엇을 읽는지 모르는
    호출자에게는 보수적인 쪽이 안전하다.
    """
    from app.core.schemas import (
        METRIC_REQUIREMENTS,
        PREDICTION_ROLES_BY_TASK,
        TRUTH_ROLE_BY_TASK,
        TaskType,
    )
    from app.evaluation.preprocessor import prediction_is_needed

    try:
        task = TaskType(task_type)
    except ValueError:
        return set(mapped_roles) - {_LATENCY_ROLE}

    requirements = METRIC_REQUIREMENTS[task]
    primary, alternatives = PREDICTION_ROLES_BY_TASK[task]

    roles = {TRUTH_ROLE_BY_TASK[task].value, primary.value}
    for metric_id in selected_metric_ids or []:
        roles |= {r.value for r in requirements.get(metric_id, set())}

    # 예측을 파생해야 하면 그 출처 확률 컬럼도 '읽는 컬럼'이다.
    if primary.value not in mapped_roles and prediction_is_needed(task_type, selected_metric_ids):
        roles |= {r.value for r in alternatives}

    roles.discard(_LATENCY_ROLE)   # 부가 측정 — 못 잰 샘플을 지표에서 뺄 이유가 없다
    return roles & set(mapped_roles)


def dropna_columns(df: pd.DataFrame, mapping_dict: dict) -> List[str]:
    """결측 시 행을 버리는 대상 컬럼(역할 목록을 모를 때의 보수적 기본값)."""
    latency_col = mapping_dict.get(_LATENCY_ROLE)
    return [c for c in df.columns if c != latency_col]


def build_evaluation_frame(
    df: pd.DataFrame,
    mappings: List[dict],
    task_type: str,
    selected_metric_ids: List[str] | None = None,
) -> Tuple[pd.DataFrame, int, List[str]]:
    """평가에 실제로 쓰일 프레임을 확정한다.

    Args:
        df: 원본 데이터프레임
        mappings: [{"column": ..., "role": ...}, ...]
        task_type: "binary" | "multiclass" | "multilabel"
        selected_metric_ids: 선택된 지표. 결측 제거 대상을 이 지표들이 실제로 읽는
            컬럼으로 좁힌다(ISSUES.md D-06). None/빈 목록이면 종전대로 전 컬럼을
            대상으로 삼는다 — 무엇을 읽는지 모르는 호출자에게는 보수적인 쪽이 안전하다.
            (API 경로는 빈 목록을 422 로 거절하므로 여기 도달하지 않는다.)

    Returns:
        (df_clean, dropped_rows, notes)

    Raises:
        ValueError: 매핑된 필수 컬럼이 데이터에 없을 때.
    """
    required = required_columns(mappings)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"데이터셋에 매핑된 필수 컬럼이 없습니다: {missing}")

    mapping_dict = {m["role"]: m["column"] for m in mappings if m.get("role") != "ignore"}

    work = df[required].copy()

    # 멀티레이블 빈 셀은 '해당 레이블 없음'이므로 결측이 아니다.
    for role in _MULTILABEL_ROLES:
        col = mapping_dict.get(role)
        if col and col in work.columns:
            work[col] = work[col].fillna("")

    # 결측으로 행을 버리는 대상을 선택 지표가 읽는 역할로 좁힌다(D-06).
    roles = dropna_roles(task_type, selected_metric_ids, set(mapping_dict))
    if selected_metric_ids:
        subset = [mapping_dict[r] for r in roles if mapping_dict.get(r) in work.columns]
    else:
        subset = dropna_columns(work, mapping_dict)

    before = len(work)
    work = work.dropna(subset=subset) if subset else work
    dropped = before - len(work)

    notes: List[str] = []
    if dropped > 0:
        notes.append(
            f"{dropped}개 행이 결측치(NaN)로 인해 제외되었습니다 "
            f"(전체 {before}개 중 {len(work)}개 평가)."
        )

    return work, dropped, notes
