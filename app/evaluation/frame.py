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


def dropna_columns(df: pd.DataFrame, mapping_dict: dict) -> List[str]:
    """결측 시 행을 버리는 대상 컬럼."""
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
        selected_metric_ids: 선택된 지표. **현재는 사용하지 않는다** —
            결측 제거 대상을 '선택 지표가 실제로 읽는 컬럼'으로 좁히는 것은 D-06 이고,
            그것은 골든 값을 또 다른 이유로 바꾼다. 한 변경에 골든이 서로 무관한 두
            이유로 흔들리지 않도록 분리했다. 나중에 붙일 때 **호출부와 시그니처를 다시
            고치지 않도록** 지금부터 받아 둔다.

    Returns:
        (df_clean, dropped_rows, notes)

    Raises:
        ValueError: 매핑된 필수 컬럼이 데이터에 없을 때.
    """
    del selected_metric_ids  # D-06 예약 — 위 docstring 참조

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

    before = len(work)
    work = work.dropna(subset=dropna_columns(work, mapping_dict))
    dropped = before - len(work)

    notes: List[str] = []
    if dropped > 0:
        notes.append(
            f"{dropped}개 행이 결측치(NaN)로 인해 제외되었습니다 "
            f"(전체 {before}개 중 {len(work)}개 평가)."
        )

    return work, dropped, notes
