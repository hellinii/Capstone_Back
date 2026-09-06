"""
prompt_builder.py — LLM 프롬프트 생성 (간소화 버전)
속도 우선: 컬럼명과 샘플 값만으로 빠른 매핑에 집중합니다.
"""

import pandas as pd
from app.core.schemas import TaskType, VALID_ROLES_BY_TASK


# task_type별 역할 설명 (LLM에게 필요한 최소 정보만)
_ROLE_HINTS: dict[TaskType, str] = {
    TaskType.binary: (
        "sample_id: 샘플 식별자 (id, index 등)\n"
        "y_true: 실제 정답 레이블 (0/1)\n"
        "y_pred: 예측 레이블 (0/1)\n"
        "score_positive: 양성 클래스 확률 (0~1 실수, 단일 컬럼)\n"
        "latency: 추론 지연시간 (ms)\n"
        "ignore: 평가와 무관한 컬럼"
    ),
    # multiclass/multilabel 은 확률 컬럼을 평가에 쓰지 않으므로 ignore 로 보내도록 안내한다.
    TaskType.multiclass: (
        "sample_id: 샘플 식별자\n"
        "y_true: 실제 정답 클래스명 (예: cat, dog)\n"
        "y_pred: 예측 클래스명\n"
        "latency: 추론 지연시간 (ms)\n"
        "ignore: 평가와 무관한 컬럼 (클래스별 확률 컬럼도 여기에 해당)"
    ),
    TaskType.multilabel: (
        "sample_id: 샘플 식별자\n"
        "true_labels: 실제 정답 레이블 집합 (| 구분자, 예: sports|news)\n"
        "pred_labels: 예측 레이블 집합 (| 구분자)\n"
        "latency: 추론 지연시간 (ms)\n"
        "ignore: 평가와 무관한 컬럼 (레이블별 확률 컬럼도 여기에 해당)"
    ),
}


def build_system_prompt(task_type: TaskType) -> str:
    valid_roles = [r.value for r in VALID_ROLES_BY_TASK[task_type]]
    role_hints = _ROLE_HINTS[task_type]

    return (
        f"You are an expert in ISO/IEC TS 4213:2022 classification model evaluation.\n"
        f"Task type: {task_type.value}\n\n"
        f"Available roles:\n{role_hints}\n\n"
        f"Map every column to exactly one of: {valid_roles}\n"
        f"Respond with JSON only. No explanation."
    )


# 프롬프트 증폭 상한(G-03). 실측: 컬럼 2,000개 x 셀 200자 → 2,102,678자(~525k 토큰).
MAX_PROMPT_COLUMNS = 200   # 프롬프트에 실을 컬럼 수
MAX_CELL_CHARS = 100       # 셀 값 하나의 길이
MAX_PROMPT_CHARS = 200_000  # 조립 후 최종 백스톱


def _clip(value):
    """긴 셀 값만 잘라낸다. 정상 값은 표현을 그대로 보존해야 매핑 품질이 떨어지지 않는다
    (숫자를 문자열로 바꾸면 LLM 이 보는 것이 달라진다)."""
    if isinstance(value, str):
        return value[:MAX_CELL_CHARS] + "…" if len(value) > MAX_CELL_CHARS else value
    text = repr(value)
    return text[:MAX_CELL_CHARS] + "…" if len(text) > MAX_CELL_CHARS else value


def build_user_prompt(columns: list[str], sample_df: pd.DataFrame) -> str:
    # 컬럼명 + unique 샘플 값 (최대 5개) — 30행 샘플 기준으로 클래스 파악에 충분.
    # 행 수는 df.head(30) 으로 제한돼 있었으나 컬럼 수와 셀 길이에는 상한이 없어
    # 무인증 요청 하나가 프롬프트를 2백만 자까지 부풀릴 수 있었다(G-03).
    shown = columns[:MAX_PROMPT_COLUMNS]
    lines = []
    for col in shown:
        vals = [_clip(v) for v in list(dict.fromkeys(sample_df[col].dropna().tolist()))[:5]]
        lines.append(f"{col}: {vals}")

    prompt = "Columns:\n" + "\n".join(lines)
    omitted = len(columns) - len(shown)
    if omitted > 0:
        prompt += f"\n... ({omitted} more columns omitted)"
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS] + "\n... (truncated)"
    return prompt
