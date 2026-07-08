"""
prompt_builder.py — LLM 프롬프트 생성 (간소화 버전)
속도 우선: 컬럼명과 샘플 값만으로 빠른 매핑에 집중합니다.
"""

import pandas as pd
from schemas import TaskType, VALID_ROLES_BY_TASK


# task_type별 역할 설명 (LLM에게 필요한 최소 정보만)
_ROLE_HINTS: dict[TaskType, str] = {
    TaskType.binary: (
        "sample_id: 샘플 식별자 (id, index 등)\n"
        "y_true: 실제 정답 레이블 (0/1)\n"
        "y_pred: 예측 레이블 (0/1)\n"
        "score_positive: 양성 클래스 확률 (0~1 실수, 단일 컬럼)\n"
        "ignore: 평가와 무관한 컬럼"
    ),
    TaskType.multiclass: (
        "sample_id: 샘플 식별자\n"
        "y_true: 실제 정답 클래스명 (예: cat, dog)\n"
        "y_pred: 예측 클래스명\n"
        "prob_per_class: 클래스별 예측 확률 (prob_cat, prob_dog 같은 여러 컬럼)\n"
        "ignore: 평가와 무관한 컬럼"
    ),
    TaskType.multilabel: (
        "sample_id: 샘플 식별자\n"
        "true_labels: 실제 정답 레이블 집합 (| 구분자, 예: sports|news)\n"
        "pred_labels: 예측 레이블 집합 (| 구분자)\n"
        "score_per_label: 레이블별 예측 확률 (score_sports 같은 여러 컬럼)\n"
        "ignore: 평가와 무관한 컬럼"
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


def build_user_prompt(columns: list[str], sample_df: pd.DataFrame) -> str:
    # 컬럼명 + unique 샘플 값 (최대 5개) — 30행 샘플 기준으로 클래스 파악에 충분
    lines = []
    for col in columns:
        vals = list(dict.fromkeys(sample_df[col].dropna().tolist()))[:5]
        lines.append(f"{col}: {vals}")

    return "Columns:\n" + "\n".join(lines)
