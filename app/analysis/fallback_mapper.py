"""app/analysis/fallback_mapper.py — 규칙 기반 컬럼 매핑 폴백

OpenAI 키가 없거나 LLM 호출이 실패했을 때 컬럼명 패턴으로 역할을 추정하는 폴백 매핑.

상호작용
- 의존(import): pandas, app.core.schemas, app.analysis.metadata(extract_metadata)
- 사용처: app.analysis.analysis_service (무키/LLM 실패 시 강등 경로)
"""
import pandas as pd

from app.core.schemas import ColumnMapping, ColumnRole, TaskType
from app.analysis.schemas import AnalysisResponse, ColumnMatchNote
from app.analysis.metadata import extract_metadata


def analyze_columns_fallback(
    task_type: TaskType,
    columns: list[str],
    df: pd.DataFrame,
) -> AnalysisResponse:
    """OpenAI API 키가 없을 때 작동하는 규칙 기반 컬럼 매핑 폴백 함수"""
    column_mappings = []

    for col in columns:
        col_lower = col.lower()
        role = ColumnRole.ignore

        if "id" in col_lower or "index" in col_lower:
            role = ColumnRole.sample_id
        elif col_lower in ["y_true", "actual", "ground_truth", "label", "target"]:
            if task_type == TaskType.multilabel:
                role = ColumnRole.true_labels
            else:
                role = ColumnRole.y_true
        elif col_lower in ["y_pred", "predicted", "pred", "prediction"]:
            if task_type == TaskType.multilabel:
                role = ColumnRole.pred_labels
            else:
                role = ColumnRole.y_pred
        elif task_type == TaskType.binary and ("score" in col_lower or "prob" in col_lower or "pos" in col_lower):
            role = ColumnRole.score_positive
        elif task_type == TaskType.multiclass and ("prob" in col_lower or "p_" in col_lower or "class_" in col_lower):
            role = ColumnRole.prob_per_class
        elif task_type == TaskType.multilabel and ("score" in col_lower or "prob" in col_lower or "p_" in col_lower):
            role = ColumnRole.score_per_label

        samples = []
        if col in df.columns:
            samples = [str(v) for v in df[col].dropna().head(3).tolist()]

        column_mappings.append(
            ColumnMapping(
                column=col,
                role=role,
                sample_values=samples
            )
        )

    sample_df = df.head(30)
    metadata = extract_metadata(task_type, df, sample_df, column_mappings)

    # 어느 규칙에도 걸리지 않은 컬럼은 조용히 ignore 가 된다 — 그 사실을 알린다
    # (ISSUES.md B-03). elif 사슬에 else 가 없어 종전에는 안내가 **항상 비어 있었고**,
    # 무키·예산 소진처럼 **가장 안내가 필요한 순간에** 사용자는 아무 설명도 못 받았다.
    # (LLM 경로의 reconcile 은 오히려 성실히 안내를 만든다 — 비어 있던 것은 이쪽이다.)
    column_notes = [
        ColumnMatchNote(
            llm_column=m.column,
            matched_column=None,
            status="unmapped_header",
            message=(
                f"'{m.column}' 컬럼의 역할을 규칙으로 판단하지 못해 평가에서 제외했습니다"
                "(자동 매핑이 규칙 기반으로 동작했습니다). 필요하면 매핑 화면에서 직접 지정해 주세요."
            ),
        )
        for m in column_mappings
        if m.role == ColumnRole.ignore
    ]

    return AnalysisResponse(
        task_type=task_type,
        column_mappings=column_mappings,
        metadata=metadata,
        column_notes=column_notes,
    )
