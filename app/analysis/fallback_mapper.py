"""app/analysis/fallback_mapper.py — 규칙 기반 컬럼 매핑 폴백

OpenAI 키가 없거나 LLM 호출이 실패했을 때 컬럼명 패턴으로 역할을 추정하는 폴백 매핑.

상호작용
- 의존(import): pandas, app.core.schemas, app.analysis.metadata(extract_metadata)
- 사용처: app.analysis.analysis_service (무키/LLM 실패 시 강등 경로)
"""
import pandas as pd

from app.core.schemas import AnalysisResponse, ColumnMapping, ColumnRole, TaskType
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

    return AnalysisResponse(
        task_type=task_type,
        column_mappings=column_mappings,
        metadata=metadata
    )
