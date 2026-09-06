"""app/evaluation/schemas.py — 평가 도메인 요청/응답 스키마."""

from pydantic import BaseModel, Field, field_validator
from typing import Any
from app.core.schemas import ColumnMapping, DataMetadata, TaskType


class EvaluateRequest(BaseModel):
    """[Step 3] 평가 실행 요청 스펙"""
    task_type:       TaskType            = Field(description="분류 모델 유형")
    column_mappings: list[ColumnMapping] = Field(description="확정된 컬럼 매핑 목록")
    selected_metric_ids:    list[str]           = Field(description="계산할 평가 지표 목록")
    metadata:        DataMetadata        = Field(description="클래스 및 positive_class 등이 들어있는 메타데이터")
    beta:            float               = Field(default=1.0, description="F-beta score 계산용 가중치 beta 값")
    decision_threshold: float | dict[str, float] | None = Field(
        default=None,
        description=(
            "하드 예측이 없을 때 확률에서 예측을 파생하는 결정 임계값. "
            "스칼라면 전 컬럼 공통, dict 면 확률 컬럼명별 값(multilabel 레이블별 임계값). "
            "multiclass 는 argmax 라 사용하지 않는다. 생략하면 0.5. "
            "**성적서 합격 목표값(threshold)과는 다른 개념이다.**"
        ),
    )

    @field_validator("decision_threshold")
    @classmethod
    def _threshold_in_unit_interval(cls, v):
        """임계값은 확률과 같은 [0,1] 구간이다. 벗어나면 422 로 거절한다."""
        values = v.values() if isinstance(v, dict) else ([v] if v is not None else [])
        for item in values:
            if not (0.0 <= float(item) <= 1.0):
                raise ValueError(f"decision_threshold 는 0.0~1.0 이어야 합니다(받은 값: {item}).")
        return v


class EvaluateResponse(BaseModel):
    """[Step 3] 평가 결과 응답 스펙"""
    results:            dict[str, Any] = Field(description="지표별 연산 결과 수치 맵 (예: {'M1': 0.95})")
    warnings:           list[str]      = Field(default=[], description="전처리 단계에서 발생한 경고 로그 목록")
    dropped_rows:       int            = Field(default=0, description="제거된 결측치 행 수")
    n_samples:          int            = Field(default=0, description="실제로 지표를 계산한 행 수(결측 제거 후)")
    class_distribution: dict[str, int] = Field(
        default={},
        description=(
            "클래스별 등장 횟수. binary/multiclass 는 각 클래스의 샘플 수와 같지만, "
            "multilabel 은 **레이블 등장 횟수**라 합계가 n_samples 를 넘는다. "
            "표본 수가 필요하면 n_samples 를 쓸 것."
        ),
    )
