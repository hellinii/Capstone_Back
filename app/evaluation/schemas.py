"""app/evaluation/schemas.py — 평가 도메인 요청/응답 스키마."""

from pydantic import BaseModel, Field
from typing import Any
from app.core.schemas import ColumnMapping, DataMetadata, TaskType


class EvaluateRequest(BaseModel):
    """[Step 3] 평가 실행 요청 스펙"""
    task_type:       TaskType            = Field(description="분류 모델 유형")
    column_mappings: list[ColumnMapping] = Field(description="확정된 컬럼 매핑 목록")
    selected_tcs:    list[str]           = Field(description="계산할 평가 지표(TC) 목록")
    metadata:        DataMetadata        = Field(description="클래스 및 positive_class 등이 들어있는 메타데이터")
    beta:            float               = Field(default=1.0, description="F-beta score 계산용 가중치 beta 값")


class EvaluateResponse(BaseModel):
    """[Step 3] 평가 결과 응답 스펙"""
    results:            dict[str, Any] = Field(description="TC별 연산 결과 수치 맵 (예: {'TC1': 0.95})")
    warnings:           list[str]      = Field(default=[], description="전처리 단계에서 발생한 경고 로그 목록")
    dropped_rows:       int            = Field(default=0, description="제거된 결측치 행 수")
    class_distribution: dict[str, int] = Field(default={}, description="클래스별 샘플 수")
