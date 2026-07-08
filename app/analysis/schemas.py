"""app/analysis/schemas.py — 분석 도메인 요청/응답 스키마(컬럼 매핑·매핑 검증·데이터 검증)."""

from pydantic import BaseModel, Field
from app.core.schemas import ColumnMapping, DataMetadata, TaskType


class ColumnMatchNote(BaseModel):
    """LLM 반환 컬럼명과 실제 헤더 대조 결과(보정/제외/미매핑 안내). 프론트 배너용."""
    llm_column:      str        = Field(description="LLM이 반환한 컬럼명(미반환 헤더 보완 시 빈 문자열)")
    matched_column:  str | None = Field(default=None, description="실제 데이터의 매칭 컬럼명(없으면 None)")
    status:          str        = Field(description="corrected | unmatched | unmapped_header")
    message:         str        = Field(description="사용자 안내 메시지")


class AnalysisResponse(BaseModel):
    """[Step 1] LLM 컬럼 자동 매핑 결과 + 데이터 메타데이터"""
    task_type:       TaskType            = Field(description="분류 모델 유형")
    column_mappings: list[ColumnMapping] = Field(description="컬럼별 역할 매핑")
    metadata:        DataMetadata        = Field(description="데이터에서 추출한 클래스/레이블 정보")
    column_notes:    list[ColumnMatchNote] = Field(default=[], description="컬럼명 대조 보정/제외 안내(없으면 빈 배열)")


class ConfirmMappingRequest(BaseModel):
    """[Step 2] 사용자가 검토·수정 후 확정한 매핑 제출"""
    task_type:       TaskType           = Field(description="분류 모델 유형")
    column_mappings: list[ColumnMapping] = Field(description="확정된 컬럼 매핑 목록")
    selected_tcs:    list[str]          = Field(default=[], description="사전에 선택된 평가 지표(TC) 목록")


class MappingValidationError(BaseModel):
    code:    str = Field(description="오류 코드")
    message: str = Field(description="오류 메시지")


class MappingValidationWarning(BaseModel):
    code:    str = Field(description="경고 코드")
    message: str = Field(description="경고 메시지")


class ConfirmMappingResponse(BaseModel):
    """[Step 2] 매핑 확정 검증 결과"""
    is_valid:           bool                          = Field(description="TC 계산 진행 가능 여부")
    errors:             list[MappingValidationError]  = Field(description="치명적 오류 목록")
    warnings:           list[MappingValidationWarning] = Field(description="경고 목록")
    available_tcs:      list[str]                     = Field(description="계산 가능한 TC 목록")
    unavailable_tcs:    list[str]                     = Field(description="계산 불가 TC 목록")
    confirmed_mappings: list[ColumnMapping]           = Field(description="확정된 매핑 (그대로 반환)")


class ValidationCheckItem(BaseModel):
    """개별 검증 항목 결과"""
    name:     str = Field(description="검증 항목 이름 (예: Missing value)")
    result:   str = Field(description="검증 결과 (예: 3 rows, None)")
    handling: str = Field(description="처리 방법 (예: Exclude affected rows)")
    status:   str = Field(description="pass | warning | error")
    group:    str = Field(description="common | binary | multiclass | multilabel | latency")


class ExecutionSummaryItem(BaseModel):
    """실행 요약 항목"""
    label: str = Field(description="항목 라벨")
    value: str = Field(description="항목 값")
    note:  str = Field(description="부가 설명")


class ValidateDataResponse(BaseModel):
    """[Step 2.5] 데이터 검증(전처리 dry-run) 결과"""
    task_type:          str                       = Field(description="분류 모델 유형")
    selected_metric_ids: list[str]                = Field(description="선택된 메트릭 ID 목록")
    execution_summary:  list[ExecutionSummaryItem] = Field(description="실행 요약 테이블")
    validation_details: list[ValidationCheckItem]  = Field(description="개별 검증 항목 결과 목록")
    error_count:        int                       = Field(description="에러 수")
    warning_count:      int                       = Field(description="경고 수")
