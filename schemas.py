"""
schemas.py — API 요청/응답 데이터 형태 정의
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Any


class TaskType(str, Enum):
    binary     = "binary"
    multiclass = "multiclass"
    multilabel = "multilabel"


class ColumnRole(str, Enum):
    """
    ISO/IEC TS 4213:2022 기반 컬럼 역할 정의.

    [Binary]      sample_id, y_true, y_pred, score_positive, ignore
    [Multiclass]  sample_id, y_true, y_pred, prob_per_class, ignore
    [Multilabel]  sample_id, true_labels, pred_labels, score_per_label, ignore
    """
    sample_id       = "sample_id"
    ignore          = "ignore"

    # Binary / Multiclass
    y_true          = "y_true"
    y_pred          = "y_pred"

    # Binary 전용
    score_positive  = "score_positive"

    # Multiclass 전용
    prob_per_class  = "prob_per_class"

    # Multilabel 전용
    true_labels     = "true_labels"
    pred_labels     = "pred_labels"
    score_per_label = "score_per_label"


# task_type별 허용 역할 목록
VALID_ROLES_BY_TASK: dict[TaskType, list[ColumnRole]] = {
    TaskType.binary: [
        ColumnRole.sample_id, ColumnRole.y_true, ColumnRole.y_pred,
        ColumnRole.score_positive, ColumnRole.ignore,
    ],
    TaskType.multiclass: [
        ColumnRole.sample_id, ColumnRole.y_true, ColumnRole.y_pred,
        ColumnRole.prob_per_class, ColumnRole.ignore,
    ],
    TaskType.multilabel: [
        ColumnRole.sample_id, ColumnRole.true_labels, ColumnRole.pred_labels,
        ColumnRole.score_per_label, ColumnRole.ignore,
    ],
}


# ── Step 1: LLM 분석 결과 ─────────────────────────────────────────────────────

class ColumnMapping(BaseModel):
    """컬럼 → 역할 매핑 (LLM 결과 or 사용자 확정)"""
    column: str        = Field(description="파일의 컬럼명")
    role:   ColumnRole = Field(description="ISO 4213 기준 역할")
    sample_values: list[str] = Field(default=[], description="파일에서 추출한 샘플 값 3개 내외")


class DataMetadata(BaseModel):
    """
    파일 데이터에서 자동으로 추출한 클래스/레이블 메타데이터.

    [Binary]
      - positive_class: 양성 클래스 값 (e.g., "1", "yes", "spam")
      - negative_class: 음성 클래스 값 (e.g., "0", "no", "ham")
      - positive_class_ambiguous: True이면 자동 판단 불확실 → 사용자 확인 필요

    [Multiclass]
      - detected_classes: y_true에서 발견된 클래스 목록 (e.g., ["cat","dog","bird"])

    [Multilabel]
      - detected_labels: true_labels에서 파싱한 레이블 목록 (e.g., ["sports","news"])

    [공통]
      - class_distribution: 클래스(레이블)별 샘플 수
    """
    # Binary
    positive_class:           str | None       = Field(default=None, description="양성 클래스 값")
    negative_class:           str | None       = Field(default=None, description="음성 클래스 값")
    positive_class_ambiguous: bool             = Field(default=False, description="양성 클래스 자동 판단이 불확실한 경우 True")

    # Multiclass
    detected_classes: list[str]                = Field(default=[], description="감지된 클래스 목록 (Multiclass)")

    # Multilabel
    detected_labels:  list[str]                = Field(default=[], description="감지된 레이블 목록 (Multilabel)")

    # 공통
    class_distribution: dict[str, int]         = Field(default={}, description="클래스(또는 레이블)별 샘플 수")


class AnalysisResponse(BaseModel):
    """[Step 1] LLM 컬럼 자동 매핑 결과 + 데이터 메타데이터"""
    task_type:       TaskType            = Field(description="분류 모델 유형")
    column_mappings: list[ColumnMapping] = Field(description="컬럼별 역할 매핑")
    metadata:        DataMetadata        = Field(description="데이터에서 추출한 클래스/레이블 정보")


# ── Step 2: 사용자 확정 매핑 검증 ─────────────────────────────────────────────

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


# ── Step 3: 최종 평가 실행 ─────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    """[Step 3] 평가 실행 요청 스펙"""
    task_type:       TaskType            = Field(description="분류 모델 유형")
    column_mappings: list[ColumnMapping] = Field(description="확정된 컬럼 매핑 목록")
    selected_tcs:    list[str]           = Field(description="계산할 평가 지표(TC) 목록")
    metadata:        DataMetadata        = Field(description="클래스 및 positive_class 등이 들어있는 메타데이터")
    beta:            float               = Field(default=1.0, description="F-beta score 계산용 가중치 beta 값")


class EvaluateResponse(BaseModel):
    """[Step 3] 평가 결과 응답 스펙"""
    results:      dict[str, Any] = Field(description="TC별 연산 결과 수치 맵 (예: {'TC1': 0.95})")
    warnings:     list[str]      = Field(default=[], description="전처리 단계에서 발생한 경고 로그 목록")
    dropped_rows: int            # 제거된 결측치(NaN) 행 수


# ── Step 2.5: 데이터 검증 (Validation) ────────────────────────────────────────

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
