"""
schemas.py — API 요청/응답 데이터 형태 정의
"""

from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Any


class TaskType(str, Enum):
    binary     = "binary"
    multiclass = "multiclass"
    multilabel = "multilabel"


class ReportPurpose(str, Enum):
    """성적서 용도. 허용값 외 문자열은 pydantic 이 422 로 거부(프롬프트 주입 여지 차단, D7[4])."""
    internal = "internal"
    external = "external"
    project  = "project"


class ColumnRole(str, Enum):
    """
    ISO/IEC TS 4213:2022 기반 컬럼 역할 정의.

    [Binary]      sample_id, y_true, y_pred, score_positive, ignore
    [Multiclass]  sample_id, y_true, y_pred, prob_per_class, ignore
    [Multilabel]  sample_id, true_labels, pred_labels, score_per_label, ignore
    """
    sample_id       = "sample_id"
    ignore          = "ignore"

    # 공통(선택) — 추론 지연시간 컬럼(ms 가정). 모든 task_type에서 선택적으로 매핑 가능.
    latency         = "latency"

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
        ColumnRole.score_positive, ColumnRole.latency, ColumnRole.ignore,
    ],
    TaskType.multiclass: [
        ColumnRole.sample_id, ColumnRole.y_true, ColumnRole.y_pred,
        ColumnRole.prob_per_class, ColumnRole.latency, ColumnRole.ignore,
    ],
    TaskType.multilabel: [
        ColumnRole.sample_id, ColumnRole.true_labels, ColumnRole.pred_labels,
        ColumnRole.score_per_label, ColumnRole.latency, ColumnRole.ignore,
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
    column_unique_values: dict[str, list[str]] = Field(default={}, description="컬럼별 전체 고유값 목록")



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
    results:            dict[str, Any] = Field(description="TC별 연산 결과 수치 맵 (예: {'TC1': 0.95})")
    warnings:           list[str]      = Field(default=[], description="전처리 단계에서 발생한 경고 로그 목록")
    dropped_rows:       int            = Field(default=0, description="제거된 결측치 행 수")
    class_distribution: dict[str, int] = Field(default={}, description="클래스별 샘플 수")


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


# ── Step 4: LLM 서술 생성 (성적서 7·8·9절) ────────────────────────────────────
#
# 핵심 원칙: LLM은 수치 생산자가 아니라 "번역기"다. 모든 숫자는 프론트가 계산해
# fact_sheet 로 보내고, 백엔드는 그 수치를 한국어 산문으로 옮기기만 한다.

class MetricFact(BaseModel):
    """fact_sheet 내 개별 지표 (이미 계산된 값)"""
    tc_id:        str          = Field(description="Metric 표시 ID (예: M1)")
    display_name: str          = Field(description="지표명 (예: Accuracy)")
    value:        float        = Field(description="산출값")
    threshold:    float | None = Field(default=None, description="합격 임계값(없으면 정보 제공)")
    status:       str          = Field(description="pass | fail | warning | info")


class PerClassFact(BaseModel):
    label:     str          = Field(description="클래스 라벨")
    precision: float | None = None
    recall:    float | None = None
    f1:        float | None = None
    support:   int   | None = None


class ConfusionFact(BaseModel):
    labels: list[str]        = Field(default=[], description="클래스 라벨")
    matrix: list[list[int]]  = Field(default=[], description="혼동 행렬")
    positive_class: str | None = Field(default=None, description="양성 클래스 라벨(2x2 FN/FP 매핑 기준)")


class DistributionFact(BaseModel):
    class_distribution: dict[str, int] = Field(default={}, description="클래스별 샘플 수")
    imbalance_ratio:    float | None   = Field(default=None, description="불균형 비율")


class LatencyFact(BaseModel):
    available: bool          = Field(default=False, description="측정 여부")
    mean: float | None = None
    p50:  float | None = None
    p95:  float | None = None
    p99:  float | None = None
    unit: str = "ms"


class FactSheet(BaseModel):
    """프론트가 평가 결과로 조립해 보내는 '사실 시트'. LLM은 이 안의 숫자만 사용한다."""
    n_samples:    int  = Field(default=0, description="평가 샘플 수")
    dropped_rows: int  = Field(default=0, description="제외된 행 수")
    metrics:      list[MetricFact]   = Field(default=[], description="지표 결과")
    per_class:    list[PerClassFact] = Field(default=[], description="클래스별 지표")
    confusion:    ConfusionFact | None     = None
    distribution: DistributionFact | None  = None
    verdict:      str   = Field(description="PASS | CONDITIONAL_PASS | FAIL (프론트 규칙 산출)")
    score:        float = Field(description="통과율 점수 (프론트 규칙 산출)")
    latency:      LatencyFact = Field(default_factory=LatencyFact)


class NarrativeRequest(BaseModel):
    """[Step 4] LLM 서술 생성 요청"""
    task_type:      TaskType      = Field(description="분류 모델 유형")
    report_purpose: ReportPurpose = Field(default=ReportPurpose.external, description="internal | external | project")
    fact_sheet:     FactSheet     = Field(description="평가 결과 사실 시트")


class InterpretationOut(BaseModel):
    confusion_analysis:    str = ""
    distribution_analysis: str = ""


class ConclusionOut(BaseModel):
    verdict:   str = Field(description="PASS | CONDITIONAL_PASS | FAIL (서버가 fact_sheet 값으로 강제)")
    benchmark: str = ""
    narrative: str = ""
    risks:     str = ""


class RecommendationNarrativeOut(BaseModel):
    data_quality: str = ""
    model_ops:    str = ""


class RecommendationOut(BaseModel):
    priority:        str = Field(description="HIGH | MEDIUM | LOW")
    category:        str
    action:          str
    expected_impact: str


class GroundingInfo(BaseModel):
    checked:    int       = 0
    violations: list[str] = []
    passed:     bool      = True


class NarrativeMeta(BaseModel):
    source:    str             = Field(description="llm | fallback")
    model:     str | None      = None
    reason:    str | None      = Field(default=None, description="폴백 사유: no_key | api_error | grounding_failed")
    grounding: GroundingInfo   = Field(default_factory=GroundingInfo)


class NarrativeResponse(BaseModel):
    """[Step 4] LLM 서술 생성 결과 (성적서 7·8·9절)"""
    interpretation:          InterpretationOut
    conclusion:              ConclusionOut
    recommendation_narrative: RecommendationNarrativeOut
    recommendations:         list[RecommendationOut] = []
    meta:                    NarrativeMeta


# ── 발급 메타 (수행기관 / 성적서 번호 / 발급 이력) — 설계 문서 P2-11 §5 ──────────
#
# 하나의 IssuanceOut 으로 프론트의 meta.reportId + performer + signature 를 채운다.

class OrganizationOut(BaseModel):
    """수행기관(performer) 조회 응답."""
    org_name:   str        = Field(description="수행기관명")
    department: str | None = Field(default=None, description="부서(issuer 조합용)")
    evaluator:  str | None = Field(default=None, description="평가자(performer.evaluator)")
    contact:    str | None = Field(default=None, description="연락처")
    address:    str | None = Field(default=None, description="주소(선택)")


class OrganizationIn(BaseModel):
    """(선택) 기관 정보 수정 요청."""
    org_name:   str        = Field(description="수행기관명")
    department: str | None = Field(default=None, description="부서")
    evaluator:  str | None = Field(default=None, description="평가자")
    contact:    str | None = Field(default=None, description="연락처")
    address:    str | None = Field(default=None, description="주소")


class IssueRequest(BaseModel):
    """[발급] 채번 요청. 같은 run_id 로 재호출 시 신규 채번 없이 기존 발급본 반환(멱등)."""
    run_id:        str        = Field(min_length=1, description="프론트 워크스페이스 run 식별자(멱등 키)")
    model_name:    str | None = Field(default=None, description="대상 모델명")
    model_version: str | None = Field(default=None, description="대상 모델 버전")
    note:          str | None = Field(default=None, description="발급 비고(미지정 시 '최초 발급')")
    issuer:        str | None = Field(default=None, description="발급자(미지정 시 기관 기본값)")

    @field_validator("run_id")
    @classmethod
    def _run_id_not_blank(cls, v: str) -> str:
        # 공백/빈 문자열은 멱등 키로 부적합 — 서로 다른 평가가 한 번호로 병합되는 것을 차단.
        if not v or not v.strip():
            raise ValueError("run_id 는 비어 있을 수 없습니다.")
        return v


class ReissueRequest(BaseModel):
    """[재발급] 정정 발급 요청. 같은 번호 유지 + 버전 차수 증가(v1.0→v1.1)."""
    note:   str        = Field(description="정정 사유(필수)")
    issuer: str | None = Field(default=None, description="발급자(미지정 시 기관 기본값)")


class IssuanceHistoryItem(BaseModel):
    """발급 이력 1건 → signature.history 요소."""
    version:   str        = Field(description="발급 버전")
    issued_at: str        = Field(description="발급 일시(ISO8601)")
    note:      str | None = Field(default=None, description="비고")


class IssuanceOut(BaseModel):
    """발급 결과 — meta.reportId + performer + signature 를 한 번에 채운다."""
    report_no:    str                       = Field(description="성적서 번호 → meta.reportId")
    version:      str                       = Field(description="최신 발급 버전(current_version)")
    issuer:       str                       = Field(description="발급자 → signature.issuer")
    issued_at:    str                       = Field(description="최신 발급 일시(ISO8601) → signature.signedAt")
    organization: OrganizationOut           = Field(description="수행기관 → performer")
    history:      list[IssuanceHistoryItem] = Field(description="발급 이력 → signature.history")
