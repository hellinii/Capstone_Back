"""app/narrative/schemas.py — 서술 도메인 스키마(FactSheet·서술 요청/응답·출력 모델)."""

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator
from app.core.schemas import ReportPurpose, TaskType


# 입력 상한(G-03). 무인증 과금 경로라 배열 길이가 곧 프롬프트 토큰 수다.
# 실측: metrics/per_class 각 20,000개 → 5,369,661자(~1.34M 토큰).
# 정상 사용 시 지표는 M1~M23 (23개), 클래스는 많아야 수십 개다.
MAX_METRIC_FACTS = 64
MAX_CLASS_ENTRIES = 256
# 고객 입력(client_inputs)의 직렬화 길이 상한. 정상 입력은 수천 자다(카드 다섯 장의 짧은 답과
# 자유서술 몇 칸). 무인증 과금 경로라 배열 상한과 같은 이유로 막는다(G-03).
MAX_CLIENT_INPUTS_CHARS = 20_000


class MetricFact(BaseModel):
    """fact_sheet 내 개별 지표 (이미 계산된 값)"""
    metric_id:        str          = Field(description="Metric 표시 ID (예: M1)")
    display_name: str          = Field(description="지표명 (예: Accuracy)")
    value:        float        = Field(description="산출값")
    threshold:    float | None = Field(default=None, description="합격 임계값(없으면 정보 제공)")
    status:       str          = Field(description="pass | fail | warning | info")
    is_core:      bool         = Field(default=False, description="핵심 지표 — 미달하면 불합격(서술 설계 D19)")


class PerClassFact(BaseModel):
    label:     str          = Field(description="클래스 라벨")
    precision: float | None = None
    recall:    float | None = None
    f1:        float | None = None
    support:   int   | None = None


class ConfusionFact(BaseModel):
    labels: list[str]        = Field(default=[], max_length=MAX_CLASS_ENTRIES, description="클래스 라벨")
    matrix: list[list[int]]  = Field(default=[], max_length=MAX_CLASS_ENTRIES, description="혼동 행렬")
    positive_class: str | None = Field(default=None, description="양성 클래스 라벨(2x2 FN/FP 매핑 기준)")

    @field_validator("matrix")
    @classmethod
    def _rows_within_limit(cls, v: list[list[int]]) -> list[list[int]]:
        # 바깥 길이만 재면 한 행에 전부 밀어넣어 우회할 수 있다(G-03).
        for row in v:
            if len(row) > MAX_CLASS_ENTRIES:
                raise ValueError(f"혼동 행렬의 행 길이가 상한({MAX_CLASS_ENTRIES})을 넘습니다.")
        return v


class DistributionFact(BaseModel):
    class_distribution: dict[str, int] = Field(
        default={},
        description=(
            "클래스별 등장 횟수. multilabel 은 레이블 등장 횟수라 합계가 표본 수를 넘는다 "
            "— 표본 수는 FactSheet.n_samples 를 쓸 것(ISSUES.md B-02)."
        ),
    )
    imbalance_ratio:    float | None   = Field(default=None, description="불균형 비율")

    @field_validator("class_distribution")
    @classmethod
    def _keys_within_limit(cls, v: dict[str, int]) -> dict[str, int]:
        if len(v) > MAX_CLASS_ENTRIES:
            raise ValueError(f"클래스 수가 상한({MAX_CLASS_ENTRIES})을 넘습니다.")
        return v


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
    metrics:      list[MetricFact]   = Field(default=[], max_length=MAX_METRIC_FACTS, description="지표 결과")
    per_class:    list[PerClassFact] = Field(default=[], max_length=MAX_CLASS_ENTRIES, description="클래스별 지표")
    confusion:    ConfusionFact | None     = None
    distribution: DistributionFact | None  = None
    verdict:      str   = Field(description="PASS | CONDITIONAL_PASS | FAIL | INCONCLUSIVE (프론트 규칙 산출, D19)")
    score:        float = Field(description="통과율 점수 (프론트 규칙 산출)")
    latency:      LatencyFact = Field(default_factory=LatencyFact)


class NarrativeRequest(BaseModel):
    """[Step 4] LLM 서술 생성 요청"""
    task_type:      TaskType      = Field(description="분류 모델 유형")
    report_purpose: ReportPurpose = Field(default=ReportPurpose.external, description="internal | external | project")
    fact_sheet:     FactSheet     = Field(description="평가 결과 사실 시트")
    client_inputs:  dict[str, Any] | None = Field(
        default=None,
        description=(
            "고객이 제공한 입력(서술 설계 D25). 선언, 학습·평가 데이터, 정답 라벨, 실행 환경 등. "
            "회사·과제 정보는 담지 않는다(D34). 계산값이 아니라 의뢰자 제공 정보다."
        ),
    )

    @field_validator("client_inputs")
    @classmethod
    def _client_inputs_within_limit(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return v
        if len(json.dumps(v, ensure_ascii=False)) > MAX_CLIENT_INPUTS_CHARS:
            raise ValueError(f"고객 입력이 상한({MAX_CLIENT_INPUTS_CHARS}자)을 넘습니다.")
        return v


class InterpretationOut(BaseModel):
    confusion_analysis:    str = ""
    distribution_analysis: str = ""


class ConclusionOut(BaseModel):
    verdict:   str = Field(description="PASS | CONDITIONAL_PASS | FAIL | INCONCLUSIVE (서버가 fact_sheet 값으로 강제)")
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
