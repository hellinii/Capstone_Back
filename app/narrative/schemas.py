"""app/narrative/schemas.py — 서술 도메인 스키마(FactSheet·서술 요청/응답·출력 모델)."""

import json
from typing import Any, Literal

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
# 6.0 신뢰구간 방식 한 줄(ci_method_note)의 길이 상한. 방식 종류를 나열한 한 문장이라 수백 자를 넘지 않는다.
MAX_CI_METHOD_NOTE_CHARS = 500


class MetricFact(BaseModel):
    """fact_sheet 내 개별 지표 (이미 계산된 값)"""
    metric_id:        str          = Field(description="Metric 표시 ID (예: M1)")
    display_name: str          = Field(description="지표명 (예: Accuracy)")
    value:        float        = Field(description="산출값")
    threshold:    float | None = Field(default=None, description="합격 임계값(없으면 정보 제공)")
    status:       str          = Field(description="pass | fail | warning | info")
    is_core:      bool         = Field(default=False, description="핵심 지표 — 미달하면 불합격(서술 설계 D19)")
    # 95% 신뢰구간(서술 설계 D29). 프론트가 평가 응답 confidence_intervals 에서 판정 대상 지표에만 싣는다.
    # 구간을 내지 못한 지표(skipped)와 옛 요청은 둘 다 None.
    ci_low:       float | None = Field(default=None, description="95% 신뢰구간 하한")
    ci_high:      float | None = Field(default=None, description="95% 신뢰구간 상한")


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


class ConfusionCountFact(BaseModel):
    """TP·TN·FP·FN 건수 한 줄. 이진은 양성 클래스 한 줄, 다중·멀티는 클래스(레이블)마다(일대다)."""
    label: str
    tp: int
    fp: int
    fn: int
    tn: int


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


class DerivedPredictionFact(BaseModel):
    """예측을 확률·점수에서 파생했다는 사실(재료 목록 9절 "파생 예측 사실", 고정 문구 9)."""
    method:         str = Field(description="threshold | argmax | threshold_per_label")
    threshold:      float | dict[str, float] | None = Field(
        default=None, description="결정 임계값. 멀티레이블은 레이블(컬럼)별, argmax 는 없음"
    )
    source_columns: list[str] = Field(default=[], max_length=MAX_CLASS_ENTRIES, description="파생에 쓴 점수 컬럼")

    @field_validator("threshold")
    @classmethod
    def _per_label_within_limit(cls, v):
        if isinstance(v, dict) and len(v) > MAX_CLASS_ENTRIES:
            raise ValueError(f"레이블별 임계값 수가 상한({MAX_CLASS_ENTRIES})을 넘습니다.")
        return v


class ErrorCountFact(BaseModel):
    """경계 오류·확신 오류 한 종류의 건수와 비율(ratio = count / misclassified, 4자리)."""
    count: int          = Field(ge=0)
    ratio: float | None = Field(default=None, ge=0.0, le=1.0, description="오분류가 0건이면 None")


class ErrorRatiosFact(BaseModel):
    """오분류의 성격 — 결정 경계 근처(경계 오류)와 높은 확신으로 틀림(확신 오류)(서술 설계 D30).

    평가 응답 success_metrics.error_ratios 를 프론트가 그대로 옮겨 싣는다. 확률이 없으면 키 자체가 없다.
    """
    unit:          Literal["sample", "label_cell"] = Field(
        description="sample(이진·다중: 표본 단위) | label_cell(멀티레이블: 틀린 레이블 칸 단위)"
    )
    misclassified: int = Field(ge=0, description="분모 — 확률이 있는 오분류 수(멀티레이블은 틀린 칸 수)")
    boundary:      ErrorCountFact | None = Field(default=None, description="경계 오류. 임계값을 모르면 None")
    boundary_note: Literal["threshold_unknown"] | None = Field(
        default=None, description="boundary 가 None 인 이유 — 예측을 직접 올려 임계값을 모름"
    )
    confident:       ErrorCountFact
    boundary_margin: float = Field(default=0.1, description="경계 거리 기준(엄격 미만)")
    confident_level: float = Field(default=0.9, description="확신 기준(틀린 쪽 확률 이상)")
    threshold:       float | dict[str, float] | None = Field(
        default=None, description="경계 판정에 쓴 임계값. 멀티레이블은 컬럼별일 수 있고, 다중은 None"
    )

    @field_validator("threshold")
    @classmethod
    def _per_label_within_limit(cls, v):
        if isinstance(v, dict) and len(v) > MAX_CLASS_ENTRIES:
            raise ValueError(f"레이블별 임계값 수가 상한({MAX_CLASS_ENTRIES})을 넘습니다.")
        return v


class FactSheet(BaseModel):
    """프론트가 평가 결과로 조립해 보내는 '사실 시트'. LLM은 이 안의 숫자만 사용한다."""
    n_samples:    int  = Field(default=0, description="평가 샘플 수")
    dropped_rows: int  = Field(default=0, description="제외된 행 수")
    metrics:      list[MetricFact]   = Field(default=[], max_length=MAX_METRIC_FACTS, description="지표 결과")
    per_class:    list[PerClassFact] = Field(default=[], max_length=MAX_CLASS_ENTRIES, description="클래스별 지표")
    class_averages: list[PerClassFact] = Field(
        default=[],
        max_length=4,
        description=(
            "클래스별 표(M22)의 평균 줄 — 매크로·가중·마이크로(멀티레이블은 샘플 평균도). "
            "평균 방식 차이를 8.2 에 쓰는 재료(서술 설계 D17·D39). M22 미선택이면 빈 배열."
        ),
    )
    confusion:    ConfusionFact | None     = None
    confusion_counts: list[ConfusionCountFact] = Field(
        default=[],
        max_length=MAX_CLASS_ENTRIES,
        description="TP·TN·FP·FN 건수 — 혼동행렬(M21) 선택과 무관하게 늘 실린다(재료 목록 9절)",
    )
    distribution: DistributionFact | None  = None
    verdict:      str   = Field(description="PASS | CONDITIONAL_PASS | FAIL | INCONCLUSIVE (프론트 규칙 산출, D19)")
    score:        float = Field(description="통과율 점수 (프론트 규칙 산출)")
    latency:      LatencyFact = Field(default_factory=LatencyFact)
    derived_prediction: DerivedPredictionFact | None = Field(
        default=None,
        description="예측을 확률에서 파생했을 때만. 서술은 수치가 그 임계값의 함수임을 알아야 한다",
    )
    error_ratios: ErrorRatiosFact | None = Field(
        default=None,
        description="경계·확신 오류 비율(D30) — 확률이 있을 때만. 7.1 에서 오류의 성격을 쓰는 재료",
    )
    ci_method_note: str | None = Field(
        default=None,
        max_length=MAX_CI_METHOD_NOTE_CHARS,
        description="6.0 에 찍는 신뢰구간 방식 한 줄(D29) — 서술 참고용. 신뢰구간이 없으면 None",
    )


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


class BenchmarkRefOut(BaseModel):
    """8.1 에 숫자로 찍을 벤치마크 비교 한 줄 — 서버가 계산한 값 그대로(서술 설계 §4.8)."""
    metric:      str
    model_value: float
    ref_low:     float
    ref_high:    float
    position:    str = Field(description="above | within | below (단순 수치 위치)")
    direction:   str = Field(description="higher | lower (높을수록/낮을수록 좋음)")
    quality:     str = Field(description="better | within | worse (방향을 반영한 우열)")
    # 목표값 수준(서술 설계 D31). 목표값이 없는 지표는 둘 다 None.
    target_value:    float | None = Field(default=None, description="그 지표의 목표값(threshold)")
    target_position: str | None   = Field(
        default=None, description="below | within | above — 목표값의 참조 범위 대비 단순 수치 위치"
    )
    source_note: str = Field(description="기준의 출처·성격 — '내부 참조 기준치'")


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
    benchmark_refs:          list[BenchmarkRefOut] = Field(
        default=[],
        description=(
            "벤치마크 비교값. 8.1 에 참조 범위를 숫자로 찍는다(재료 목록 9절). 종전에는 AI 가 쓴 "
            "문장 안에만 나왔다. 서술 경로(llm·fallback)와 무관하게 늘 같은 값이다"
        ),
    )
