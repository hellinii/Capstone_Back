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


class LimitExceeded(ValueError):
    """입력 상한을 넘었다(G-03). 옛 경로에서는 다른 ValueError 처럼 422 가 되고, 작업 번호 경로는 이 종류의
    오류만 골라 상한 안으로 줄인 뒤 규칙 문장으로 끝낸다(app.narrative.jobs — 사유 input_too_large)."""


def client_inputs_length(v: dict[str, Any]) -> int:
    """고객 입력의 직렬화 길이 — 공백 없는 JSON(구분자 ","·":", 한글은 그대로)으로 잰다(B7).

    프론트 `fitClientInputs`(Capstone_Front/src/lib/report/reportMaterials.ts)가 `JSON.stringify` 로
    재는 것과 같은 기준이다. 문자열·정수·보통 소수는 두 쪽 길이가 같고, JS 는 UTF-16 단위로 세므로
    (이모지 등은 2) 대개 프론트 값이 크거나 같다. 예외는 지수 표기 실수뿐이다(파이썬 '1e-07' 이 JS '1e-7'
    보다 한 자 길다) — 고객 입력은 글이라 사실상 없고, 프론트 목표(상한의 90%)의 여유가 덮는다(S9 검토).
    """
    return len(json.dumps(v, ensure_ascii=False, separators=(",", ":")))

# 6.0 신뢰구간 방식 한 줄(ci_method_note)의 길이 상한. 방식 종류를 나열한 한 문장이라 수백 자를 넘지 않는다.
MAX_CI_METHOD_NOTE_CHARS = 500
# 9절 문제 목록(problems)의 상한(서술 설계 D22). 문제는 종류(9종)마다 많아야 하나라 9개를 넘지 않는다 —
# 여유를 두되 무인증 과금 경로라 막는다(G-03). 근거 수치(evidence)는 자유 dict 라 직렬화 길이로 잰다:
# 가장 큰 것은 클래스마다 학습·평가 비율을 싣는 분포 차이로, 클래스 256개에서도 수만 자를 넘지 않는다.
MAX_PROBLEMS = 16
MAX_PROBLEM_EVIDENCE_CHARS = 20_000
MAX_PROBLEM_TEXT_CHARS = 1_000

# 문제 종류 9종(서술 설계 D22). 순서가 우선순위 순서다(프론트 problemList.ts 의 KINDS 와 같다).
ProblemKind = Literal[
    "core_fail",
    "inconclusive",
    "noncore_fail",
    "core_near_target",
    "threshold_tuning",
    "imbalance_effect",
    "weakest_class",
    "distribution_shift",
    "label_caution",
]


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
    # 목표값과의 차이(서술 설계 D12). 프론트가 판정 대상 지표에만 싣는다. 높을수록 좋은 지표는
    # value − threshold, 낮을수록 좋은 지표는 threshold − value — 그래서 0 이상이면 충족 쪽이다.
    margin:       float | None = Field(default=None, description="목표값과의 차이(부호 그대로, 0 이상이면 충족 쪽)")
    # 아슬아슬 — |margin| < |threshold| × 3%(엄격 미만). 판정은 프론트 computeVerdict 한 곳에서 하고
    # 서버는 받은 값을 읽기만 한다. 판정 대상이 아니거나 옛 요청이면 False.
    near_target:  bool         = Field(default=False, description="목표값과의 차이가 목표값의 3% 미만")


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
                raise LimitExceeded(f"혼동 행렬의 행 길이가 상한({MAX_CLASS_ENTRIES})을 넘습니다.")
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
            raise LimitExceeded(f"클래스 수가 상한({MAX_CLASS_ENTRIES})을 넘습니다.")
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
            raise LimitExceeded(f"레이블별 임계값 수가 상한({MAX_CLASS_ENTRIES})을 넘습니다.")
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
            raise LimitExceeded(f"레이블별 임계값 수가 상한({MAX_CLASS_ENTRIES})을 넘습니다.")
        return v


class F1ThresholdGainFact(BaseModel):
    """임계값 조정 여지(서술 설계 D40) — 원본 PR 곡선의 최대 F1 과 지금 예측의 F1(4자리).

    평가 응답 success_metrics.f1_threshold_gain 을 프론트가 그대로 옮겨 싣는다. 이진이고 점수가 있을 때만.
    gain 은 같은 평가 데이터로 고른 임계값의 이득이라 '이 데이터에서 오를 수 있는 최대폭'이다.
    """
    max_f1:         float = Field(ge=0.0, le=1.0)
    current_f1:     float = Field(ge=0.0, le=1.0)
    gain:           float = Field(ge=0.0, le=1.0, description="max(0, max_f1 − current_f1)")
    best_threshold: float = Field(description="최대 F1 을 내는 결정 임계값")
    notable:        bool | None = Field(
        default=None,
        description="엔진이 반올림 전 gain 으로 정한 '임계값 부적합'(≥ 0.10, S2). 없으면(옛 요청) gain 으로 판정",
    )


class ImbalanceEffectFact(BaseModel):
    """불균형 영향(서술 설계 D20) — 평가 응답 success_metrics.imbalance_effect 를 그대로 옮겨 싣는다.

    level 판정(IR 1.5 미만 경미, 차이 0.10 이상 영향 있음)은 평가 엔진이 이미 했다. 서술은 level 을 읽기만 한다.
    """
    level: Literal["affected", "no_distortion", "mild"] = Field(
        description="affected(영향 있음) | no_distortion(불균형 있으나 왜곡 미관찰) | mild(경미)"
    )
    imbalance_ratio:            float        = Field(ge=1.0, description="다수/소수(M23 과 같은 정의)")
    accuracy_minus_macro_f1:    float | None = Field(default=None, description="정확도 − 매크로 F1. 멀티레이블은 None")
    majority_minus_minority_f1: float        = Field(description="다수 F1 − 소수 F1 (음수일 수 있다)")
    majority_class:             str
    minority_class:             str


class ProblemFact(BaseModel):
    """9절 권고가 대응할 문제 하나(서술 설계 D22). 프론트 problemList.ts 가 정해진 조건으로 만든다.

    서버는 문제를 판정하지 않는다 — 받은 목록에 권고를 묶기만 한다(번호·우선순위도 받은 그대로).
    """
    id:                str = Field(pattern=r"^P[0-9]{1,3}$", description="P1… 번호(우선순위 순)")
    kind:              ProblemKind
    priority:          Literal["HIGH", "MEDIUM", "LOW"]
    title:             str = Field(max_length=MAX_PROBLEM_TEXT_CHARS, description="문제 이름(지표·클래스 이름 포함)")
    source_section:    str = Field(default="", max_length=32, description="문제가 드러난 절(예: 8.2)")
    recommend_section: str = Field(default="", max_length=32, description="권고를 쓰는 절(9.1 데이터 | 9.2 모델)")
    evidence:          dict[str, Any] = Field(default={}, description="근거 수치 — 숫자는 grounding 허용 목록에 들어간다")
    note:              str | None = Field(default=None, max_length=MAX_PROBLEM_TEXT_CHARS, description="덧붙임")

    @field_validator("evidence")
    @classmethod
    def _evidence_within_limit(cls, v: dict[str, Any]) -> dict[str, Any]:
        # 자유 dict 라 배열 상한으로는 못 막는다 — 고객 입력과 같이 직렬화 길이로 잰다(G-03).
        if len(json.dumps(v, ensure_ascii=False)) > MAX_PROBLEM_EVIDENCE_CHARS:
            raise LimitExceeded(f"문제 근거 수치가 상한({MAX_PROBLEM_EVIDENCE_CHARS}자)을 넘습니다.")
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
    f1_threshold_gain: F1ThresholdGainFact | None = Field(
        default=None,
        description="임계값 조정 여지(D40) — 이진이고 점수가 있을 때만. 8.2·8.3 에서 쓰는 재료",
    )
    imbalance_effect: ImbalanceEffectFact | None = Field(
        default=None,
        description="불균형 영향(D20) — 예측이 있을 때만. 7.2 분포 해석의 결론(level)",
    )
    ci_method_note: str | None = Field(
        default=None,
        max_length=MAX_CI_METHOD_NOTE_CHARS,
        description="6.0 에 찍는 신뢰구간 방식 한 줄(D29) — 서술 참고용. 신뢰구간이 없으면 None",
    )
    problems: list[ProblemFact] = Field(
        default=[],
        max_length=MAX_PROBLEMS,
        description="9절 권고가 대응할 문제 목록(D22). 비어 있으면 권고도 없다",
    )

    @field_validator("problems")
    @classmethod
    def _problem_ids_unique(cls, v: list[ProblemFact]) -> list[ProblemFact]:
        # 권고는 번호로 문제에 묶인다 — 번호가 겹치면 어느 문제의 권고인지 정할 수 없다.
        ids = [p.id for p in v]
        if len(ids) != len(set(ids)):
            raise ValueError("문제 번호(id)가 겹칩니다.")
        return v


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
        if client_inputs_length(v) > MAX_CLIENT_INPUTS_CHARS:
            raise LimitExceeded(f"고객 입력이 상한({MAX_CLIENT_INPUTS_CHARS}자)을 넘습니다.")
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
    # 권고는 모두 문제 목록의 번호 하나에 묶인다(서술 설계 D22). 우선순위는 그 문제의 것을 따른다.
    problem_id:      str = Field(description="대응하는 문제 번호(fact_sheet.problems[].id)")
    priority:        str = Field(description="HIGH | MEDIUM | LOW (서버가 문제의 우선순위로 강제)")
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
    reason:    str | None      = Field(default=None, description="폴백 사유: no_key | api_error | grounding_failed | impact_number 등")
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


# ── 작업 번호(job) 경로 (서술 라운드 S3a) ─────────────────────────────────────
# 고객용과 내부용 기록을 나눈다(계획 원칙 6). 작업 응답에는 칸별 출처만 싣고, 강등 사유·호출 기록·토큰·비용은
# narrative_jobs.internal_json 에만 둔다. 그래서 옛 응답의 meta(source·model·reason·grounding)가 여기엔 없다.

JobStatus = Literal["running", "done", "failed"]
JobStage = Literal["queued", "generating", "checking", "regenerating", "done", "failed"]
# 실패 종류 — 프론트가 안내 문구를 고르는 데 쓰는 상태 이름이다(내부 강등 사유가 아니다).
# timeout: 전체 10분 제한을 넘김 · lost: 서버가 재시작되는 등으로 작업이 멈춤 · error: 작업 중 예기치 않은 오류 ·
# expired: 끝난 작업의 결과 글을 보관 시간이 지나 지움. 모두 "서술 다시 만들기"로 푼다.
JobFailure = Literal["timeout", "lost", "error", "expired"]
# 칸 출처 — llm(AI 생성) · fallback(규칙 문장으로 대체) · fixed(고정 문구). 칸 ID 는 소절 번호(7.1 …).
SectionSource = Literal["llm", "fallback", "fixed"]


class NarrativeJobResult(BaseModel):
    """끝난 작업의 서술 본문 — 옛 응답(NarrativeResponse)에서 meta 를 칸별 출처로 바꾼 모양."""
    interpretation:           InterpretationOut
    conclusion:               ConclusionOut
    recommendation_narrative: RecommendationNarrativeOut
    recommendations:          list[RecommendationOut] = []
    section_sources:          dict[str, SectionSource] = Field(
        description="칸(소절 번호)마다 출처. 지금 칸: 7.1·7.2·8.1·8.2·8.3·9 (7.3 은 S4b·S5 에서 더함)",
    )


class NarrativeJobCreated(BaseModel):
    """작업을 만든 즉시의 응답. 8.1 표는 서술 성공·실패와 무관하게 여기서 받는다(점검 유-2)."""
    job_id:         str
    status:         JobStatus
    stage:          JobStage
    benchmark_refs: list[BenchmarkRefOut] = []


class NarrativeJobOut(BaseModel):
    """작업 조회 응답. result 는 status 가 done 일 때만 있다."""
    job_id:         str
    status:         JobStatus
    stage:          JobStage
    failure:        JobFailure | None = None
    benchmark_refs: list[BenchmarkRefOut] = Field(
        default=[], description="만들 때 계산한 8.1 표 — 새로고침 뒤 이어 조회해도 다시 받는다(결과 글을 지우면 빈 배열)",
    )
    result:         NarrativeJobResult | None = None
    created_at:     str = Field(description="UTC ISO 8601(끝에 Z)")
    updated_at:     str = Field(description="마지막으로 단계가 바뀐 시각, UTC ISO 8601(끝에 Z)")
