"""app/core/schemas.py — 전 도메인 공유 계약(enum·역할 규칙·파이프라인 인계 모델).

TaskType/ColumnRole 등 공용 enum 과 지표 요건 테이블, 그리고 분석→평가로 넘어가는 인계
계약(ColumnMapping, DataMetadata)만 여기 둔다. 도메인별 요청/응답 스키마는 각 <도메인>/schemas.py.
"""

from enum import Enum
from pydantic import BaseModel, Field


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


VALID_ROLES_BY_TASK: dict[TaskType, list[ColumnRole]] = {
    TaskType.binary: [
        ColumnRole.sample_id, ColumnRole.y_true, ColumnRole.y_pred,
        ColumnRole.score_positive, ColumnRole.latency, ColumnRole.ignore,
    ],
    # 확률 역할은 세 task 모두에서 정식 입력이다(2026-09-07 결정 1).
    # 하드 예측이 없으면 확률에서 예측을 파생한다 — binary 는 임계값, multiclass 는 argmax,
    # multilabel 은 레이블별 임계값(PREDICTION_ROLES_BY_TASK 참조). 종전에는 "확률을 읽는
    # 지표가 없다"는 이유로 multiclass/multilabel 에서 뺐으나, 그 판단은 확률을 **예측의
    # 대체 입력**으로 보지 않고 지표의 직접 입력으로만 본 데서 나왔다.
    TaskType.multiclass: [
        ColumnRole.sample_id, ColumnRole.y_true, ColumnRole.y_pred,
        ColumnRole.prob_per_class, ColumnRole.latency, ColumnRole.ignore,
    ],
    TaskType.multilabel: [
        ColumnRole.sample_id, ColumnRole.true_labels, ColumnRole.pred_labels,
        ColumnRole.score_per_label, ColumnRole.latency, ColumnRole.ignore,
    ],
}


# task 별 "예측 역할"과 그것을 대신할 수 있는 확률 역할 — (주 역할, 대체 역할들).
#
# **왜 별도 표인가.** METRIC_REQUIREMENTS 의 값은 set 이라 AND 만 표현한다. SPEC §1~§3 이
# 규정한 "y_pred **또는** 확률"이라는 택일을 그 표에 넣으려면 자료구조를 바꿔야 하고,
# 그러면 프론트 계약 테스트의 고정 사본까지 재설계된다. 대신 택일을 이 표 하나로 분리해
# "예측 역할은 주 역할 또는 대체 역할 중 하나로 충족된다"는 규칙을 단일 출처로 둔다.
# validator(가용 지표 판정)와 preprocessor(파생)가 함께 읽는다.
#
# 대체 역할이 매핑되면 preprocess 단계에서 주 역할 컬럼을 **파생**해 붙인다. 파생값은
# 모델의 실제 출력이 아니므로 파생 사실과 임계값을 응답에 실어 성적서가 인쇄한다(SPEC §0).
PREDICTION_ROLES_BY_TASK: dict[TaskType, tuple[ColumnRole, tuple[ColumnRole, ...]]] = {
    TaskType.binary:     (ColumnRole.y_pred,      (ColumnRole.score_positive,)),
    TaskType.multiclass: (ColumnRole.y_pred,      (ColumnRole.prob_per_class,)),
    TaskType.multilabel: (ColumnRole.pred_labels, (ColumnRole.score_per_label,)),
}

# task 별 "정답 역할" — 어떤 지표를 고르든 항상 필수.
TRUTH_ROLE_BY_TASK: dict[TaskType, ColumnRole] = {
    TaskType.binary:     ColumnRole.y_true,
    TaskType.multiclass: ColumnRole.y_true,
    TaskType.multilabel: ColumnRole.true_labels,
}

# 한 역할에 여러 컬럼이 매핑될 수 있는 역할(확률 컬럼은 클래스·레이블마다 하나씩).
MULTI_COLUMN_ROLES: frozenset[ColumnRole] = frozenset({
    ColumnRole.ignore, ColumnRole.prob_per_class, ColumnRole.score_per_label,
})


METRIC_REQUIREMENTS: dict[TaskType, dict[str, set[ColumnRole]]] = {
    TaskType.binary: {
        "M1":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M2":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M3":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M4":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M5":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M6":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M7":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M8":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M9":  {ColumnRole.y_true, ColumnRole.score_positive},
        "M10": {ColumnRole.y_true, ColumnRole.score_positive},
        "M19": {ColumnRole.y_true, ColumnRole.score_positive},
        "M20": {ColumnRole.y_true, ColumnRole.y_pred},
        "M21": {ColumnRole.y_true, ColumnRole.y_pred},
        "M22": {ColumnRole.y_true, ColumnRole.y_pred},
        # M23 은 정답 분포만으로 계산된다(common.calculate_imbalance_ratio) — 예측 컬럼 불필요.
        "M23": {ColumnRole.y_true},
    },
    TaskType.multiclass: {
        "M1":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M2":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M3":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M4":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M5":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M6":  {ColumnRole.y_true, ColumnRole.y_pred},
        "M11": {ColumnRole.y_true, ColumnRole.y_pred},
        "M12": {ColumnRole.y_true, ColumnRole.y_pred},
        "M13": {ColumnRole.y_true, ColumnRole.y_pred},
        "M14": {ColumnRole.y_true, ColumnRole.y_pred},
        "M21": {ColumnRole.y_true, ColumnRole.y_pred},
        "M22": {ColumnRole.y_true, ColumnRole.y_pred},
        # M23 은 정답 분포만으로 계산된다(common.calculate_imbalance_ratio) — 예측 컬럼 불필요.
        "M23": {ColumnRole.y_true},
    },
    # multilabel 은 값이 다른 지표와 완전히 겹치는 넷(M1·M11·M12·M13)을 노출하지 않는다
    # (2026-09-07 결정 2). 근거는 셋이 서로 다르다:
    #   M1(subset accuracy) == M16(Exact Match Ratio) — 정의상 같은 값
    #   M11(macro) == M2·M3·M4 — multilabel 의 M2~M4 가 이미 macro 평균이다(SPEC §3 규칙 5)
    #   M12(micro)·M13(weighted) == M22(classification_report)의 micro avg·weighted avg 행
    # 같은 수를 두 이름으로 인쇄하면 독자는 서로 다른 측정이라고 읽는다.
    # multiclass 에서는 넷 다 유지한다 — 거기서는 M2~M4 가 macro 가 아니라 값이 다르다.
    TaskType.multilabel: {
        "M2":  {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M3":  {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M4":  {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M5":  {ColumnRole.true_labels, ColumnRole.pred_labels},
        # M6 는 multilabel 미지원 — common.calculate_kl_divergence 는 1-D 라벨만 처리한다
        # (이진화된 2-D 배열에 pd.Series() 를 호출해 ValueError). 프론트도 노출하지 않는다.
        "M15": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M16": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M17": {ColumnRole.true_labels, ColumnRole.pred_labels},
        # M18 은 정답/예측 라벨의 빈도 벡터 간 코사인 거리다(multilabel.calculate_distribution_diff_ml).
        # 확률(score_per_label)은 읽지 않는다.
        "M18": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M21": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M22": {ColumnRole.true_labels, ColumnRole.pred_labels},
        # M23 은 정답 분포만으로 계산된다(common.calculate_imbalance_ratio) — 예측 컬럼 불필요.
        "M23": {ColumnRole.true_labels},
    },
}


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
