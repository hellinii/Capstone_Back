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
    TaskType.multiclass: [
        ColumnRole.sample_id, ColumnRole.y_true, ColumnRole.y_pred,
        ColumnRole.prob_per_class, ColumnRole.latency, ColumnRole.ignore,
    ],
    TaskType.multilabel: [
        ColumnRole.sample_id, ColumnRole.true_labels, ColumnRole.pred_labels,
        ColumnRole.score_per_label, ColumnRole.latency, ColumnRole.ignore,
    ],
}


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
        "M23": {ColumnRole.y_true, ColumnRole.y_pred},
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
        "M23": {ColumnRole.y_true, ColumnRole.y_pred},
    },
    TaskType.multilabel: {
        # M1(Accuracy)은 multilabel 에서 sklearn subset accuracy(전 라벨 일치 비율)로
        # 계산된다 — 정의상 M16(Exact Match Ratio)과 동일 값.
        "M1":  {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M2":  {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M3":  {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M4":  {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M5":  {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M6":  {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M11": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M12": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M13": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M15": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M16": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M17": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M18": {ColumnRole.true_labels, ColumnRole.score_per_label},  # 분포 기반
        "M21": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M22": {ColumnRole.true_labels, ColumnRole.pred_labels},
        "M23": {ColumnRole.true_labels, ColumnRole.pred_labels},
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
