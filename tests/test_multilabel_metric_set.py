"""tests/test_multilabel_metric_set.py — multilabel 중복 지표 제거.

ISSUES.md A-04 (2026-09-07 ★확정된 제품 결정 2).

multilabel 에서 값이 다른 지표와 완전히 겹치는 넷을 뺀다. 근거는 **셋이 서로 다르다**:

  - **M1**(subset accuracy) == **M16**(Exact Match Ratio) — 정의상 같은 값.
    저장소 골든에도 둘 다 0.065 로 인쇄돼 있었다.
  - **M11**(macro average) == (**M2**, **M3**, **M4**) — 실측 16자리 일치.
    multilabel 의 M2~M4 가 이미 macro 평균이기 때문이다(SPEC §3 규칙 5).
  - **M12**(micro) · **M13**(weighted) == **M22**(classification_report)의
    `micro avg` · `weighted avg` 행 — 실측 16자리 일치. M22 는 multilabel 추천 지표라
    기본 경로에서 항상 켜진다.

같은 수를 두 이름으로 인쇄하면 독자는 서로 다른 측정이라고 읽는다.
multiclass 에서는 넷 다 유지한다 — 거기서는 M2~M4 가 macro 가 아니라 값이 다르다.
"""
import pytest

from app.core.schemas import METRIC_REQUIREMENTS, TaskType
from app.evaluation.engine import VALID_METRICS_BY_TASK

_REMOVED = ["M1", "M11", "M12", "M13"]


@pytest.mark.parametrize("metric_id", _REMOVED)
def test_removed_from_multilabel_requirements(metric_id):
    assert metric_id not in METRIC_REQUIREMENTS[TaskType.multilabel]


@pytest.mark.parametrize("metric_id", _REMOVED)
def test_still_supported_in_multiclass(metric_id):
    """multiclass 에서는 값이 겹치지 않으므로 유지한다(과잉 제거 방지)."""
    assert metric_id in METRIC_REQUIREMENTS[TaskType.multiclass]


@pytest.mark.parametrize("metric_id", _REMOVED)
def test_engine_rejects_removed_metric_for_multilabel(metric_id):
    """엔진의 허용 목록은 요구표에서 파생되므로 함께 닫혀야 한다."""
    assert metric_id not in VALID_METRICS_BY_TASK["multilabel"]


def test_replacement_metrics_remain():
    """뺀 자리를 대신하는 지표는 남아 있어야 한다 — 정보가 사라지면 안 된다."""
    multilabel = METRIC_REQUIREMENTS[TaskType.multilabel]
    for metric_id in ["M16", "M2", "M3", "M4", "M22"]:
        assert metric_id in multilabel, f"{metric_id} 가 없으면 제거한 지표의 정보가 사라진다"


def test_multilabel_metric_set_matches_spec():
    """SPEC §3 이 정본이다 — 허용 지표 목록이 그와 일치한다."""
    assert set(METRIC_REQUIREMENTS[TaskType.multilabel]) == {
        "M2", "M3", "M4", "M5", "M15", "M16", "M17", "M18", "M21", "M22", "M23",
    }
