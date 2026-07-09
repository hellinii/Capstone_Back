"""tests/test_narrative_router.py — POST /api/generate-narrative 골든 characterization 테스트.

narrator.py 는 함수 단위 테스트(test_narrator.py)는 있으나 엔드포인트(HTTP) 테스트가 없다.
PR-F(grounding.py/derived.py 분리, generate_narrative 얇게)의 안전망으로, 라우터 배선 +
규칙 기반 폴백 서술의 현재 출력을 골든으로 고정한다.

lifespan 을 띄우지 않으므로 app.state.openai_client 가 없어(→ None) 결정론적 폴백 경로를 탄다.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.core.schemas import TaskType
from app.narrative.schemas import ConfusionFact, DistributionFact, FactSheet, MetricFact, NarrativeRequest
from app.main import app
from golden_utils import assert_golden

client = TestClient(app)  # lifespan 미실행 → openai_client 없음 → 폴백


@pytest.fixture(autouse=True)
def _force_fallback():
    """app 은 싱글턴이라 다른 테스트가 app.state.openai_client 에 mock 을 남길 수 있다.
    폴백 경로를 결정론적으로 태우기 위해 명시적으로 None 을 강제하고 원복한다."""
    prev = getattr(app.state, "openai_client", None)
    app.state.openai_client = None
    try:
        yield
    finally:
        app.state.openai_client = prev


def _sample_request() -> NarrativeRequest:
    fs = FactSheet(
        n_samples=200,
        dropped_rows=3,
        verdict="CONDITIONAL_PASS",
        score=66.7,
        metrics=[
            MetricFact(tc_id="M1", display_name="Accuracy", value=0.94, threshold=0.85, status="pass"),
            MetricFact(tc_id="M3", display_name="Recall", value=0.70, threshold=0.80, status="fail"),
            MetricFact(tc_id="M9", display_name="AUROC", value=0.88, threshold=0.80, status="pass"),
        ],
        confusion=ConfusionFact(labels=["0", "1"], matrix=[[120, 10], [15, 55]]),
        distribution=DistributionFact(
            class_distribution={"0": 130, "1": 70}, imbalance_ratio=1.857
        ),
    )
    return NarrativeRequest(task_type=TaskType.binary, report_purpose="internal", fact_sheet=fs)


def test_generate_narrative_fallback_golden():
    req = _sample_request()
    resp = client.post("/api/generate-narrative", json=json.loads(req.model_dump_json()))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 폴백 경로 확인(메타 source) — 키 없음/None client
    assert body.get("meta", {}).get("source") == "fallback"
    assert_golden("narrative_binary_fallback", body)
