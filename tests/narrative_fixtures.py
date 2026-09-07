"""tests/narrative_fixtures.py — 서술 생성 테스트용 최소 요청(수집 대상 아님)."""
from app.core.schemas import ReportPurpose, TaskType
from app.narrative.schemas import FactSheet, MetricFact, NarrativeRequest


def minimal_narrative_request() -> NarrativeRequest:
    return NarrativeRequest(
        task_type=TaskType.binary,
        report_purpose=ReportPurpose.internal,
        fact_sheet=FactSheet(
            metrics=[MetricFact(metric_id="M1", display_name="Accuracy", value=0.9, threshold=0.8, status="pass")],
            n_samples=100,
            verdict="PASS",
            score=100.0,
        ),
    )
