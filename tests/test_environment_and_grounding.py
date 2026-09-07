"""tests/test_environment_and_grounding.py — 평가 환경 실측 보고 · 판정 모순 차단.

ISSUES.md F-09 (결정 4 — 앞으로 발급되는 것만 정정) · G-05 (결정 9).
"""
import pytest
from fastapi.testclient import TestClient

from app.core.schemas import ColumnMapping, ColumnRole, DataMetadata, TaskType
from app.evaluation.schemas import EvaluateRequest
from app.main import app
from app.narrative.grounding import find_verdict_contradictions

client = TestClient(app)

_CSV = "y,p\n1,1\n1,0\n0,0\n0,1\n"


def _evaluate():
    req = EvaluateRequest(
        task_type=TaskType.binary,
        column_mappings=[
            ColumnMapping(column="y", role=ColumnRole.y_true),
            ColumnMapping(column="p", role=ColumnRole.y_pred),
        ],
        selected_metric_ids=["M1"],
        metadata=DataMetadata(positive_class="1", negative_class="0"),
    )
    return client.post(
        "/api/evaluate",
        files={"file": ("d.csv", _CSV.encode("utf-8"), "text/csv")},
        data={"data": req.model_dump_json()},
    )


# ── F-09: 평가 수행 환경을 실측으로 보고한다 ──────────────────────────────

def test_response_reports_actual_library_versions():
    """성적서 4절의 '평가 도구'가 실제로 계산에 쓰인 버전이어야 한다.

    종전에는 프론트 상수에 "scikit-learn 1.4.0" 등이 박혀 있었고, 백엔드가 실제로
    무엇을 쓰는지와 무관했다. 버전을 아는 곳은 계산을 수행한 프로세스뿐이다.
    """
    import sklearn

    env = _evaluate().json()["environment"]
    assert env["libraries"]["scikit-learn"] == sklearn.__version__
    assert env["libraries"]["python"].startswith(".".join(map(str, __import__("sys").version_info[:2])))


def test_response_reports_evaluation_timestamp():
    """평가 일시도 하드코딩이 아니라 실제 수행 시각이어야 한다(KST)."""
    env = _evaluate().json()["environment"]
    assert env["evaluated_at"]
    assert "+09:00" in env["evaluated_at"] or env["evaluated_at"].endswith("+0900")


def test_environment_lists_every_library_that_computes_metrics():
    env = _evaluate().json()["environment"]
    assert set(env["libraries"]) >= {"python", "scikit-learn", "pandas", "numpy"}


# ── G-05: 판정과 모순되는 서술을 잡는다 ───────────────────────────────────

@pytest.mark.parametrize("verdict, text", [
    ("FAIL", "모든 시험항목이 합격 기준을 충족하였다."),
    ("FAIL", "본 모델은 목표 성능을 달성하였다."),
    ("PASS", "일부 지표가 합격 기준에 미달하였다."),
    ("PASS", "본 모델은 기준을 충족하지 못하였다."),
])
def test_verdict_contradicting_sentences_are_detected(verdict, text):
    """[G-05] 숫자가 없는 정성 서술은 종전에 **무조건 통과**했다.

    grounding 은 숫자 토큰만 대조하므로 "모든 항목이 합격했다"처럼 수치가 없는 문장은
    검증 대상이 아니었다. 판정이 FAIL 인 성적서에 그 문장이 실리면 독자는 정반대
    결론을 읽는다 — 숫자가 없다는 이유로 가장 위험한 문장이 무검증으로 통과했다.
    """
    assert find_verdict_contradictions([text], verdict)


@pytest.mark.parametrize("verdict, text", [
    ("PASS", "모든 시험항목이 합격 기준을 충족하였다."),
    ("FAIL", "일부 지표가 합격 기준에 미달하였다."),
    ("CONDITIONAL_PASS", "핵심 지표는 충족하였으나 일부 항목이 미달하였다."),
    ("FAIL", "정밀도가 0.72 로 낮게 나타났다."),
    ("PASS", "데이터셋의 클래스 분포는 균형에 가깝다."),
])
def test_consistent_sentences_pass(verdict, text):
    """판정과 어긋나지 않는 문장은 잡지 않는다(과잉 차단 방지)."""
    assert find_verdict_contradictions([text], verdict) == []


def test_narrative_falls_back_when_llm_contradicts_the_verdict(make_fake_openai_client):
    """모순이 발견되면 그 서술을 쓰지 않고 규칙 폴백으로 강등한다."""
    from app.core import llm_budget
    from app.narrative.service import generate_narrative
    from tests.narrative_fixtures import minimal_narrative_request

    llm_budget.reset()
    request = minimal_narrative_request()
    request.fact_sheet.verdict = "FAIL"

    fake = make_fake_openai_client({
        "interpretation": {"confusionAnalysis": "모든 시험항목이 합격 기준을 충족하였다.",
                           "distributionAnalysis": ""},
        "conclusion": {"benchmark": "", "narrative": "", "risks": ""},
        "recommendationNarrative": {"dataQuality": "", "modelOps": ""},
        "recommendations": [],
    })
    result = await_sync(generate_narrative(fake, request))

    assert result.meta.source == "fallback"
    assert result.meta.reason == "verdict_contradiction"
    llm_budget.reset()


def await_sync(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)
