"""tests/test_environment_and_grounding.py — 평가 환경 실측 보고 · 판정 모순 차단.

F-09 (결정 4 — 앞으로 발급되는 것만 정정) · G-05 (결정 9).
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
    # 2026-10-19: 일부 통과 예외를 넣으며 놓치면 안 되는 꼴을 더한다 — 'N개 중 N개'는 전부 통과 주장이다.
    ("CONDITIONAL_PASS", "전체 5개 지표 중 5개가 통과하였다."),
    ("FAIL", "전체 3개 중 3개 항목이 기준을 충족하였다."),
    ("INCONCLUSIVE", "모든 시험항목이 충족되었다."),
    ("CONDITIONAL_PASS", "일부는 미달이나 전체 항목이 통과하였다."),
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
    # 2026-10-19 Claude 실호출(다중, 조건부 합격)에서 오탐으로 폴백된 문장.
    ("CONDITIONAL_PASS", "전체 5개 지표 중 3개가 통과하였으며 점수는 60.0이다."),
    ("FAIL", "전체 4개 중 2개가 기준을 충족하였다."),
    ("FAIL", "정밀도가 0.72 로 낮게 나타났다."),
    ("PASS", "데이터셋의 클래스 분포는 균형에 가깝다."),
])
def test_consistent_sentences_pass(verdict, text):
    """판정과 어긋나지 않는 문장은 잡지 않는다(과잉 차단 방지)."""
    assert find_verdict_contradictions([text], verdict) == []


@pytest.mark.parametrize("text", [
    "Recall 은 목표값과의 차이가 작아 표본이 달라지면 기준에 미달할 위험이 있다.",
    "F1 Score 는 아슬아슬하게 합격해 미달 가능성이 남아 있다.",
    "표본 변동에 따라 미달 위험이 존재한다.",
    "다른 데이터에서는 기준에 미달할 가능성이 있다.",
    "운영 환경에서는 기준에 미달할 우려가 있다.",
    "표본이 바뀌면 기준에 미달할 수 있다.",
    "임계값이 설정된 모든 시험항목이 기준을 충족하여 식별된 기준 미달 위험은 없다.",
    # 세션 3 검토 — 가정·권고 꼴도 미달 주장이 아니다.
    "표본이 바뀌면 Recall 이 기준을 충족하지 못할 수도 있다.",
    "운영 데이터에서는 기준에 이르지 못할 가능성이 있다.",
    "재평가에서는 불합격으로 판정될 수 있다.",
    "표본 변동으로 판정이 미달로 바뀔 수 있다.",
    "결정 임계값을 점검해 기준 미달을 예방한다.",
    "Recall 미달 방지를 위해 소수 클래스를 보강한다.",
    "Recall 은 미달 리스크가 남아 있다.",
    # 세션 3 검토 2 — (으)로 뒤 앞일을 말하는 꼴(받침 ㄹ: 떨어질·내려갈·될)만 가정이다.
    "표본이 줄면 Recall 이 미달로 떨어질 위험이 있다.",
    "운영 데이터에서는 미달로 내려갈 가능성이 있다.",
    "재평가 시 미충족으로 분류될 우려가 있다.",
    "표본이 바뀌면 불합격으로 이어질 수도 있다.",
])
def test_pass_allows_hypothetical_shortfall_wording(text):
    """[B7, D12] 아슬아슬 합격을 8.3 에 쓰면 합격 성적서에도 '미달 위험·가능성'이 나온다.

    미달했다는 주장이 아니라 미달할 수 있다는 경고라 판정과 모순되지 않는다. 종전 정규식은 '미달'
    한 글자만 보고 이 문장들까지 잡아 맞는 서술을 폴백으로 강등했다.
    """
    assert find_verdict_contradictions([text], "PASS") == []


@pytest.mark.parametrize("text", [
    "Recall 이 기준에 미달하였다.",
    "기준 미달 항목은 Recall 이다.",
    "Recall 은 목표에 미달했다.",
    "본 모델은 불합격이다.",
    "Recall 은 목표를 미충족하였다.",
    # 같은 글 안에 경고와 진짜 미달 주장이 함께 있으면 진짜 쪽에서 걸린다.
    "Accuracy 는 미달할 위험이 있고, Recall 은 기준에 미달하였다.",
    "Recall 은 기준을 충족하지 못하였다.",
    "Recall 은 기준에 이르지 못했다.",
    "Recall 은 미달로 판정되었다.",
    "Recall 은 불합격으로 판정되어 개선 가능성이 있다.",
    # 세션 3 검토 2 — (으)로 뒤가 이미 일어난 일(되어·돼·나타나·이어져)이면 위험을 덧붙여도 미달 주장이다.
    "재현율이 미달로 판정되어 위험이 크다.",
    "Recall 은 미달로 확인되어 우려된다.",
    "Recall 은 불합격으로 이어져 리스크가 있다.",
    "Recall 은 미충족으로 분류돼 위험하다.",
    "Recall 은 미달로 나타나 위험하다.",
    # 세션 3 검토 3 — 받침이 ㄹ 이어도 능동 판단 동사(볼·판단할·평가할·간주할)는 지금 결과를 미달로 보는 주장이다.
    "재현율이 기준 미달로 볼 수 있다.",
    "재현율은 미달로 판단할 수 있다.",
    "정밀도는 불합격으로 평가할 수 있다.",
    "재현율은 미달로 간주할 수 있다.",
])
def test_pass_still_rejects_real_shortfall_claims(text):
    assert find_verdict_contradictions([text], "PASS") == [text]


def test_narrative_falls_back_when_llm_contradicts_the_verdict(make_fake_anthropic_client):
    """모순이 발견되면 그 서술을 쓰지 않고 규칙 폴백으로 강등한다."""
    from app.core import llm_budget
    from app.narrative.legacy import generate_narrative
    from tests.narrative_fixtures import minimal_narrative_request

    llm_budget.reset()
    request = minimal_narrative_request()
    request.fact_sheet.verdict = "FAIL"

    fake = make_fake_anthropic_client({
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
