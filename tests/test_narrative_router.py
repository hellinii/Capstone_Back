"""tests/test_narrative_router.py — POST /api/generate-narrative 골든 characterization 테스트.

narrator.py 는 함수 단위 테스트(test_narrator.py)는 있으나 엔드포인트(HTTP) 테스트가 없다.
PR-F(grounding.py/derived.py 분리, generate_narrative 얇게)의 안전망으로, 라우터 배선 +
규칙 기반 폴백 서술의 현재 출력을 골든으로 고정한다.

lifespan 을 띄우지 않으므로 app.state.anthropic_client 가 없어(→ None) 결정론적 폴백 경로를 탄다.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.core.schemas import TaskType
from app.narrative.schemas import ConfusionFact, DistributionFact, FactSheet, MetricFact, NarrativeRequest
from app.main import app
from golden_utils import assert_golden

client = TestClient(app)  # lifespan 미실행 → anthropic_client 없음 → 폴백


@pytest.fixture(autouse=True)
def _force_fallback():
    """app 은 싱글턴이라 다른 테스트가 app.state.anthropic_client 에 가짜를 남길 수 있다.
    폴백 경로를 결정론적으로 태우기 위해 명시적으로 None 을 강제하고 원복한다."""
    prev = getattr(app.state, "anthropic_client", None)
    app.state.anthropic_client = None
    try:
        yield
    finally:
        app.state.anthropic_client = prev


def _sample_request() -> NarrativeRequest:
    fs = FactSheet(
        n_samples=200,
        dropped_rows=3,
        verdict="CONDITIONAL_PASS",
        score=66.7,
        metrics=[
            MetricFact(metric_id="M1", display_name="Accuracy", value=0.94, threshold=0.85, status="pass"),
            MetricFact(metric_id="M3", display_name="Recall", value=0.70, threshold=0.80, status="fail"),
            MetricFact(metric_id="M9", display_name="AUROC", value=0.88, threshold=0.80, status="pass"),
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


# ── 9절 문제 목록(서술 설계 D22) ─────────────────────────────────────────────────

def _problems_request() -> NarrativeRequest:
    """핵심 지표 미달 + 임계값 부적합 + 불균형 영향(편향 노력 '수행 안 함') — 프론트 problemList 가 낼 모양."""
    fs = FactSheet(
        n_samples=200,
        dropped_rows=3,
        verdict="FAIL",
        score=66.7,
        metrics=[
            MetricFact(metric_id="M1", display_name="Accuracy", value=0.94, threshold=0.85, status="pass"),
            MetricFact(metric_id="M3", display_name="Recall", value=0.70, threshold=0.80, status="fail",
                       is_core=True, margin=-0.1),
            MetricFact(metric_id="M9", display_name="AUROC", value=0.88, threshold=0.80, status="pass"),
        ],
        confusion=ConfusionFact(labels=["0", "1"], matrix=[[120, 10], [15, 55]]),
        distribution=DistributionFact(class_distribution={"0": 130, "1": 70}, imbalance_ratio=1.857),
        f1_threshold_gain={"max_f1": 0.8421, "current_f1": 0.7012, "gain": 0.1409, "best_threshold": 0.37},
        imbalance_effect={
            "level": "affected", "imbalance_ratio": 1.857, "accuracy_minus_macro_f1": 0.12,
            "majority_minus_minority_f1": 0.18, "majority_class": "0", "minority_class": "1",
        },
        problems=[
            {"id": "P1", "kind": "core_fail", "priority": "HIGH", "title": "핵심 지표 미달 (Recall)",
             "source_section": "8.2", "recommend_section": "9.2",
             "evidence": {"Recall": {"value": 0.7, "target": 0.8}}, "note": None},
            {"id": "P2", "kind": "threshold_tuning", "priority": "MEDIUM", "title": "임계값 부적합",
             "source_section": "8.3", "recommend_section": "9.2",
             "evidence": {"max_f1": 0.8421, "current_f1": 0.7012, "gain": 0.1409, "best_threshold": 0.37},
             "note": None},
            {"id": "P3", "kind": "imbalance_effect", "priority": "MEDIUM", "title": "불균형 영향 있음 (다수 0 · 소수 1)",
             "source_section": "7.2", "recommend_section": "9.1",
             "evidence": {"level": "affected", "imbalance_ratio": 1.857, "accuracy_minus_macro_f1": 0.12,
                          "majority_minus_minority_f1": 0.18, "majority_class": "0", "minority_class": "1"},
             "note": "의뢰자가 편향 감소 노력을 '수행 안 함'으로 밝혔다."},
        ],
    )
    return NarrativeRequest(task_type=TaskType.binary, report_purpose="external", fact_sheet=fs)


def test_generate_narrative_fallback_problems_golden():
    req = _problems_request()
    resp = client.post("/api/generate-narrative", json=json.loads(req.model_dump_json()))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["meta"]["source"] == "fallback"
    assert [r["problem_id"] for r in body["recommendations"]] == ["P1", "P2", "P3"]
    assert_golden("narrative_binary_fallback_problems", body)


# 옛 프론트의 서술 호출부가 읽는 응답 필드 전부(경로 → 허용 타입). 이름이 바뀌면 프론트가 말없이 빈 칸을 찍는다.
_TOP_FIELDS = {
    ("interpretation", "confusion_analysis"): str,
    ("interpretation", "distribution_analysis"): str,
    ("conclusion", "benchmark"): str,
    ("conclusion", "narrative"): str,
    ("conclusion", "risks"): str,
    ("recommendation_narrative", "data_quality"): str,
    ("recommendation_narrative", "model_ops"): str,
    ("meta", "source"): str,
}
_RECOMMENDATION_FIELDS = {
    "problem_id": str, "priority": str, "category": str, "action": str, "expected_impact": str,
}
_BENCHMARK_FIELDS = {
    "metric": str, "model_value": (int, float), "ref_low": (int, float), "ref_high": (int, float),
    "quality": str, "direction": str, "source_note": str,
    "target_value": (int, float, type(None)), "target_position": (str, type(None)),
}


def _assert_response_contract(body: dict, problem_ids: list[str]) -> None:
    for (outer, inner), typ in _TOP_FIELDS.items():
        assert isinstance(body[outer][inner], typ), (outer, inner)
    assert body["meta"]["source"] in ("llm", "fallback")
    assert isinstance(body["recommendations"], list)
    for rec in body["recommendations"]:
        assert set(_RECOMMENDATION_FIELDS) <= set(rec), rec
        for key, typ in _RECOMMENDATION_FIELDS.items():
            assert isinstance(rec[key], typ), key
        assert rec["priority"] in ("HIGH", "MEDIUM", "LOW")
        assert rec["problem_id"] in problem_ids
    # 문제마다 권고가 하나 이상, 문제 번호 순서
    seen = list(dict.fromkeys(r["problem_id"] for r in body["recommendations"]))
    assert seen == problem_ids
    assert body["benchmark_refs"], "이 요청은 벤치마크 참조가 있는 지표를 고른다"
    for ref in body["benchmark_refs"]:
        assert set(_BENCHMARK_FIELDS) <= set(ref), ref
        for key, typ in _BENCHMARK_FIELDS.items():
            assert isinstance(ref[key], typ), key
        assert ref["target_position"] in ("below", "within", "above", None)


def test_response_field_contract_fallback():
    body = client.post(
        "/api/generate-narrative", json=json.loads(_problems_request().model_dump_json())
    ).json()
    _assert_response_contract(body, ["P1", "P2", "P3"])


def test_response_field_contract_llm(make_fake_anthropic_client):
    llm_json = {
        "interpretation": {"confusion_analysis": "오분류는 25건이다.", "distribution_analysis": ""},
        "conclusion": {"verdict": "FAIL", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": [
            {"problem_id": "P2", "priority": "LOW", "category": "임계값", "action": "임계값을 다시 고른다",
             "expected_impact": "F1 개선"},
        ],
    }
    app.state.anthropic_client = make_fake_anthropic_client(llm_json)  # autouse 픽스처가 원복한다
    body = client.post(
        "/api/generate-narrative", json=json.loads(_problems_request().model_dump_json())
    ).json()
    assert body["meta"]["source"] == "llm", body["meta"]
    _assert_response_contract(body, ["P1", "P2", "P3"])
    assert [r["priority"] for r in body["recommendations"]] == ["HIGH", "MEDIUM", "MEDIUM"]


def test_response_without_problems_has_no_recommendations():
    body = client.post("/api/generate-narrative", json=json.loads(_sample_request().model_dump_json())).json()
    assert body["recommendations"] == []
