"""
test_narrator.py — LLM 서술 모듈의 환각 방어선(grounding) + 규칙 폴백 단위 테스트.
LLM·API 키 없이 검증 가능한 순수 함수 대상.
"""
from schemas import (
    FactSheet, MetricFact, ConfusionFact, DistributionFact,
)
from narrator import compute_derived, build_number_whitelist, verify_grounding
from narrative_fallback import build_fallback_narrative
from benchmark_baselines import build_benchmark_refs


def _sample_fact_sheet() -> FactSheet:
    return FactSheet(
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


def _all_texts(resp):
    return [
        resp.interpretation.confusion_analysis,
        resp.interpretation.distribution_analysis,
        resp.conclusion.benchmark,
        resp.conclusion.narrative,
        resp.conclusion.risks,
        resp.recommendation_narrative.data_quality,
        resp.recommendation_narrative.model_ops,
        *[r.action for r in resp.recommendations],
        *[r.expected_impact for r in resp.recommendations],
    ]


def test_compute_derived():
    fs = _sample_fact_sheet()
    d = compute_derived(fs)
    assert d["confusion"]["total"] == 200
    assert d["confusion"]["correct"] == 175
    assert d["confusion"]["misclassified"] == 25
    assert d["confusion"]["fn"] == 15 and d["confusion"]["fp"] == 10
    assert d["distribution"]["total"] == 200
    assert d["distribution"]["percentages"]["0"] == 65.0
    assert d["counts"]["target"] == 3 and d["counts"]["passed"] == 2


def test_fallback_passes_own_grounding():
    """폴백 서술의 모든 숫자는 fact_sheet/파생값에서만 나오므로 grounding 을 통과해야 한다."""
    fs = _sample_fact_sheet()
    derived = compute_derived(fs)
    refs = build_benchmark_refs("binary", fs.metrics)
    wl = build_number_whitelist(fs, refs, derived)

    resp = build_fallback_narrative(fs, refs, derived, reason="no_key")
    grounding = verify_grounding(_all_texts(resp), wl)

    assert grounding.passed, f"폴백이 화이트리스트 밖 숫자 사용: {grounding.violations}"
    assert grounding.checked > 0, "검사된 숫자가 없음(템플릿이 비었는지 확인)"
    assert resp.conclusion.verdict == "CONDITIONAL_PASS"
    assert resp.meta.source == "fallback"


def test_grounding_catches_hallucination():
    """fact_sheet 에 없는 숫자(환각)를 주입하면 검증이 위반으로 잡아야 한다."""
    fs = _sample_fact_sheet()
    derived = compute_derived(fs)
    refs = build_benchmark_refs("binary", fs.metrics)
    wl = build_number_whitelist(fs, refs, derived)

    hallucinated = ["정확도는 무려 99.7%에 달하며 P99 지연시간은 41.2ms이다."]
    grounding = verify_grounding(hallucinated, wl)

    assert not grounding.passed
    assert any("99.7" in v for v in grounding.violations)
    assert any("41.2" in v for v in grounding.violations)


def test_exempt_tokens_not_flagged():
    """절 번호/ISO 연도 같은 면제 토큰은 위반으로 잡지 않는다."""
    fs = _sample_fact_sheet()
    derived = compute_derived(fs)
    wl = build_number_whitelist(fs, [], derived)
    g = verify_grounding(["본 시험은 ISO/IEC TS 4213:2022 기준으로 7절·8절·9절을 구성한다."], wl)
    assert g.passed, f"면제 토큰 오탐: {g.violations}"


# ── D2: compute_derived 의 FN/FP 가 positive 클래스 index 를 따라 매핑되는지 ──

def _confusion_fs(labels, matrix, positive_class=None) -> FactSheet:
    return FactSheet(
        verdict="PASS", score=100.0,
        confusion=ConfusionFact(labels=labels, matrix=matrix, positive_class=positive_class),
    )


def test_compute_derived_positive_index0():
    """positive('fraud')가 정렬상 index 0 → 하드코딩(index1)이면 FN/FP 가 반전된다."""
    fs = _confusion_fs(["fraud", "normal"], [[40, 10], [5, 45]], positive_class="fraud")
    c = compute_derived(fs)["confusion"]
    assert c["tp"] == 40 and c["fn"] == 10 and c["fp"] == 5 and c["tn"] == 45


def test_compute_derived_positive_index1_default():
    fs = _confusion_fs(["0", "1"], [[120, 10], [15, 55]], positive_class="1")
    c = compute_derived(fs)["confusion"]
    assert c["tp"] == 55 and c["fn"] == 15 and c["fp"] == 10 and c["tn"] == 120


def test_compute_derived_positive_none_fallback():
    """positive 미지정 → index 1 폴백(기존 동작 하위호환)."""
    fs = _confusion_fs(["0", "1"], [[120, 10], [15, 55]], positive_class=None)
    c = compute_derived(fs)["confusion"]
    assert c["fn"] == 15 and c["fp"] == 10


def test_compute_derived_positive_not_in_labels_fallback():
    fs = _confusion_fs(["0", "1"], [[120, 10], [15, 55]], positive_class="xyz")
    c = compute_derived(fs)["confusion"]
    assert c["fn"] == 15 and c["fp"] == 10


def test_compute_derived_positive_float_match():
    """positive_class='0.0' 는 라벨 '0'(index 0)과 float 매칭되어야 한다."""
    fs = _confusion_fs(["0", "1"], [[40, 10], [5, 45]], positive_class="0.0")
    c = compute_derived(fs)["confusion"]
    assert c["tp"] == 40 and c["fn"] == 10 and c["fp"] == 5 and c["tn"] == 45
