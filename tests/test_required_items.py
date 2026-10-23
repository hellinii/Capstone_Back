"""tests/test_required_items.py — 필수 서술 항목 목록(서술 설계 5절)과 select_items(결론 목록, D7)."""
import json
from collections import Counter
from pathlib import Path

import pytest

from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.grounding import build_number_whitelist, verify_grounding
from app.narrative.required_items import (
    REQUIRED_ITEMS,
    conclusions_by_section,
    select_items,
)
from app.narrative.schemas import FactSheet, LatencyFact, MetricFact, NarrativeFactsFact, NarrativeRequest
from app.narrative.sections import SECTION_IDS

CONTRACT = Path(__file__).parent / "contract"


def _req(name: str) -> NarrativeRequest:
    return NarrativeRequest.model_validate(json.loads((CONTRACT / f"fact_sheet_{name}.json").read_text(encoding="utf-8")))


def _ids(req: NarrativeRequest) -> list[str]:
    return [c.item_id for c in select_items(req.fact_sheet, req.client_inputs, req.task_type.value, edition=3)]


# ── 표 ─────────────────────────────────────────────────────────────────────────

def test_table_has_the_41_rows_of_section_5():
    assert len(REQUIRED_ITEMS) == 41
    assert Counter(i.section for i in REQUIRED_ITEMS) == {
        "7.1": 12, "7.2": 4, "7.3": 1, "8.1": 3, "8.2": 13, "8.3": 5, "9": 3,
    }
    assert [i.section for i in REQUIRED_ITEMS] == sorted((i.section for i in REQUIRED_ITEMS), key=SECTION_IDS.index)
    assert {i.author for i in REQUIRED_ITEMS} == {"ai", "ai_fact", "fixed_interpret", "fixed", "form"}


def test_plan_decisions_on_authors():
    by_id = {i.id: i for i in REQUIRED_ITEMS}
    # 7.1 오류 성격·8.3 근소 = 고정 문구 17·18 + 해석, 8.2 평균 방식·지표 선택 기준 = 양식, 해석형 = AI(템플릿은 사실)
    assert (by_id["7.1-11"].author, by_id["8.3-02"].author) == ("fixed_interpret", "fixed_interpret")
    assert (by_id["8.2-10"].author, by_id["8.2-12"].author) == ("form", "form")
    assert (by_id["8.2-13"].author, by_id["8.3-01"].author) == ("ai_fact", "ai_fact")
    # D24 확인은 세 항목뿐
    assert {i.id for i in REQUIRED_ITEMS if i.d24 is not None} == {"7.1-03", "8.2-11", "8.3-05"}


# ── 계약 예시로 고르기 ────────────────────────────────────────────────────────────

EXPECTED = {
    "binary": [
        "7.1-01", "7.1-02", "7.1-03", "7.1-07", "7.1-10", "7.1-11",
        "7.2-01", "7.2-02", "7.2-03", "7.3-01",
        "8.1-01", "8.1-02", "8.1-03",
        "8.2-01", "8.2-02", "8.2-03", "8.2-04", "8.2-05", "8.2-06",
        "8.3-01", "8.3-02", "8.3-03",
        "9-01", "9-02",
    ],
    "multiclass": [
        "7.1-01", "7.1-04", "7.1-06", "7.1-08", "7.1-10", "7.1-11",
        "7.2-01", "7.2-02", "7.2-03", "7.3-01",
        "8.1-01", "8.1-02",
        "8.2-01", "8.2-02", "8.2-03", "8.2-04", "8.2-10", "8.2-11",
        "8.3-01",
        "9-01", "9-02",
    ],
    "multilabel": [
        "7.1-01", "7.1-05", "7.1-09", "7.1-10", "7.1-11",
        "7.2-01", "7.2-02", "7.3-01",
        "8.1-01", "8.1-02", "8.1-03",
        "8.2-01", "8.2-02", "8.2-03", "8.2-04", "8.2-09", "8.2-11",
        "8.3-01",
        "9-01", "9-02",
    ],
}


@pytest.mark.parametrize("name", EXPECTED)
def test_contract_examples_select_these_items(name):
    assert _ids(_req(name)) == EXPECTED[name]


@pytest.mark.parametrize("name", [*EXPECTED, "legacy_main"])
def test_conclusion_numbers_are_grounded(name):
    """AI·템플릿이 쓸 결론(ai·ai_fact)의 숫자는 모두 허용 목록 안 — 결론을 그대로 써도 숫자 검문에 걸리지 않는다."""
    req = _req(name)
    fs, task = req.fact_sheet, req.task_type.value
    derived = compute_derived(fs, task_type=task)
    wl = build_number_whitelist(fs, build_benchmark_refs(task, fs.metrics, edition=3), derived, req.client_inputs, edition=3)
    texts = [
        c.text for c in select_items(fs, req.client_inputs, task, edition=3)
        if c.author in ("ai", "ai_fact") and not c.open_ended
    ]
    assert texts
    result = verify_grounding(texts, wl)
    assert result.passed, result.violations


def test_legacy_request_without_facts_still_selects():
    ids = _ids(_req("legacy_main"))
    assert "7.1-01" in ids and "8.2-02" in ids
    assert "7.1-10" not in ids  # 최약 클래스는 facts 가 있어야(옛 요청은 계산하지 않는다)


def test_d24_values_are_attached_only_to_d24_items():
    concl = select_items(_req("binary").fact_sheet, None, "binary", edition=3)
    with_values = {c.item_id: c.d24_values for c in concl if c.d24_values}
    assert with_values == {"7.1-03": (43.0, 0.434343434)}


def test_conclusions_grouped_by_the_seven_sections():
    grouped = conclusions_by_section(select_items(_req("multiclass").fact_sheet, None, "multiclass", edition=3))
    assert list(grouped) == list(SECTION_IDS)
    assert [c.item_id for c in grouped["7.3"]] == ["7.3-01"] and grouped["7.3"][0].open_ended


# ── 조건 갈래 ───────────────────────────────────────────────────────────────────

def _fs(**over) -> FactSheet:
    base = dict(verdict="PASS", score=100.0, n_samples=100)
    base.update(over)
    return FactSheet(**base)


def _m(**over) -> MetricFact:
    base = dict(metric_id="M1", display_name="Accuracy", value=0.9, threshold=0.8, status="pass", margin=0.1)
    base.update(over)
    return MetricFact(**base)


def _select(fs: FactSheet, task="binary", client=None) -> dict[str, str]:
    return {c.item_id: c.text for c in select_items(fs, client, task, edition=3)}


def test_d24_items_switch_off_for_similar_equal_batch_or_unknown():
    risk = {"side": "fn", "count": 3, "total_errors": 5, "share": 0.6, "share_text": "0.6"}
    on = _fs(facts=NarrativeFactsFact(declarations={"risk_direction": "fn"}, declared_risk_errors=risk))
    assert "7.1-03" in _select(on)
    off = _fs(facts=NarrativeFactsFact(declarations={"risk_direction": "similar"}))
    assert "7.1-03" not in _select(off)
    lat = LatencyFact(available=True, mean=10.0, p50=9.0, p95=20.0, p99=30.0)
    realtime = _fs(latency=lat, facts=NarrativeFactsFact(declarations={"usage_mode": "realtime"}))
    picked = {c.item_id: c for c in select_items(realtime, None, "binary", edition=3)}
    assert picked["8.3-05"].d24_values == (10.0, 9.0, 20.0, 30.0)
    batch = _fs(latency=lat, facts=NarrativeFactsFact(declarations={"usage_mode": "batch"}))
    assert "8.3-05" not in _select(batch) and "8.3-04" in _select(batch)
    no_latency = _fs(facts=NarrativeFactsFact(declarations={"usage_mode": "realtime"}))
    assert "8.3-05" not in _select(no_latency)


def test_average_gap_needs_m22_but_weakest_class_does_not():
    gap = {
        "kind": "macro_vs_weighted", "macro_f1": 0.5, "other_f1": 0.6, "difference": 0.1, "difference_text": "0.1",
        "reference_class": {"label": "a", "support": 50, "f1": 0.7},
    }
    weak = {
        "label": "c", "f1": 0.3, "support": 40, "share": 0.4, "share_text": "0.4", "macro_f1": 0.5,
        "gap_from_macro": 0.2, "gap_text": "0.2", "low_support": False, "far_below_macro": True,
    }
    without = _fs(facts=NarrativeFactsFact(selected_metrics=["M1"], average_gap=gap, weakest_class=weak))
    picked = _select(without, "multiclass")
    assert "8.2-07" not in picked and "7.1-10" in picked and "8.3-01" in picked
    with_m22 = _fs(facts=NarrativeFactsFact(selected_metrics=["M1", "M22"], average_gap=gap, weakest_class=weak))
    assert "8.2-07" in _select(with_m22, "multiclass")


def test_precision_recall_needs_m2_and_m3():
    gap = {"precision": 0.9, "recall": 0.7, "difference": 0.2, "difference_text": "0.2", "flagged": True}
    assert "8.2-05" not in _select(_fs(facts=NarrativeFactsFact(selected_metrics=["M2"], precision_recall_gap=gap)))
    picked = _select(_fs(facts=NarrativeFactsFact(selected_metrics=["M2", "M3"], precision_recall_gap=gap)))
    # S6: 서술체 — 'Precision 쪽이 … 높고, … 치우쳐 있다'(종전 'Precision 이 높다, … 치우침이 있다' 는 쉼표로 끊긴 개조식).
    assert "Precision 쪽이 0.2 높고" in picked["8.2-05"] and "한쪽으로 치우쳐 있다" in picked["8.2-05"]


def test_ci_has_two_branches_and_a_skipped_reason():
    inside = _fs(metrics=[_m(ci_low=0.75, ci_high=0.95, ci_contains_target=True)])
    assert "포함하는 지표" in _select(inside)["8.2-03"]
    apart = _fs(metrics=[_m(ci_low=0.85, ci_high=0.95, ci_contains_target=False)])
    assert "모두 목표값과 떨어져" in _select(apart)["8.2-03"]
    skipped = _fs(
        metrics=[_m()],
        facts=NarrativeFactsFact(ci_skipped=[{"metric_id": "M1", "display_name": "Accuracy", "reason": "점추정 불일치"}]),
    )
    assert "신뢰구간은 산출하지 않았다(생략한 지표와 사유: Accuracy(점추정 불일치))" in _select(skipped)["8.2-03"]
    assert "8.2-03" not in _select(_fs(metrics=[_m(threshold=None, margin=None)]))


def test_verdict_reason_lists_d19_causes_and_score_as_reference():
    facts = NarrativeFactsFact(verdict_detail={
        "verdict": "INCONCLUSIVE", "judged_count": 2, "passed_count": 2,
        "inconclusive_cause": "core_not_computed",
        "inconclusive_metrics": [{"metric_id": "M4", "display_name": "F1 Score"}],
        "unavailable_core": [{"metric_id": "M4", "display_name": "F1 Score"}],
    })
    text = _select(_fs(verdict="INCONCLUSIVE", facts=facts))["8.2-02"]
    assert "'판정 불가'" in text and "핵심 지표 산출 실패(F1 Score)" in text
    assert "통과율은 판정 근거가 아니다" in text
    assert text.count("F1 Score") == 1  # 산출 실패를 두 번 말하지 않는다


def test_no_problem_and_recommendation_are_exclusive():
    none = _select(_fs(facts=NarrativeFactsFact(fixed_sentences=[{"id": "6", "section": "9", "text": "문제 없음."}])))
    assert none.get("9-03") == "문제 없음." and "9-02" not in none
