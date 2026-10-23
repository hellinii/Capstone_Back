"""tests/test_s4b_review.py — S4b 독립 검토(wf_3c000d40-7a1)·변이에서 나온 구멍의 회귀 테스트."""
import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.narrative import jobs
from app.narrative.derived import compute_derived
from app.narrative.fallback import build_fallback_narrative
from app.narrative.grounding import build_number_whitelist, verify_grounding
from app.narrative.required_items import select_items
from app.narrative.schemas import (
    DistributionFact,
    F1ThresholdGainFact,
    FactSheet,
    ImbalanceEffectFact,
    MetricFact,
    NarrativeFactsFact,
    NarrativeRequest,
)

CONTRACT = Path(__file__).parent / "contract"


def _example(name: str = "multiclass") -> dict:
    return json.loads((CONTRACT / f"fact_sheet_{name}.json").read_text(encoding="utf-8"))


def _long_shift(raw: dict, n: int = 250) -> dict:
    """facts 를 3만 자 넘게 — 분포 차 목록을 길게(목록 상한 256 안)."""
    raw["fact_sheet"]["facts"]["distribution_shift"] = [
        {"label": f"class_{i:04d}_{'x' * 40}", "training_ratio": 0.1, "evaluation_ratio": 0.1,
         "difference": 0.0, "difference_text": "0", "flagged": False}
        for i in range(n)
    ]
    return raw


# ── 작업 경로: 숨은 형식 오류·omitted·표시 문자열 ───────────────────────────────────

@pytest.mark.parametrize("break_it", [
    lambda f: f["declarations"].__setitem__("usage_mode", "weird"),
    lambda f: f.__setitem__("prediction_bias", {"unit": "bogus", "rows": []}),
])
def test_format_error_hidden_behind_limit_error_still_drops_facts(break_it):
    raw = _long_shift(_example())
    break_it(raw["fact_sheet"]["facts"])
    req, trimmed = jobs.parse_request(raw)
    assert jobs.FACTS_DROPPED in trimmed
    assert req.fact_sheet.facts is None or req.fact_sheet.facts.verdict_detail is None


def test_fixed_sentences_over_limit_with_bad_item_is_not_422():
    raw = _example()
    raw["fact_sheet"]["facts"]["fixed_sentences"] = [{"id": "x", "section": "7.1"}] + [
        {"id": str(i), "section": "7.1", "text": "문장"} for i in range(16)
    ]
    req, trimmed = jobs.parse_request(raw)
    assert trimmed  # 422 가 아니라 고쳐서 받는다


def test_omitted_with_too_many_keys_is_emptied_not_422():
    raw = _example()
    raw["fact_sheet"]["facts"]["omitted"] = {f"k{i}": 1 for i in range(40)}
    req, trimmed = jobs.parse_request(raw)
    assert "fact_sheet.facts.omitted" in trimmed and req.fact_sheet.facts.omitted == {}


def test_dropping_invalid_facts_keeps_omitted():
    raw = _example()
    raw["fact_sheet"]["facts"]["omitted"] = {"per_class": 44, "class_distribution": 44}
    raw["fact_sheet"]["facts"]["declarations"]["risk_direction"] = "maybe"
    req, trimmed = jobs.parse_request(raw)
    assert trimmed == [jobs.FACTS_DROPPED]
    assert req.fact_sheet.facts is not None
    assert req.fact_sheet.facts.omitted == {"per_class": 44, "class_distribution": 44}
    assert req.fact_sheet.facts.weakest_class is None and req.fact_sheet.facts.verdict_detail is None


def test_display_text_that_is_not_a_number_is_dropped_in_jobs_and_422_on_legacy():
    raw = _example()
    raw["fact_sheet"]["imbalance_effect"]["display"]["majority_minus_minority_f1"] = "n/a"
    raw["fact_sheet"]["metrics"][0]["value_text"] = "약 0.3"
    req, trimmed = jobs.parse_request(raw)
    assert trimmed == [jobs.DISPLAY_DROPPED]
    assert req.fact_sheet.imbalance_effect.display is None and req.fact_sheet.metrics[0].value_text is None
    with pytest.raises(ValidationError):
        NarrativeRequest.model_validate(raw)
    # 옛 경로는 422(500 이 아니다)
    resp = TestClient(app).post("/api/generate-narrative", json=raw)
    assert resp.status_code == 422


def test_display_drop_does_not_force_rule_sentences(monkeypatch):
    raw = _example("binary")
    raw["fact_sheet"]["metrics"][0]["margin_text"] = "-"
    _, trimmed = jobs.parse_request(raw)
    assert all(t in jobs.SOFT_REPAIRS for t in trimmed)


# ── 허용 목록 ──────────────────────────────────────────────────────────────────

def _fs(**over) -> FactSheet:
    # 바탕값은 다른 칸에서 우연히 '0'·'100' 이 나오지 않게 고른다(제외 행 0·통과율 100 은 그 자체로 허용된다 —
    # 그래서 '차이 0' 같은 모순은 숫자 검문만으로 다 막을 수 없다. 판정 모순 검사·S5 칸 검사 몫).
    base = dict(verdict="PASS", score=33.3, n_samples=7, dropped_rows=3)
    base.update(over)
    return FactSheet(**base)


def _passes(fs: FactSheet, text: str) -> bool:
    return verify_grounding([text], build_number_whitelist(fs, [], {}, edition=3)).passed


def test_structural_fact_fields_do_not_leak_numbers():
    facts = NarrativeFactsFact(
        selected_metrics=["M17", "M21"],
        fixed_sentences=[{"id": "17", "section": "7.1", "text": "고정 문장"}, {"id": "11a", "section": "8.2", "text": "x"}],
    )
    fs = _fs(facts=facts)
    for text in ("100% 맞혔다.", "17건이다.", "비율이 7.1 배 늘었다.", "11개다.", "1건이다."):
        assert not _passes(fs, text), text


def test_wrong_sign_with_unicode_minus_is_caught():
    fs = _fs(metrics=[MetricFact(metric_id="M1", display_name="Accuracy", value=0.8213, threshold=0.8,
                                 status="pass", margin=0.0213, margin_text="0.0213")])
    assert _passes(fs, "목표보다 0.0213 높다.")
    assert not _passes(fs, "목표와의 차이는 −0.0213이다.")  # 충족인데 음수로 — 유니코드 마이너스도 부호로 읽는다


def test_tiny_failing_margin_is_not_grounded_as_zero():
    fs = _fs(metrics=[MetricFact(metric_id="M1", display_name="Accuracy", value=0.79996, threshold=0.8,
                                 status="fail", margin=-0.00004, margin_text="-0.00004", value_text="0.79996")])
    assert _passes(fs, "목표와의 차이는 -0.00004 이다.")
    assert not _passes(fs, "목표와의 차이는 0 이다.")
    assert not _passes(fs, "목표와의 차이는 -0 이다.")


def test_negative_judged_difference_drops_short_rounding():
    # 학습 − 평가 비율 차 −0.09996(10%p 미만) — '−0.1' 은 판정과 반대로 읽히므로 넣지 않는다(크기 0.1 은 기준값으로 허용).
    row = {"label": "a", "training_ratio": 0.4, "evaluation_ratio": 0.49996, "difference": -0.09996,
           "difference_text": "-0.09996", "flagged": False}
    fs = _fs(facts=NarrativeFactsFact(distribution_shift=[row]))
    assert _passes(fs, "차이는 -0.09996 이다.")
    assert not _passes(fs, "차이는 -0.1 이다.")


def test_judged_share_pair_drops_short_rounding():
    weak = {"label": "c", "f1": 0.3, "support": 4999, "share": 0.04999, "share_text": "0.04999", "macro_f1": 0.5,
            "gap_from_macro": 0.2, "gap_text": "0.2", "low_support": True, "far_below_macro": True}
    fs = _fs(facts=NarrativeFactsFact(weakest_class=weak))
    assert _passes(fs, "전체의 4.999%로 5% 미만이다.")
    # '5%' 는 기준값으로 허용된다 — 숫자 검문으로는 '기준과 같아 보이는 값' 모순을 다 막지 못한다(한계, 판정 모순 검사·S5 칸 검사 몫).
    assert _passes(fs, "기준은 5% 이다.")


# ── 폴백 ───────────────────────────────────────────────────────────────────────

def _fallback(fs: FactSheet, task_type: str | None = None):
    return build_fallback_narrative(fs, [], compute_derived(fs, task_type=task_type), reason="no_key", edition=3)


def test_failed_metric_target_is_not_rounded_when_value_text_is_used():
    m = MetricFact(metric_id="M1", display_name="Accuracy", value=0.812, threshold=0.8125, status="fail",
                   margin=-0.0005, value_text="0.812", margin_text="-0.0005", is_core=True)
    out = _fallback(_fs(verdict="FAIL", score=0.0, metrics=[m]))
    assert "Accuracy(0.812, 목표 0.8125)" in out.conclusion.narrative


def test_failed_metric_uses_value_text_where_three_digits_look_equal():
    m = MetricFact(metric_id="M1", display_name="Accuracy", value=0.79996, threshold=0.8, status="fail",
                   margin=-0.00004, value_text="0.79996", margin_text="-0.00004", is_core=True)
    out = _fallback(_fs(verdict="FAIL", score=0.0, metrics=[m]))
    assert "Accuracy(0.79996, 목표 0.8)" in out.conclusion.narrative
    assert "근소하게 못 미침" not in out.conclusion.narrative


def test_near_target_tiny_margin_uses_margin_text():
    m = MetricFact(metric_id="M4", display_name="F1 Score", value=0.80004, threshold=0.8, status="pass",
                   margin=0.00004, near_target=True, value_text="0.80004", margin_text="0.00004")
    risks = _fallback(_fs(metrics=[m])).conclusion.risks
    assert "차이 0.00004" in risks and "차이 0)" not in risks


def test_threshold_sentence_uses_gain_text_not_four_digit_gain():
    gain = F1ThresholdGainFact(max_f1=0.6, current_f1=0.5, gain=0.1, best_threshold=0.3, notable=True, gain_text="0.10004")
    # S6: 8.2 로 옮김(결론 목록 8.2-06).
    assert "이 향상폭(0.10004)은" in _fallback(_fs(f1_threshold_gain=gain)).conclusion.narrative


def test_trimmed_multilabel_distribution_says_labels():
    facts = NarrativeFactsFact(omitted={"class_distribution": 9})
    fs = _fs(distribution=DistributionFact(class_distribution={"a": 5, "b": 4}, imbalance_ratio=1.25), facts=facts)
    text = _fallback(fs, "multilabel").interpretation.distribution_analysis
    assert "외 9개 레이블" in text


def test_emptied_distribution_still_reports_engine_imbalance():
    ie = ImbalanceEffectFact(level="affected", imbalance_ratio=3.0, accuracy_minus_macro_f1=0.2,
                             majority_minus_minority_f1=0.3, majority_class="a", minority_class="b")
    fs = _fs(imbalance_effect=ie, facts=NarrativeFactsFact(omitted={"class_distribution": 300}))
    text = _fallback(fs).interpretation.distribution_analysis
    assert "생략" in text and "불균형에 따른 성능 지표 왜곡이 관찰되었다" in text
    assert "분포 분석을 생략한다" not in text


# ── 결론 목록 ───────────────────────────────────────────────────────────────────

def _ids_texts(fs: FactSheet, task="binary", client=None) -> dict[str, str]:
    return {c.item_id: c.text for c in select_items(fs, client, task, edition=3)}


def test_weakest_share_is_printed_with_judged_digits_and_split_reasons():
    weak = {"label": "c", "f1": 0.3, "support": 40, "share": 0.04996, "share_text": "0.04996", "macro_f1": 0.35,
            "gap_from_macro": 0.05, "gap_text": "0.05", "low_support": True, "far_below_macro": False}
    text = _ids_texts(_fs(facts=NarrativeFactsFact(weakest_class=weak)), "multiclass")["7.1-10"]
    assert "4.996%" in text and "5%)" not in text
    assert "5% 미만이다" in text and "30건 미만" not in text


def test_ci_partially_skipped_does_not_claim_all():
    m1 = MetricFact(metric_id="M1", display_name="Accuracy", value=0.9, threshold=0.8, status="pass", margin=0.1,
                    ci_low=0.85, ci_high=0.95, ci_contains_target=False)
    m10 = MetricFact(metric_id="M10", display_name="AUPRC", value=0.9, threshold=0.8, status="pass", margin=0.1)
    facts = NarrativeFactsFact(ci_skipped=[{"metric_id": "M10", "display_name": "AUPRC", "reason": "점추정 불일치"}])
    text = _ids_texts(_fs(metrics=[m1, m10], facts=facts))["8.2-03"]
    assert "모두" not in text and "흔들리지 않는다" not in text
    assert "Accuracy" in text and "AUPRC(점추정 불일치)" in text


def test_skipped_only_without_judged_metrics_is_silent():
    facts = NarrativeFactsFact(ci_skipped=[{"metric_id": "M10", "display_name": "AUPRC", "reason": "x"}])
    m = MetricFact(metric_id="M10", display_name="AUPRC", value=0.9, status="info")
    assert "8.2-03" not in _ids_texts(_fs(metrics=[m], facts=facts))


@pytest.mark.parametrize("value, josa", [("0.3", "으로"), ("0.6", "으로"), ("0.13", "으로"), ("0.2", "로"), ("0.17", "로")])
def test_number_josa(value, josa):
    gain = F1ThresholdGainFact(max_f1=0.7, current_f1=0.5, gain=0.2, best_threshold=float(value), notable=True)
    text = _ids_texts(_fs(f1_threshold_gain=gain))["8.2-06"]
    assert f"결정 임계값을 {value}{josa} 바꾸면" in text


def test_threshold_gain_item_follows_engine_notable():
    gain = F1ThresholdGainFact(max_f1=0.7, current_f1=0.5, gain=0.2, best_threshold=0.3, notable=False)
    picked = _ids_texts(_fs(f1_threshold_gain=gain))
    assert "8.2-06" not in picked and "8.3-03" not in picked


def test_imbalance_rule_wording_by_task():
    ml = ImbalanceEffectFact(level="no_distortion", imbalance_ratio=2.0, accuracy_minus_macro_f1=None,
                             majority_minus_minority_f1=0.05, majority_class="a", minority_class="b")
    text = _ids_texts(_fs(imbalance_effect=ml), "multilabel")["7.2-02"]
    assert "다수·소수 레이블 F1 차이가 0.1 미만" in text and "두 F1" not in text


def test_metric_rationale_not_provided_is_off():
    fs = _fs()
    assert "8.2-12" not in _ids_texts(fs, "multilabel", {"metric_rationale": "제공되지 않음"})
    assert "8.2-12" in _ids_texts(fs, "multilabel", {"metric_rationale": "레이블 오류를 고르게 본다"})


def test_per_class_errors_listed_by_size():
    from app.narrative.schemas import ConfusionCountFact

    rows = [ConfusionCountFact(label=l, tp=1, fp=fp, fn=0, tn=1) for l, fp in (("a", 1), ("b", 9), ("c", 5), ("d", 0))]
    text = _ids_texts(_fs(confusion_counts=rows), "multiclass")["7.1-04"]
    # S6: '이름 FP n건·FN n건' 꼴(클래스 설명이 붙으면 '이름(설명)' 이라 괄호를 겹치지 않게). 많은 순 상위 셋만.
    assert text.index("b FP") < text.index("c FP") < text.index("a FP") and "d FP" not in text


def test_legacy_table_metrics_are_not_listed_as_missing_benchmarks():
    legacy = NarrativeRequest.model_validate(_example("legacy_main"))
    text = _ids_texts(legacy.fact_sheet)["8.1-03"]
    assert "Confusion Matrix" not in text and "Classification Report" not in text


def test_weakest_risk_says_label_for_multilabel():
    weak = {"label": "sports", "f1": 0.4, "support": 97, "share": 0.2, "share_text": "0.2", "macro_f1": 0.5,
            "gap_from_macro": 0.1, "gap_text": "0.1", "low_support": False, "far_below_macro": True}
    text = _ids_texts(_fs(facts=NarrativeFactsFact(weakest_class=weak)), "multilabel")["8.3-01"]
    assert text.startswith("가장 약한 레이블은 F1 0.4인 sports")


# ── 지시문 덜어내기 표시 ────────────────────────────────────────────────────────────

def test_prompt_trimming_marks_what_was_dropped():
    from app.narrative.baselines import build_benchmark_refs
    from app.narrative.legacy import build_prompts_within_limit
    from tests.test_s4b_narrative import _many_classes

    req = _many_classes(150)
    labels = [r.label for r in req.fact_sheet.per_class]
    fs = req.fact_sheet.model_copy(update={
        "facts": NarrativeFactsFact(),
        # 분포 단계까지 가도록 긴 분포도 싣는다 — 그 단계는 파생 백분율도 지시문 사본에서만 뺀다.
        "distribution": DistributionFact(class_distribution={f"{l}_{'y' * 300}": 20 for l in labels}, imbalance_ratio=1.0),
    })
    req = req.model_copy(update={"fact_sheet": fs})
    derived = compute_derived(fs, task_type="multiclass")
    before = copy.deepcopy(derived)
    _, user, dropped = build_prompts_within_limit(req, fs, derived, build_benchmark_refs("multiclass", fs.metrics, edition=3))
    assert dropped[0] == "confusion.matrix" and "distribution" in dropped
    assert '"omitted": {"confusion": 150' in user and '"class_distribution": 150' in user
    assert derived == before  # 원본 파생값(허용 목록·폴백이 씀)은 그대로
    assert "percentages" in derived["distribution"]
