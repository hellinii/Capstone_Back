"""tests/test_s4b_narrative.py — S4b: derived 가 facts 를 옮기는지, 폴백이 표시값을 쓰는지, 지시문 덜어내기.

- derived 는 판정 카운트·구간 포함·행렬 없는 오분류 총건수를 프론트 facts 에서 옮긴다(같은 숫자를 두 곳에서
  세지 않는다 — 계획 설계 요지 4). facts 없는 옛 요청만 서버가 센다.
- 폴백은 엔진·프론트가 판정과 같은 쪽으로 읽히게 만든 표기(display·gain_text·value_text·margin_text)를 쓴다.
- 혼동행렬·클래스별 표가 늘 실려(S4a) 지시문이 8만 자를 넘으면, 지시문에서만 큰 표부터 덜어 내고 AI 를 부른다.
"""
import json
from pathlib import Path

import pytest

from app.core.schemas import TaskType
from app.narrative import legacy as service
from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.fallback import build_fallback_narrative
from app.narrative.grounding import (
    _collect_grounding_texts,
    build_number_whitelist,
    find_verdict_contradictions,
    verify_grounding,
)
from app.narrative.schemas import (
    ConfusionCountFact,
    ConfusionFact,
    F1ThresholdGainFact,
    FactSheet,
    ImbalanceEffectFact,
    MetricFact,
    NarrativeFactsFact,
    NarrativeRequest,
    PerClassFact,
)
from app.narrative.legacy import build_prompts_within_limit, generate_narrative
from tests.conftest import fake_beta_message

CONTRACT = Path(__file__).parent / "contract"


def _fs(**over) -> FactSheet:
    base = dict(verdict="PASS", score=100.0, n_samples=200)
    base.update(over)
    return FactSheet(**base)


def _metric(**over) -> MetricFact:
    base = dict(metric_id="M1", display_name="Accuracy", value=0.9, threshold=0.8, status="pass", margin=0.1)
    base.update(over)
    return MetricFact(**base)


# ── derived ─────────────────────────────────────────────────────────────────

def test_counts_come_from_facts_verdict_detail_when_present():
    # 서버 셈(목표 있는 지표 2개 중 통과 1개)과 일부러 다른 값 — facts 를 옮기는지 본다.
    metrics = [_metric(), _metric(metric_id="M4", status="fail", value=0.7, margin=-0.1)]
    facts = NarrativeFactsFact(verdict_detail={"verdict": "FAIL", "judged_count": 3, "passed_count": 2})
    assert compute_derived(_fs(metrics=metrics, facts=facts))["counts"]["target"] == 3
    assert compute_derived(_fs(metrics=metrics, facts=facts))["counts"]["passed"] == 2
    assert compute_derived(_fs(metrics=metrics))["counts"] == {"target": 2, "passed": 1, "n_metrics": 2}


def test_ci_contains_target_follows_front_flag_over_rounded_values():
    # 실린 4자리 구간은 목표 0.85 를 포함해 보이지만(0.8 ~ 0.85), 프론트가 반올림 전 값으로 '포함 아님'을 정했다.
    m = _metric(threshold=0.85, value=0.82, ci_low=0.8, ci_high=0.85, ci_contains_target=False, status="fail", margin=-0.03)
    assert compute_derived(_fs(metrics=[m]))["ci_contains_target"] == []
    legacy = _metric(threshold=0.85, value=0.82, ci_low=0.8, ci_high=0.85, status="fail", margin=-0.03)
    assert compute_derived(_fs(metrics=[legacy]))["ci_contains_target"] == ["Accuracy"]


def test_error_totals_stand_in_when_matrix_was_trimmed():
    facts = NarrativeFactsFact(error_totals={"unit": "sample", "total": 5000, "misclassified": 1002})
    conf = compute_derived(_fs(facts=facts))["confusion"]
    assert conf == {"total": 5000, "correct": 3998, "misclassified": 1002, "correct_pct": 80.0, "misclassified_pct": 20.0}
    # 행렬이 있으면 행렬이 먼저다.
    with_matrix = _fs(facts=facts, confusion=ConfusionFact(labels=["0", "1"], matrix=[[3, 1], [1, 5]]))
    assert compute_derived(with_matrix)["confusion"]["total"] == 10
    label_cells = NarrativeFactsFact(error_totals={"unit": "label_cell", "total": 800, "misclassified": 416})
    assert compute_derived(_fs(facts=label_cells))["confusion"]["unit"] == "label_cell"


def test_threshold_tuning_is_the_engine_notable():
    gain = dict(max_f1=0.6, current_f1=0.5, best_threshold=0.3)
    assert compute_derived(_fs(f1_threshold_gain=F1ThresholdGainFact(**gain, gain=0.1, notable=False)))["threshold_tuning"] is False
    assert compute_derived(_fs(f1_threshold_gain=F1ThresholdGainFact(**gain, gain=0.05, notable=True)))["threshold_tuning"] is True
    # notable 이 없는 옛 요청만 4자리 gain 으로.
    assert compute_derived(_fs(f1_threshold_gain=F1ThresholdGainFact(**gain, gain=0.1)))["threshold_tuning"] is True
    assert compute_derived(_fs(f1_threshold_gain=F1ThresholdGainFact(**gain, gain=0.0999)))["threshold_tuning"] is False


# ── 폴백이 표시값을 쓴다 ───────────────────────────────────────────────────────────

def _fallback(fs: FactSheet):
    return build_fallback_narrative(fs, [], compute_derived(fs), reason="no_key", edition=3)


def _imbalance(level: str, ir: float, acc: float, gap: float, shown: tuple[str, str, str]):
    return ImbalanceEffectFact(
        level=level, imbalance_ratio=ir, accuracy_minus_macro_f1=acc, majority_minus_minority_f1=gap,
        majority_class="a", minority_class="b",
        display={"imbalance_ratio": shown[0], "accuracy_minus_macro_f1": shown[1], "majority_minus_minority_f1": shown[2]},
    )


def _dist():
    from app.narrative.schemas import DistributionFact

    return DistributionFact(class_distribution={"a": 150, "b": 50}, imbalance_ratio=3.0)


def test_mild_imbalance_prints_the_judged_display_value():
    fs = _fs(distribution=_dist(), imbalance_effect=_imbalance("mild", 1.5, 0.01, 0.02, ("1.49996", "0.01", "0.02")))
    text = _fallback(fs).interpretation.distribution_analysis
    assert "불균형 비율(1.49996)이 1.5 미만" in text


def test_affected_imbalance_prints_display_values_for_gaps():
    shown = ("3.0012", "0.09996", "0.25")
    fs = _fs(distribution=_dist(), imbalance_effect=_imbalance("affected", 3.0012, 0.1, 0.25, shown))
    text = _fallback(fs).interpretation.distribution_analysis
    assert "불균형 비율(3.0012)" in text
    assert "F1 차이(0.25)" in text
    assert "정확도와 매크로 F1 의 차이(0.09996)는 기준(0.1) 미만" in text


def test_failed_and_near_target_metrics_use_value_and_margin_texts():
    failed = _metric(value=0.7996, threshold=0.8, status="fail", margin=-0.0004, value_text="0.7996", margin_text="-0.0004", is_core=True)
    near = _metric(metric_id="M4", display_name="F1 Score", value=0.8100004, threshold=0.8, status="pass",
                   margin=0.0100004, near_target=True, value_text="0.810", margin_text="0.01")
    out = _fallback(_fs(verdict="FAIL", score=50.0, metrics=[failed, near]))
    assert "Accuracy(0.7996, 목표 0.8)" in out.conclusion.narrative
    assert "F1 Score(0.810, 목표 0.8, 차이 0.01)" in out.conclusion.risks


def test_threshold_sentence_uses_gain_text():
    gain = F1ThresholdGainFact(max_f1=0.65, current_f1=0.5123, gain=0.1377, best_threshold=0.03, notable=True, gain_text="0.1377")
    out = _fallback(_fs(f1_threshold_gain=gain))
    # S6: 임계값 숫자는 8.2(결론 목록 8.2-06)에. 종전 '이 0.1377는' 은 조사도 틀렸다(칠 → '은').
    assert "이 향상폭(0.1377)은" in out.conclusion.narrative


def test_trimmed_distribution_is_not_listed_as_complete():
    facts = NarrativeFactsFact(omitted={"class_distribution": 44})
    text = _fallback(_fs(distribution=_dist(), facts=facts)).interpretation.distribution_analysis
    assert "외 44개 클래스" in text


@pytest.mark.parametrize("name", ["binary", "multiclass", "multilabel", "legacy_main"])
def test_fallback_on_contract_examples_passes_its_own_checks(name):
    req = NarrativeRequest.model_validate(json.loads((CONTRACT / f"fact_sheet_{name}.json").read_text(encoding="utf-8")))
    fs = req.fact_sheet
    derived = compute_derived(fs, task_type=req.task_type.value)
    refs = build_benchmark_refs(req.task_type.value, fs.metrics, edition=3)
    out = build_fallback_narrative(fs, refs, derived, reason="no_key", edition=3)
    data = out.model_dump(exclude={"meta", "benchmark_refs"})
    grounding = verify_grounding(_collect_grounding_texts(data), build_number_whitelist(fs, refs, derived, edition=3), edition=3)
    assert grounding.passed, grounding.violations
    assert find_verdict_contradictions(_collect_grounding_texts(data), fs.verdict) == []
    # M21 을 고르지 않은 새 요청도 혼동행렬 문장이 있다(S4a — 늘 계산하는 표).
    assert "제공되지 않아" not in out.interpretation.confusion_analysis


# ── 지시문 덜어내기 ──────────────────────────────────────────────────────────────

def _many_classes(n: int) -> NarrativeRequest:
    labels = [f"class_{i:03d}" for i in range(n)]
    fs = _fs(
        verdict="FAIL", score=0.0, n_samples=n * 20,
        metrics=[_metric(value=0.5, status="fail", margin=-0.3, is_core=True)],
        per_class=[PerClassFact(label=l, precision=0.5, recall=0.5, f1=0.5, support=20) for l in labels],
        confusion=ConfusionFact(labels=labels, matrix=[[10 if i == j else 0 for j in range(n)] for i in range(n)]),
        confusion_counts=[ConfusionCountFact(label=l, tp=10, fp=10, fn=10, tn=n * 20 - 30) for l in labels],
    )
    return NarrativeRequest(task_type=TaskType.multiclass, fact_sheet=fs)


def test_prompt_drops_big_tables_first_and_keeps_within_limit():
    req = _many_classes(150)
    fs = req.fact_sheet
    derived = compute_derived(fs, task_type="multiclass")
    refs = build_benchmark_refs("multiclass", fs.metrics, edition=3)
    system, user, dropped = build_prompts_within_limit(req, fs, derived, refs)
    assert dropped[0] == "confusion.matrix"
    assert len(system) + len(user) <= service.MAX_NARRATIVE_PROMPT_CHARS
    # 작은 요청은 덜어 내지 않는다.
    small = _many_classes(3)
    _, _, none = build_prompts_within_limit(small, small.fact_sheet, compute_derived(small.fact_sheet), [])
    assert none == []


async def test_many_class_request_still_calls_the_model_after_trimming(make_fake_anthropic_client):
    clean = {
        "interpretation": {"confusion_analysis": "클래스별 오류를 정리했다.", "distribution_analysis": ""},
        "conclusion": {"verdict": "FAIL", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
    }
    client = make_fake_anthropic_client(clean)
    trace: dict = {}
    resp = await generate_narrative(client, _many_classes(150), trace=trace)
    assert resp.meta.source == "llm", resp.meta.reason
    assert trace["prompt_trimmed"][0] == "confusion.matrix"
    client.beta.messages.create.assert_called_once()
    sent = client.beta.messages.create.call_args.kwargs["messages"][0]["content"]
    assert '"matrix": []' in sent
