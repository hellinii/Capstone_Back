"""test_narrative_fallback_p0.py — 규칙 폴백 문장의 P0 세 건(서술 라운드 S1).

10-23 이후 Anthropic 키가 없거나 호출이 실패하면 모든 성적서가 이 문장을 쓴다. 점검에서 재현된 틀린 문장:
- BE-02 7.2 불균형: 두 차이 중 하나만 0.10 이상이어도 '영향 있음'인데, 둘 다 늘어놓고 '0.1 이상 벌어져'라고 썼다.
- BE-03 8.2·8.3 판정: 통과율로 판정을 설명했고, 판정 불가에 '모든 시험항목이 기준을 충족'이라 써 판정 모순 검사에 걸렸다.
- BE-09 멀티레이블 7.1: 레이블 칸 합계(표본 × 레이블)를 '전체 N건'으로 써 7.2 의 표본 수와 어긋났다.
"""
import re

import pytest

from app.core.schemas import TaskType
from app.narrative.derived import compute_derived
from app.narrative.fallback import build_fallback_narrative
from app.narrative.grounding import build_number_whitelist, find_verdict_contradictions, verify_grounding
from app.narrative.schemas import (
    ConfusionFact, DistributionFact, FactSheet, ImbalanceEffectFact, MetricFact, NarrativeRequest, ProblemFact,
)

_CONF = ConfusionFact(labels=["0", "1"], matrix=[[120, 10], [15, 55]], positive_class="1")
_DIST = DistributionFact(class_distribution={"0": 130, "1": 70}, imbalance_ratio=1.8571)


def _m(mid, name, value, threshold, status, core=False):
    return MetricFact(metric_id=mid, display_name=name, value=value, threshold=threshold, status=status, is_core=core)


def _fs(**kw) -> FactSheet:
    base = dict(n_samples=200, confusion=_CONF, distribution=_DIST, verdict="PASS", score=100.0,
                metrics=[_m("M1", "Accuracy", 0.94, 0.85, "pass", True)])
    base.update(kw)
    return FactSheet(**base)


def _run(fs: FactSheet, task_type: str | None = None):
    derived = compute_derived(fs, task_type=task_type)
    resp = build_fallback_narrative(fs, [], derived, reason="no_key", edition=3)
    texts = [resp.interpretation.confusion_analysis, resp.interpretation.distribution_analysis,
             resp.conclusion.narrative, resp.conclusion.risks]
    grounding = verify_grounding(texts, build_number_whitelist(fs, [], derived, edition=3))
    assert grounding.passed, grounding.violations
    return resp, texts


# ── BE-02 7.2 불균형 ────────────────────────────────────────────────────────

def _ie(level, acc, cls, ir=2.9683):
    return ImbalanceEffectFact(level=level, imbalance_ratio=ir, accuracy_minus_macro_f1=acc,
                               majority_minus_minority_f1=cls, majority_class="0", minority_class="1")


def test_only_accuracy_gap_is_cited_as_over_and_negative_class_gap_is_explained():
    """점검 재현값: 정확도 차이 0.1533 만 기준 이상, 클래스 차이 −0.0213(소수가 더 잘 맞음)."""
    resp, _ = _run(_fs(imbalance_effect=_ie("affected", 0.1533, -0.0213)))
    text = resp.interpretation.distribution_analysis
    assert "정확도와 매크로 F1 의 차이(0.1533)가 기준(0.1) 이상이어서 불균형에 따른 성능 지표 왜곡이 관찰되었다." in text
    assert "F1 차이(-0.0213)는 음수로 소수 클래스의 F1 이 더 높다." in text
    assert "벌어져" not in text
    # 기준 이상이라고 말하는 문장에 음수 차이가 들어가지 않는다.
    over_sentence = next(s for s in text.split(". ") if "기준(0.1) 이상" in s)
    assert "-0.0213" not in over_sentence


def test_only_class_gap_is_cited_as_over_and_small_accuracy_gap_is_below():
    resp, _ = _run(_fs(imbalance_effect=_ie("affected", 0.03, 0.25)))
    text = resp.interpretation.distribution_analysis
    assert "다수 클래스(0)와 소수 클래스(1)의 F1 차이(0.25)가 기준(0.1) 이상" in text
    assert "정확도와 매크로 F1 의 차이(0.03)는 기준(0.1) 미만이다." in text
    over_sentence = next(s for s in text.split(". ") if "기준(0.1) 이상" in s)
    assert "0.03" not in over_sentence


def test_both_gaps_over_are_both_cited():
    resp, _ = _run(_fs(imbalance_effect=_ie("affected", 0.12, 0.18)))
    text = resp.interpretation.distribution_analysis
    assert "차이(0.12)와 다수 클래스(0)와 소수 클래스(1)의 F1 차이(0.18)가 기준(0.1) 이상" in text
    assert "미만" not in text


def test_gap_that_rounds_to_threshold_but_engine_saw_below_is_written_without_number():
    """엔진은 0.09996 을 기준 미만으로 보고 넷째 자리 0.1 로 싣는다 — '차이(0.1)는 기준(0.1) 미만'은 모순이다."""
    resp, _ = _run(_fs(imbalance_effect=_ie("no_distortion", 0.1, 0.05)))
    text = resp.interpretation.distribution_analysis
    assert "정확도와 매크로 F1 의 차이는 기준(0.1) 미만이다." in text
    assert "차이(0.1)" not in text
    assert "왜곡은 관찰되지 않았다" in text and "영향을 준" not in text


def test_mild_ratio_that_rounds_to_1_5_is_written_without_number():
    resp, _ = _run(_fs(imbalance_effect=_ie("mild", 0.01, 0.02, ir=1.5)))
    text = resp.interpretation.distribution_analysis
    assert "불균형 비율이 1.5 미만이어서 불균형은 경미하다." in text
    assert "(1.5)" not in text


def test_multilabel_imbalance_uses_label_gap_only():
    ie = ImbalanceEffectFact(level="affected", imbalance_ratio=4.0, accuracy_minus_macro_f1=None,
                             majority_minus_minority_f1=0.3, majority_class="sports", minority_class="tech")
    resp, _ = _run(_fs(imbalance_effect=ie))
    text = resp.interpretation.distribution_analysis
    assert "다수 레이블(sports)과 소수 레이블(tech)의 F1 차이(0.3)가 기준(0.1) 이상" in text
    assert "매크로" not in text and "소수 레이블의 성능 지표 해석에 유의가 필요하다." in text and "표본이 적어" not in text


# ── BE-03 8.2·8.3 판정 ──────────────────────────────────────────────────────

_INCONCLUSIVE_PROBLEM = ProblemFact(
    id="P1", kind="inconclusive", priority="HIGH", title="F1 Score", source_section="8.2", recommend_section="9.2",
    evidence={"cause": "핵심 지표 산출 실패", "reason": "core_not_computed", "metrics": ["F1 Score"], "judged_count": 2},
)

_VERDICT_CASES = {
    # 새 프론트(is_core·problems 있음)
    "FAIL": dict(verdict="FAIL", score=66.7, metrics=[
        _m("M1", "Accuracy", 0.94, 0.85, "pass", True), _m("M4", "F1 Score", 0.70, 0.80, "fail", True),
        _m("M9", "AUROC", 0.75, 0.80, "fail")]),
    "CONDITIONAL_PASS": dict(verdict="CONDITIONAL_PASS", score=66.7, metrics=[
        _m("M1", "Accuracy", 0.94, 0.85, "pass", True), _m("M4", "F1 Score", 0.82, 0.80, "pass", True),
        _m("M9", "AUROC", 0.75, 0.80, "fail")]),
    "PASS": dict(verdict="PASS", score=100.0, metrics=[
        _m("M1", "Accuracy", 0.94, 0.85, "pass", True), _m("M4", "F1 Score", 0.82, 0.80, "pass", True)]),
    "INCONCLUSIVE": dict(verdict="INCONCLUSIVE", score=100.0, problems=[_INCONCLUSIVE_PROBLEM], metrics=[
        _m("M1", "Accuracy", 0.94, 0.85, "pass"), _m("M9", "AUROC", 0.88, 0.80, "pass")]),
    # 옛 프론트(9-22) 모양 — is_core·problems 가 없다.
    "FAIL_old": dict(verdict="FAIL", score=66.7, metrics=[
        _m("M1", "Accuracy", 0.94, 0.85, "pass"), _m("M4", "F1 Score", 0.70, 0.80, "fail")]),
    "INCONCLUSIVE_old": dict(verdict="INCONCLUSIVE", score=100.0, metrics=[
        _m("M1", "Accuracy", 0.94, 0.85, "pass"), _m("M9", "AUROC", 0.88, 0.80, "pass")]),
    "INCONCLUSIVE_no_target": dict(verdict="INCONCLUSIVE", score=0.0, metrics=[_m("M1", "Accuracy", 0.94, None, "info")]),
    "INCONCLUSIVE_with_fail": dict(verdict="INCONCLUSIVE", score=50.0, problems=[_INCONCLUSIVE_PROBLEM], metrics=[
        _m("M1", "Accuracy", 0.94, 0.85, "pass"), _m("M9", "AUROC", 0.75, 0.80, "fail")]),
}

# 성적서 8.2 판정 상자(프론트 ConclusionSection VERDICT_CONFIG)와 같은 이름.
_BOX_LABEL = {"PASS": "최종 합격", "CONDITIONAL_PASS": "조건부 합격", "FAIL": "최종 불합격", "INCONCLUSIVE": "판정 불가"}


@pytest.mark.parametrize("case", list(_VERDICT_CASES))
def test_verdict_sentences_have_no_self_contradiction_and_use_box_label(case):
    fs = _fs(**_VERDICT_CASES[case])
    resp, texts = _run(fs)
    assert find_verdict_contradictions(texts, fs.verdict) == []
    narrative = resp.conclusion.narrative
    assert f"종합 판정은 '{_BOX_LABEL[fs.verdict]}'이다" in narrative
    for old in ("적합(합격)", "조건부 적합", "부적합", "임계값이 설정된"):
        assert old not in " ".join(texts), old
    # 통과율은 판정 근거가 아니라 참고 — '참고로' 문장 안에만 나온다.
    if "통과율" in narrative:
        assert narrative.index("참고로") < narrative.index("통과율")
        assert narrative.index("종합 판정은") < narrative.index("참고로")


def test_pass_rate_is_omitted_when_nothing_is_judged():
    resp, _ = _run(_fs(**_VERDICT_CASES["INCONCLUSIVE_no_target"]))
    assert "통과율" not in resp.conclusion.narrative
    assert "0개 중" not in resp.conclusion.narrative


@pytest.mark.parametrize("case", ["INCONCLUSIVE", "INCONCLUSIVE_old", "INCONCLUSIVE_no_target", "INCONCLUSIVE_with_fail"])
def test_inconclusive_never_claims_everything_passed(case):
    resp, _ = _run(_fs(**_VERDICT_CASES[case]))
    risks = resp.conclusion.risks
    assert "모든 시험항목이" not in risks and "위험은 없다" not in risks
    assert "확인하지 못했다" in risks


def test_fail_names_the_failed_core_metric_and_separates_noncore():
    resp, _ = _run(_fs(**_VERDICT_CASES["FAIL"]))
    n = resp.conclusion.narrative
    assert "핵심 지표가 목표에 미달하여 종합 판정은 '최종 불합격'이다(미달한 핵심 지표: F1 Score(0.7, 목표 0.8))." in n
    assert "핵심이 아닌 지표 중에도 목표에 미달한 지표가 있다(목표 미달 지표: AUROC(0.75, 목표 0.8))." in n


def test_fail_without_core_info_does_not_guess_which_metric_is_core():
    resp, _ = _run(_fs(**_VERDICT_CASES["FAIL_old"]))
    n = resp.conclusion.narrative
    assert "미달한 핵심 지표" not in n
    assert "핵심 지표가 목표에 미달하여 종합 판정은 '최종 불합격'이다(목표 미달 지표: F1 Score(0.7, 목표 0.8))." in n


def test_conditional_pass_says_core_met_and_noncore_failed():
    resp, _ = _run(_fs(**_VERDICT_CASES["CONDITIONAL_PASS"]))
    assert resp.conclusion.narrative.startswith(
        "핵심 지표는 모두 목표를 충족하였으나 핵심이 아닌 지표가 목표에 미달하여 종합 판정은 '조건부 합격'이다"
        "(목표 미달 지표: AUROC(0.75, 목표 0.8))."
    )


def test_inconclusive_states_cause_from_problem_list():
    resp, _ = _run(_fs(**_VERDICT_CASES["INCONCLUSIVE"]))
    assert "종합 판정은 '판정 불가'이다(원인: 핵심 지표 산출 실패 — F1 Score)." in resp.conclusion.narrative


@pytest.mark.parametrize("value, expected", [
    (0.7999, "Accuracy(0.7999, 목표 0.8)"),             # 셋째 자리로는 0.8 = 목표처럼 보인다
    (0.79999, "Accuracy(목표 0.8에 근소하게 못 미침)"),  # 넷째 자리로도 같아 보인다
    (0.7, "Accuracy(0.7, 목표 0.8)"),
])
def test_failed_value_that_looks_equal_to_target_gets_more_digits(value, expected):
    resp, texts = _run(_fs(verdict="FAIL", score=0.0, metrics=[_m("M1", "Accuracy", value, 0.8, "fail", True)]))
    assert expected in resp.conclusion.narrative and expected in resp.conclusion.risks
    assert "Accuracy(0.8, 목표 0.8)" not in " ".join(texts)


# ── BE-09 멀티레이블 7.1 ────────────────────────────────────────────────────

def _multilabel_fs() -> FactSheet:
    return FactSheet(
        n_samples=200, verdict="PASS", score=100.0,
        metrics=[_m("M1", "Hamming", 0.9, 0.8, "pass", True)],
        confusion=ConfusionFact(labels=["Negative (전체 레이블 합계)", "Positive (전체 레이블 합계)"],
                                matrix=[[600, 40], [56, 104]]),
        distribution=DistributionFact(class_distribution={"sports": 120, "news": 90, "finance": 60, "tech": 30},
                                      imbalance_ratio=4.0),
    )


def test_multilabel_confusion_is_written_in_label_cells_not_samples():
    resp, _ = _run(_multilabel_fs(), task_type="multilabel")
    c71, c72 = resp.interpretation.confusion_analysis, resp.interpretation.distribution_analysis
    # S6: 결론 목록 문장(오분류 총건수·비율). '맞은 칸'은 빼고 비율을 붙였다 — 단위는 그대로 레이블 칸.
    assert "레이블 칸 800칸 중 96칸(12%)이 틀렸다." in c71
    assert "FN)은 56칸" in c71 and "FP)은 40칸" in c71
    assert "800건" not in c71 and "전체 800" not in c71
    # 7.1 이 '건'으로 쓰는 수는 표본 단위뿐이어야 한다 — 7.2 의 표본 수와 어긋나지 않게.
    assert not re.search(r"\d+건", c71)
    assert "총 200건" in c72 and "레이블별 등장 분포" in c72


def test_binary_and_old_callers_keep_sample_wording():
    fs = _multilabel_fs()
    resp, _ = _run(fs)  # task_type 없음(옛 호출) — 종전 그대로
    assert "전체 800건" in resp.interpretation.confusion_analysis
    resp, _ = _run(_fs(), task_type="binary")
    assert "혼동행렬 기준 전체 200건 중 25건(12.5%)이 오분류되었다" in resp.interpretation.confusion_analysis


async def test_service_passes_task_type_so_multilabel_fallback_uses_label_cells():
    """배선: 서비스가 compute_derived 에 작업 유형을 넘기는지(키 없음 → 폴백 경로)."""
    from app.narrative.legacy import generate_narrative

    req = NarrativeRequest(task_type=TaskType.multilabel, report_purpose="internal", fact_sheet=_multilabel_fs())
    resp = await generate_narrative(None, req)
    assert resp.meta.reason == "no_key"
    assert "레이블 칸 800칸" in resp.interpretation.confusion_analysis


# ── 옛 프론트(9-22) 옛 판정 규칙의 조건부 합격(S1 배포 전 점검에서 발견) ─────────────
# 옛 computeVerdict 는 목표값이 하나도 없거나 핵심 지표가 산출되지 않으면 미달이 없어도 CONDITIONAL_PASS 다.

def test_old_front_conditional_pass_without_any_target_does_not_claim_failure_or_all_passed():
    fs = _fs(verdict="CONDITIONAL_PASS", score=0.0, metrics=[
        _m("M1", "Accuracy", 0.505, None, "info"), _m("M4", "F1 Score", 0.512, None, "info")])
    resp, texts = _run(fs)
    n, r = resp.conclusion.narrative, resp.conclusion.risks
    assert n == "목표값이 설정된 지표가 없어 목표 충족 여부로 판정하지 못했으며, 종합 판정은 '조건부 합격'이다."
    assert r == "목표값이 설정된 지표가 없어 목표 미달 위험을 판단하지 못했다."
    assert "미달하여" not in n and "모든 시험항목" not in r and "통과율" not in n
    assert find_verdict_contradictions(texts, fs.verdict) == []


def test_old_front_conditional_pass_with_core_unavailable_does_not_claim_noncore_failure():
    fs = _fs(verdict="CONDITIONAL_PASS", score=100.0, metrics=[_m("M9", "AUROC", 0.88, 0.80, "pass")])
    resp, texts = _run(fs)
    n = resp.conclusion.narrative
    assert n.startswith("목표값이 설정된 지표 중 목표에 미달한 지표는 없으나 핵심 지표가 산출되지 않아 종합 판정은 '조건부 합격'이다.")
    assert "핵심이 아닌 지표가 목표에 미달" not in n
    assert "확인하지 못했다" in resp.conclusion.risks and "위험은 없다" not in resp.conclusion.risks
    assert find_verdict_contradictions(texts, fs.verdict) == []



# ── 독립 검토(S1) 반영 ──────────────────────────────────────────────────────

def test_negative_accuracy_gap_does_not_claim_minority_is_better():
    """S1-4-FB-01: 정확도 − 매크로 F1 이 음수인 것은 소수 클래스가 더 잘 맞았다는 뜻이 아니다(엔진에서 나오는 값)."""
    for level, cls in (("no_distortion", 0.0763), ("affected", 0.15)):
        resp, _ = _run(_fs(imbalance_effect=_ie(level, -0.0022, cls, ir=6.5)))
        text = resp.interpretation.distribution_analysis
        assert "소수 클래스의 F1 이 더 높다" not in text
        assert "정확도와 매크로 F1 의 차이(-0.0022)는 기준(0.1) 미만이다." in text


def test_target_zero_never_prints_minus_zero():
    """S1-4-FB-02: 목표 0, 값 −0.00003 은 넷째 자리로도 0 — '-0' 이 아니라 '근소하게 못 미침'."""
    resp, texts = _run(_fs(verdict="FAIL", score=0.0, metrics=[_m("M20", "MCC", -0.00003, 0.0, "fail", True)]))
    joined = " ".join(texts)
    assert "-0," not in joined and "(-0" not in joined
    assert "MCC(목표 0에 근소하게 못 미침)" in resp.conclusion.narrative


def test_inconclusive_cause_without_metrics_has_no_dangling_dash():
    """T2: 판정 대상이 0개(no_judged)면 프론트가 metrics=[] 를 보낸다."""
    p = ProblemFact(id="P1", kind="inconclusive", priority="HIGH", title="판정 대상 지표 없음", source_section="8.2",
                    recommend_section="9.2", evidence={"cause": "판정 대상 지표 없음", "reason": "no_judged",
                                                      "metrics": [], "judged_count": 0})
    resp, _ = _run(_fs(verdict="INCONCLUSIVE", score=0.0, problems=[p], metrics=[_m("M1", "Accuracy", 0.9, None, "info")]))
    n = resp.conclusion.narrative
    assert "종합 판정은 '판정 불가'이다(원인: 판정 대상 지표 없음)." in n
    assert " — " not in n
