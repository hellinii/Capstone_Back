"""test_s5_checks.py — 서술 라운드 S5 칸별 검사(app.narrative.checks)와 판정 모순 검사의 회귀 표.

판정 모순(점검 BE-04, 생-5): 맞는 문장을 잡아 서술을 버리고, 판정을 뒤집는 문장은 놓쳤다. 점검이 실행해 본 19문장을
그대로 표로 고정한다(걸리면 안 되는 11 · 잡아야 하는 8). 기존 회귀 문장은 tests/test_environment_and_grounding.py.
"""
import pytest

from app.narrative.grounding import claimed_verdicts, find_verdict_contradictions

# ── 점검 BE-04 의 19문장 ─────────────────────────────────────────────────────────

BE04_ALLOWED = [
    # 종전에 오탐(서술 전체 폴백)이던 것 — 판정과 맞는 문장이다.
    ("PASS", "판정 대상 지표 가운데 기준 미달 지표는 없다."),
    ("PASS", "어떤 지표도 기준에 미달하지 않았다."),
    ("PASS", "충족하지 못한 지표는 없다."),
    ("PASS", "핵심 지표가 하나라도 미달하면 불합격으로 판정하는 규칙에 따라 종합 판정은 합격이다."),
    ("PASS", "표본을 늘려 재평가할 때 미달 여부를 확인한다."),
    ("PASS", "표본이 달라졌을 때 미달로 바뀌는지 확인할 필요가 있다."),
    ("PASS", "F1 Score 는 목표에 근접해 미달에 가까운 수준이다."),
    ("CONDITIONAL_PASS", "핵심 지표는 기준을 모두 충족하였으나 보조 지표 Recall이 미달하여 조건부 합격이다."),
    ("FAIL", "핵심 지표 Accuracy 가 미달하였고, 그 외 모든 지표는 기준을 충족했다."),
    ("FAIL", "전체 정확도는 기준을 통과했으나 핵심 지표인 F1 Score 가 미달하였다."),
    ("FAIL", "종합 판정은 '최종 불합격'이다."),
]

BE04_CAUGHT = [
    # 종전에 놓치던 것 — 판정을 뒤집는 문장이다.
    ("FAIL", "종합 판정은 합격이다."),
    ("FAIL", "본 모델은 목표 성능을 충족하여 운영에 적합하다."),
    ("FAIL", "모든 지표가 목표값 이상이다."),
    ("CONDITIONAL_PASS", "종합 판정은 합격이며 추가 조치는 필요하지 않다."),
    ("INCONCLUSIVE", "본 모델은 불합격으로 판정되었다."),
    ("INCONCLUSIVE", "종합 판정은 합격이다."),
    ("PASS", "종합 판정은 조건부 합격이다."),
    ("PASS", "본 모델의 판정은 FAIL이다."),
]


def test_be04_table_has_nineteen_sentences():
    assert len(BE04_ALLOWED) + len(BE04_CAUGHT) == 19


@pytest.mark.parametrize("verdict, text", BE04_ALLOWED)
def test_be04_consistent_sentences_are_not_flagged(verdict, text):
    assert find_verdict_contradictions([text], verdict) == []


@pytest.mark.parametrize("verdict, text", BE04_CAUGHT)
def test_be04_contradicting_sentences_are_flagged(verdict, text):
    assert find_verdict_contradictions([text], verdict) == [text]


# ── 판정 이름 주장 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text, codes", [
    ("종합 판정은 '최종 불합격'이다.", ["FAIL"]),
    ("종합 판정은 ‘조건부 합격’이며 보조 지표가 미달했다.", ["CONDITIONAL_PASS"]),
    ("종합 판정은 판정 불가이다.", ["INCONCLUSIVE"]),
    ("종합 판정 결과는 최종 합격으로 결정되었다.", ["PASS"]),
    ("조건부 합격으로 판정되었다.", ["CONDITIONAL_PASS"]),
    ("최종 불합격 판정을 받았다.", ["FAIL"]),
    ("판정은 PASS이다.", ["PASS"]),
    # 단정이 아니다 — 앞일·규칙.
    ("표본이 달라지면 판정이 불합격으로 바뀔 수 있다.", []),
    ("핵심 지표가 미달하면 불합격으로 판정하는 규칙이다.", []),
    ("판정이 합격 쪽으로 기울어 있다.", []),
])
def test_claimed_verdicts(text, codes):
    assert claimed_verdicts(text) == codes


@pytest.mark.parametrize("verdict", ["PASS", "CONDITIONAL_PASS", "FAIL", "INCONCLUSIVE"])
def test_naming_the_right_verdict_is_never_a_contradiction(verdict):
    name = {"PASS": "최종 합격", "CONDITIONAL_PASS": "조건부 합격", "FAIL": "최종 불합격", "INCONCLUSIVE": "판정 불가"}[verdict]
    assert find_verdict_contradictions([f"종합 판정은 '{name}'이다."], verdict) == []


@pytest.mark.parametrize("verdict, text", [
    # 문장 단위로 본다 — 앞 문장의 맞는 말이 뒤 문장의 모순을 가리지 않는다.
    ("FAIL", "핵심 지표 Accuracy 가 미달했다. 그러나 종합 판정은 합격이다."),
    # 판정 불가를 설명하며 '모두 충족'을 쓰는 것은 참이지만, 설명 없는 '모두 충족'은 모순이다.
    ("INCONCLUSIVE", "모든 지표가 목표를 충족하였다."),
    # 불합격에서 핵심 지표가 모두 충족했다는 말은 모순이다(그 외·보조 지표만 허용).
    ("FAIL", "핵심 지표는 기준을 모두 충족하였다."),
    # 조건부 합격에서 주어가 지표 전부면 모순이다.
    ("CONDITIONAL_PASS", "모든 지표가 기준을 충족하였다."),
    # 부정이 멀리 떨어져 있으면 미달 주장이다.
    ("PASS", "Recall 이 기준에 미달하였으며 다른 문제는 없다."),
    # 합격에서 규칙 조건문이 아닌 불합격 주장.
    ("PASS", "Recall 지표는 불합격이다."),
])
def test_more_contradictions_are_flagged(verdict, text):
    assert find_verdict_contradictions([text], verdict) == [text]


@pytest.mark.parametrize("verdict, text", [
    ("INCONCLUSIVE", "목표값이 설정된 지표는 모두 충족했으나 핵심 지표가 산출되지 않아 판정할 수 없다."),
    ("CONDITIONAL_PASS", "모든 지표가 충족된 것은 아니며 보조 지표 Recall 이 미달했다."),
    ("PASS", "핵심 지표가 하나라도 목표에 미달하면 최종 불합격이지만, 이 성적서의 핵심 지표는 모두 충족했다."),
    ("FAIL", "보조 지표는 모두 기준을 충족했으나 핵심 지표 F1 Score 가 미달했다."),
])
def test_narrowed_or_negated_sentences_are_allowed(verdict, text):
    assert find_verdict_contradictions([text], verdict) == []
