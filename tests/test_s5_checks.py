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
    # '모두'가 주어 뒤에 오는 꼴(S5 변이 검사에서 찾은 누락).
    ("FAIL", "판정 대상 지표는 모두 목표를 충족하였다."),
    ("CONDITIONAL_PASS", "지표들은 모두 기준을 통과하였다."),
    # 판정 불가 사유는 같은 문장에 있어야 한다 — 앞 문장의 사유가 뒤 문장의 '모두 충족'을 덮지 않는다.
    ("INCONCLUSIVE", "핵심 지표가 산출되지 않았다. 모든 지표가 목표를 충족하였다."),
])
def test_more_contradictions_are_flagged(verdict, text):
    assert find_verdict_contradictions([text], verdict) == [text]


@pytest.mark.parametrize("verdict, text", [
    ("INCONCLUSIVE", "목표값이 설정된 모든 지표는 목표를 충족했으나 핵심 지표가 산출되지 않아 판정할 수 없다."),
    ("INCONCLUSIVE", "목표값이 설정된 지표는 모두 충족했으나 핵심 지표가 산출되지 않아 판정할 수 없다."),
    ("CONDITIONAL_PASS", "모든 지표가 충족되지 않았으며 보조 지표 Recall 이 미달했다."),
    ("CONDITIONAL_PASS", "모든 지표가 통과한 것은 아니며 보조 지표 Recall 이 미달했다."),
    ("PASS", "핵심 지표가 하나라도 목표에 미달하면 최종 불합격이지만, 이 성적서의 핵심 지표는 모두 충족했다."),
    ("FAIL", "보조 지표는 모두 기준을 충족했으나 핵심 지표 F1 Score 가 미달했다."),
])
def test_narrowed_or_negated_sentences_are_allowed(verdict, text):
    assert find_verdict_contradictions([text], verdict) == []



# ── 칸별 검사(app.narrative.checks) 단위 ─────────────────────────────────────────

from app.narrative import checks  # noqa: E402
from app.narrative.schemas import FactSheet, MetricFact, PerClassFact  # noqa: E402


@pytest.mark.parametrize("text, expected", [
    ("F1 이 0.05 향상된다.", ["0.05"]),
    ("오류가 10% 줄어든다.", ["10%"]),
    ("1차 원인을 찾는다.", []),
    ("3개월 내 재평가한다.", []),
    ("2회 재평가로 확인한다.", []),
    ("0.5회 재평가한다.", ["0.5"]),   # 소수는 단위가 붙어도 예측 수치다
    ("5%차 점검", ["5%"]),            # 백분율도 마찬가지
    ("P1 문제의 원인이 분명해진다.", []),
    # S5 검토 checks-10
    ("상위 3개 클래스의 오분류를 줄여 1차 원인을 제거한다.", []),
    ("10분 내 응답을 회복한다.", []),
    ("처리 시간을 3시간 단축한다.", ["3"]),
    ("오분류 5회 감소", ["5"]),
])
def test_predicted_numbers(text, expected):
    assert checks.predicted_numbers(text, edition=3) == expected


def _fs(**kw):
    base = dict(verdict="PASS", score=100.0, metrics=[MetricFact(metric_id="M1", display_name="Accuracy", value=0.9, status="info")])
    return FactSheet(**{**base, **kw})


def test_snake_case_class_names_are_exempt_but_field_names_are_not():
    fs = _fs(per_class=[PerClassFact(label="late_payment", f1=0.5, support=10)])
    exempt = checks.name_exemptions(fs, {"model": {"모델명": "risk_model_v2"}})
    assert {"late_payment", "risk_model_v2"} <= exempt
    words = checks.code_words(["late_payment 클래스와 risk_model_v2 모델의 macro_f1 은 낮다. 판정은 PASS 이다."], exempt)
    assert words == ["PASS", "macro_f1"]


def test_section_9_is_empty_when_either_text_is_blank():
    parts = {"recommendation_narrative.data_quality": "데이터 권고다.", "recommendation_narrative.model_ops": " ",
             "recommendations": [{"problem_id": "P1", "category": "c", "action": "a", "expected_impact": "e"}]}
    r = checks.check_section("9", parts, whitelist=set(), verdict="PASS", conclusions=[], problem_ids=["P1"], exempt=set(), edition=3)
    assert r.format == ["empty"]
    parts["recommendation_narrative.model_ops"] = "모델 권고다."
    parts["recommendations"] = []
    r = checks.check_section("9", parts, whitelist=set(), verdict="PASS", conclusions=[], problem_ids=["P1"], exempt=set(), edition=3)
    assert r.format == ["empty"]



# ── S5 독립 검토(wf_a65c97fb-fec)가 재현한 문장 ──────────────────────────────────

REVIEW_CAUGHT = [
    # checks-1: '판정' 낱말 없는 단정
    ("FAIL", "본 모델은 최종 합격이다."),
    ("FAIL", "종합 판정: 합격."),
    ("FAIL", "종합 판정 결과 합격이다."),
    ("FAIL", "따라서 이 모델은 합격으로 결정되었다."),
    ("CONDITIONAL_PASS", "본 모델은 최종 불합격이다."),
    ("INCONCLUSIVE", "본 모델은 조건부 합격이다."),
    # checks-2: 지금 결과를 단정하는 '판정한다'
    ("PASS", "Recall 은 기준 미달로 판정한다."),
    ("PASS", "Recall 은 기준 미달로 판정합니다."),
    # checks-3: 주어가 좁혀지지 않은 '모두 충족'
    ("FAIL", "나머지 문제는 없으며 모든 지표가 기준을 충족했다."),
    ("FAIL", "그 외에는 모든 지표가 기준을 충족하였다."),
    ("CONDITIONAL_PASS", "핵심 지표를 포함한 모든 지표가 기준을 충족하였다."),
    ("CONDITIONAL_PASS", "핵심 지표뿐 아니라 모든 지표가 기준을 충족하였다."),
    ("INCONCLUSIVE", "모든 지표가 기준을 충족하여 판정 불가 사유가 없다."),
    ("INCONCLUSIVE", "판정 불가가 아니라 모든 지표가 기준을 충족하였다."),
    ("INCONCLUSIVE", "모든 지표가 기준을 충족하여 판정할 수 없는 사유가 없다."),
    # checks-4: 다른 동사·주어
    ("FAIL", "5개 지표 모두 목표값을 넘었다."),
    ("CONDITIONAL_PASS", "선택한 지표는 모두 목표를 달성했다."),
    ("FAIL", "모든 지표가 목표값을 상회했다."),
    ("CONDITIONAL_PASS", "모든 지표가 목표값을 넘었다."),
    ("FAIL", "전체 평가 항목이 기준을 충족하였다."),
]

REVIEW_ALLOWED = [
    # checks-5: 합격의 부정문
    ("PASS", "F1 Score 가 기준에 미달되지 않았다."),
    ("PASS", "어떤 지표도 미달하지는 않았다."),
    ("PASS", "F1 Score 는 목표값을 근소하게 넘어 미달은 아니다."),
    ("PASS", "목표를 충족하지 못한 핵심 지표는 없다."),
    ("PASS", "목표를 충족하지 못하는 지표는 없다."),
    ("PASS", "불합격 사유는 없다."),
    ("PASS", "미달한 지표가 0개이므로 종합 판정은 최종 합격이다."),
    # checks-6: 조건문·신뢰구간 끝
    ("PASS", "기준 미달 시 재학습을 권고한다."),
    ("PASS", "재평가에서 Recall 이 기준에 미달할 경우 운영을 중단한다."),
    ("PASS", "재평가에서 목표를 충족하지 못하는 경우 운영 투입을 보류한다."),
    ("PASS", "Recall 이 기준 미달로 바뀌는 경우에 대비한다."),
    ("PASS", "핵심 지표가 하나라도 미달할 경우 불합격으로 판정된다."),
    ("PASS", "Recall 의 신뢰구간 하한 0.78 은 목표값 0.8 에 미달한다."),
    # checks-7: 규칙 설명·나열·부정
    ("CONDITIONAL_PASS", "핵심 지표가 하나라도 미달하면 판정은 최종 불합격이고, 보조 지표만 미달하면 조건부 합격이다."),
    ("FAIL", "핵심 지표가 모두 충족되면 판정은 합격이다."),
    ("FAIL", "종합 판정은 합격, 조건부 합격, 최종 불합격, 판정 불가의 네 가지로 나뉜다."),
    ("CONDITIONAL_PASS", "보조 지표까지 모두 충족했다면 판정은 합격이었다."),
    ("FAIL", "합격으로 판정되었다고 보기 어렵다."),
    # checks-8: 하나를 뺀 나머지·관형절
    ("CONDITIONAL_PASS", "Recall 을 제외한 모든 지표가 기준을 충족했다."),
    ("CONDITIONAL_PASS", "보조 지표 Recall 만 제외하고 모든 지표가 기준을 충족하였다."),
    ("FAIL", "F1 Score 를 제외한 모든 지표가 기준을 충족했다."),
    ("FAIL", "목표 성능을 충족한 지표도 있으나 핵심 지표가 미달했다."),
    ("FAIL", "목표 성능을 달성한 지표는 Accuracy 하나뿐이다."),
]


@pytest.mark.parametrize("verdict, text", REVIEW_CAUGHT)
def test_review_contradictions_are_flagged(verdict, text):
    assert find_verdict_contradictions([text], verdict) == [text]


@pytest.mark.parametrize("verdict, text", REVIEW_ALLOWED)
def test_review_consistent_sentences_are_allowed(verdict, text):
    assert find_verdict_contradictions([text], verdict) == []



def test_verdict_codes_that_are_names_are_exempt_outside_verdict_context():
    """checks-9: 클래스 라벨 'PASS'·'FAIL', 모델명 'PASS-Net' 은 이름이다. '판정은 FAIL' 만 잡는다."""
    fs = _fs(per_class=[PerClassFact(label="FAIL", f1=0.7, support=10), PerClassFact(label="PASS", f1=0.8, support=10)])
    exempt = checks.name_exemptions(fs, {"model": {"모델명": "PASS-Net"}})
    assert checks.code_words(["FAIL 클래스의 재현율은 낮다. PASS-Net 모델이다."], exempt) == []
    assert checks.code_words(["종합 판정은 FAIL 이다."], exempt) == ["FAIL"]
    assert checks.code_words(["판정은 PASS"], set()) == ["PASS"]



def test_thousands_separator_numbers():
    """checks-12: '1,234.5' 는 숫자 하나, 붙여 쓴 '43,52' 는 숫자 둘."""
    from app.narrative.grounding import verify_grounding

    assert checks.d24_present("p95 응답 시간은 1,234.5ms 이다", (1234.5,), edition=3)
    assert verify_grounding(["행렬은 43,52 이다"], {"43", "52"}, edition=3).passed
    assert not verify_grounding(["행렬은 43,52 이다"], {"4352"}, edition=3).passed
    assert verify_grounding(["재표집 2,000회"], {"2000"}, edition=3).passed
