"""tests/test_judge_criteria_v11.py — 심판 기준 1.1 고침(criteria/CHANGELOG.md "## 1.1-dev" 1~5).

V2 1차(재현성 5회 모두 발급 불가)의 심판 쪽 오판을 고친 것마다 맞는 쪽과 틀린 쪽을 함께 본다(고침이 너무 넓어 진짜 위반까지
통과시키지 않게). 문장은 모두 지어낸 것이다 — 봉인 서술·판 3 발급본·예시본 글을 쓰지 않는다(검증 README 오염 방지 규칙 2·3).
"""
from __future__ import annotations

import pytest

from app.judge import consistency as co
from app.judge import criteria
from app.judge import extract as ex
from app.judge import faithfulness as fa
from app.judge.criteria.rules_numbers import first_quantity, fraction_value, matches
from app.judge.extract import Claim
from app.judge.reading import Piece
from app.judge.sources import Sources, field_catalog
from tests import judge_fakes as F

FS = {
    "n_samples": 400,
    "metrics": [{"metric_id": "M2", "display_name": "F1 Score", "value": 0.61, "threshold": 0.8, "status": "fail", "is_core": True}],
    "distribution": {"class_distribution": {"7": 100, "8": 300}, "imbalance_ratio": 3.0},
    "facts": {
        "error_totals": {"unit": "sample", "total": 400, "misclassified": 40},
        "declared_risk_errors": {"side": "fp", "count": 10, "total_errors": 40, "share": 0.25},
        "declarations": {"risk_direction": "fp", "class_priority": "larger_first", "usage_mode": "realtime"},
        "prediction_bias": {"rows": [{"label": "7", "predicted": 130, "actual": 100}]},
    },
}
CLIENT = {"test_data": {"학습 데이터와의 관계": "다른 출처 (의뢰자 제공: 다른 지점에서 모은 기록이다.)"}}
SOURCES = Sources(FS, [], {}, CLIENT)
PIECE = Piece("slot:7.1", "7.1", "5.2", "ai",
              "가상 문장 하나다. 경계 오류는 두 확률이 비슷한 경우를 뜻한다. 표본이 적은 칸은 값이 흔들리기 쉽다.")


def claim(cid, kind, quote, number_text=""):
    return Claim(cid, kind, quote, PIECE.id, PIECE.slot, PIECE.place, PIECE.source, "", "", number_text, "", 0)


def ans(**kw):
    base = {"resolution": "resolved", "expr": "", "expr_b": "", "relation": "", "name_field": "", "named": "", "metric_id": "",
            "inference_basis": ""}
    return {**base, **kw}


def judge(c, a):
    return fa.judge_answer(c, a, SOURCES, [], [], PIECE)


# ── 1. 한글 선언 ↔ 코드값 ───────────────────────────────────────────


@pytest.mark.parametrize("field, named, ok", [
    ("facts.declarations.usage_mode", "응답을 바로 돌려주는 실시간 방식", True),
    ("facts.declarations.usage_mode", "밤마다 모아 돌리는 배치 방식", False),
    ("facts.declarations.risk_direction", "거짓 양성이 더 위험하다", True),
    ("facts.declarations.risk_direction", "거짓 음성이 더 위험하다", False),
    ("facts.declarations.class_priority", "다수 클래스를 우선한다", True),
    ("facts.declarations.class_priority", "모든 클래스가 동등하다", False),
])
def test_declaration_codes_match_their_korean_names_and_only_those(field, named, ok):
    v = judge(claim("C1", "name", named), ans(name_field=field, named=named))
    assert v.verdict == ("match" if ok else "violation")


@pytest.mark.parametrize("named, ok", [
    ("학습 때와 다른 조건의 데이터", True),
    ("학습 데이터와 다른 출처", True),
    ("학습 때와 같은 조건의 데이터", False),  # 의뢰자 답은 '다른 출처'
    ("다른 곳에서 모았지만 같은 조건", False),  # 두 값의 말이 같이 있으면 맞지 않다
])
def test_client_option_paraphrase_matches_only_its_own_option(named, ok):
    v = judge(claim("C1", "name", named), ans(name_field="client.test_data.학습 데이터와의 관계", named=named))
    assert v.verdict == ("match" if ok else "violation")


# ── 2. 비율의 분모·분수 ───────────────────────────────────────────────


def test_field_notes_tell_total_samples_from_total_errors():
    cat = field_catalog(FS)
    assert "오류 건수가 아니다" in cat["facts.error_totals.total"].description
    assert "분모" in cat["facts.declared_risk_errors.total_errors"].description
    assert SOURCES.lookup("facts.error_totals.total")["note"].startswith("평가한 전체 표본 수")
    assert "note" not in SOURCES.lookup("metric.M2.value")  # 설명이 없는 칸은 그대로


def test_class_share_is_count_over_samples_and_absent_without_sample_count():
    assert SOURCES.number("distribution.class_share.7") == 0.25
    assert SOURCES.number("distribution.class_share.8") == 0.75
    assert "distribution.class_share.7" not in field_catalog({**FS, "n_samples": 0})
    v = judge(claim("C1", "number", "7 은 100건(25.0%)", "25.0%"), ans(expr="{distribution.class_share.7}"))
    assert v.verdict == "match"


@pytest.mark.parametrize("text, data, ok", [("4분의 1", 0.25, True), ("3분의 1", 0.3333333, True), ("3분의 1", 0.25, False),
                                            ("2분의 1", 0.5004, True), ("2분의 1", 0.501, False)])
def test_fraction_words_are_read_as_their_ratio(text, data, ok):
    assert fraction_value(text) is not None and matches(text, data) is ok


def test_fraction_claim_is_judged_on_its_ratio_not_on_the_first_digit():
    v = judge(claim("C1", "number", "전체 오류의 4분의 1", "4분의 1"), ans(expr="{facts.declared_risk_errors.share}"))
    assert v.verdict == "match"
    v = judge(claim("C1", "number", "전체 오류의 3분의 1", "3분의 1"), ans(expr="{facts.declared_risk_errors.share}"))
    assert v.verdict == "violation"


# ── 3. 비교를 숫자 하나로 ─────────────────────────────────────────────


@pytest.mark.parametrize("text, want", [("클래스 7로 예측된 표본이 130건", "130"), ("레이블 14 의 표본", None),
                                        ("Class 9 는 약하다", None), ("F1 0.61", "0.61"), ("오류 40건", "40")])
def test_numbers_inside_names_are_not_quantities(text, want):
    assert first_quantity(text) == want


def test_number_claim_without_quantity_is_unverifiable_not_violation():
    q = "클래스 7로 예측된 표본이 실제 클래스 7 표본보다 많다"
    assert judge(claim("C1", "number", q), ans(expr="{facts.prediction_bias.7.predicted}")).verdict == "unverifiable"


@pytest.mark.parametrize("relation, verdict", [("gt", "match"), ("lt", "violation")])
def test_number_claim_with_two_sides_is_judged_as_comparison(relation, verdict):
    q = "클래스 7로 예측된 표본이 실제 클래스 7 표본보다 많다"
    v = judge(claim("C1", "number", q), ans(expr="{facts.prediction_bias.7.predicted}",
                                            expr_b="{facts.prediction_bias.7.actual}", relation=relation))
    assert v.verdict == verdict and v.detail["as"] == "comparison"


# ── 4. 추론 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("sentence, reason", [
    ("이 값은 같은 지점 자료에 한정된 결과로 읽어야 한다.", "normative"),
    ("이 표본이 운영 자료를 대표하는지는 단정할 수 없다.", "scope"),
    ("새 자료에서도 같은지 알 수 없다.", "scope"),
    ("오류가 많은 것은 라벨 품질 때문이다.", None),  # 진짜 단정형 추론은 걸러지지 않는다
])
def test_added_normative_and_scope_endings_drop_the_inference(sentence, reason):
    assert ex._not_inference(sentence) == reason


def test_tendency_ending_counts_as_inference_marker():
    assert fa.inference_marked("표본이 적은 칸은 값이 흔들리기 쉽다.")
    assert fa.inference_marked("표본이 적어 흔들리기 쉬우므로 유의한다.")
    assert not fa.inference_marked("오류가 많은 것은 라벨 품질 때문이다.")


def test_claims_json_keeps_its_lists_and_additions_live_in_their_own_file():
    base = criteria.load_json("claims.json")
    for key in ("inference_markers", "normative_markers", "scope_markers"):
        merged = criteria.claim_markers(key)
        assert merged[: len(base[key])] == base[key] and len(merged) > len(base[key])
        assert not set(criteria.load_json("inference_rules.json")[key]) & set(base[key])


def test_inference_answered_not_claim_leaves_gate_and_denominator():
    c = claim("C1", "inference", "경계 오류는 두 확률이 비슷한 경우를 뜻한다")
    v = judge(c, ans(resolution="not_claim", inference_basis="other"))
    assert v.verdict == "not_claim"
    counts = fa._count([v])
    assert counts["not_claim"] == 1 and counts["inference_violation"] == 0 and counts["checked"] == 0


def test_not_claim_is_honoured_only_for_inference():
    v = judge(claim("C1", "number", "오류 40건", "40"), ans(resolution="not_claim", expr="{facts.error_totals.misclassified}"))
    assert v.verdict == "unverifiable"


@pytest.mark.parametrize("relation, verdict", [("lt", "match"), ("gt", "violation")])
def test_inference_restating_data_is_judged_by_code(relation, verdict):
    c = claim("C1", "inference", "가중 평균으로 보아도 목표에 못 미친다")
    v = judge(c, ans(expr="{metric.M2.value}", expr_b="0.8", relation=relation, inference_basis="data"))
    assert v.verdict == verdict and v.detail["as"] == "comparison"


def test_inference_restating_client_input_is_judged_as_name():
    c = claim("C1", "inference", "평가 데이터는 학습 데이터와 다른 출처이다")
    ok = judge(c, ans(name_field="client.test_data.학습 데이터와의 관계", named="다른 출처", inference_basis="client_input"))
    bad = judge(c, ans(name_field="client.test_data.학습 데이터와의 관계", named="같은 출처", inference_basis="client_input"))
    assert (ok.verdict, bad.verdict) == ("match", "violation")


@pytest.mark.parametrize("a", [
    ans(inference_basis="data"),  # ③ 대조 칸 없음
    ans(expr="{없는.칸}", expr_b="{metric.M2.value}", relation="lt", inference_basis="data"),  # 대조할 수 없음
    ans(resolution="no_data", expr="{metric.M2.value}", expr_b="0.8", relation="lt"),  # resolved 가 아니면 옮기지 않음
])
def test_unhedged_inference_without_usable_mapping_still_violates(a):
    v = judge(claim("C1", "inference", "오류가 많은 것은 라벨 품질 때문이다"), a)
    assert v.verdict == "inference_violation"


async def test_agent_schema_offers_not_claim_and_prompt_explains_three_ways():
    schema = fa.answer_schema(["C1"])
    assert "not_claim" in schema["properties"]["claims"]["items"]["properties"]["resolution"]["enum"]
    prompt = criteria.load_text("prompts/faithfulness.md")
    for word in ("not_claim", "①", "②", "③", "전체 표본 수가 아니다", "이름에 붙은 숫자"):
        assert word in prompt
    # 배선: 에이전트 답의 not_claim 이 판정까지 간다
    c = claim("C1", "inference", "경계 오류는 두 확률이 비슷한 경우를 뜻한다")
    fake = F.FakeOpenAI([F.json_response({"claims": [{"claim_id": "C1", **ans(resolution="not_claim")}]})])
    res = await fa.judge_faithfulness(fake, [PIECE], [c], SOURCES, [], [])
    assert [v.verdict for v in res.verdicts] == ["not_claim"] and res.violations() == []


# ── 5. 일관성 값 묶기 ─────────────────────────────────────────────────

AI = Piece("slot:8.2", "8.2", "5.1", "ai", "가상 문장.")
TABLE = Piece("m:table", "8", "4.2", "table", "가상 표.")


def test_count_glued_to_ratio_field_is_misassigned_not_a_contradiction():
    groups = {"rule.ci_level": [("95%", AI), ("2,000", AI)]}
    found, checked, skipped = co.value_mismatches(groups, {"rule.ci_level": 0.95})
    assert found == [] and checked == 1 and len(skipped) == 1


@pytest.mark.parametrize("pair", [(("95%", AI), ("90%", AI)), (("0.95", TABLE), ("0.91", AI))])
def test_real_value_mismatch_on_ratio_field_still_counts(pair):
    found, _, _ = co.value_mismatches({"rule.ci_level": list(pair)}, {"rule.ci_level": 0.95})
    assert [c.type for c in found] == ["value_mismatch"]


def test_large_number_on_count_field_is_not_skipped_by_ratio_rule():
    found, _, _ = co.value_mismatches({"facts.error_totals.misclassified": [("40", TABLE), ("2,000", AI)]},
                                      {"facts.error_totals.misclassified": 40})
    assert [c.type for c in found] == ["value_mismatch"]

