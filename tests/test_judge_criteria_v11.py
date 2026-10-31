"""tests/test_judge_criteria_v11.py — 심판 기준 1.1 고침(criteria/CHANGELOG.md "## 1.1-dev" 1~5).

V2 1차(재현성 5회 모두 발급 불가)의 심판 쪽 오판을 고친 것마다 맞는 쪽과 틀린 쪽을 함께 본다(고침이 너무 넓어 진짜 위반까지
통과시키지 않게). 문장은 모두 지어낸 것이다 — 봉인 서술·판 3 발급본·예시본 글을 쓰지 않는다(검증 README 오염 방지 규칙 2·3).
"""
from __future__ import annotations

import pytest

import json
from types import SimpleNamespace as M

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
    # 근거 범위(문장)는 그 주장 문장이다 — 1.5 부터 옮긴 대조는 문장 근거가 있어야 받는다
    return fa.judge_answer(c, a, SOURCES, [], [], Piece(PIECE.id, PIECE.slot, PIECE.place, PIECE.source, c.quote + "."))


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


@pytest.mark.parametrize("named, verdict", [
    ("학습 때와 다른 조건의 데이터", "match"),
    ("학습 데이터와 다른 출처", "match"),
    ("학습 때와 같은 조건의 데이터", "violation"),  # 의뢰자 답은 '다른 출처'
    ("다른 곳에서 모았지만 같은 조건", "unverifiable"),  # 두 값의 말이 같이 있으면 모호
    ("다른 출처지만 같은 곳", "unverifiable"),  # 글 겹침 대조로 넘어가 맞다가 되지 않는다(1.1 검토)
])
def test_client_option_paraphrase_matches_only_its_own_option(named, verdict):
    v = judge(claim("C1", "name", named), ans(name_field="client.test_data.학습 데이터와의 관계", named=named))
    assert v.verdict == verdict


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


@pytest.mark.parametrize("text, want", [("클래스 7 쪽 예측은 130건", "130"), ("레이블 14 의 표본", None),
                                        ("Class 9 는 약하다", None), ("F1 0.61", "0.61"), ("오류 40건", "40")])
def test_numbers_inside_names_are_not_quantities(text, want):
    assert first_quantity(text) == want


def test_number_claim_without_quantity_is_unverifiable_not_violation():
    q = "클래스 7 예측 건수는 실제 건수를 웃돈다"
    assert judge(claim("C1", "number", q), ans(expr="{facts.prediction_bias.7.predicted}")).verdict == "unverifiable"


@pytest.mark.parametrize("relation, verdict", [("gt", "match"), ("lt", "violation")])
def test_number_claim_with_two_sides_is_judged_as_comparison(relation, verdict):
    q = "클래스 7 예측 건수는 실제 건수를 웃돈다"
    v = judge(claim("C1", "number", q), ans(expr="{facts.prediction_bias.7.predicted}",
                                            expr_b="{facts.prediction_bias.7.actual}", relation=relation))
    assert v.verdict == verdict and v.detail["as"] == "comparison"


# ── 4. 추론 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("sentence, reason", [
    ("다음 평가 전에 라벨을 다시 맞춰야 한다.", "normative"),
    ("이 값은 같은 지점 자료에 한정된 결과로 읽어야 한다.", "normative"),  # 읽는 법 안내는 규범(1.7)
    ("이 0.62 는 클래스 3 을 배우지 못했다는 뜻으로 읽어야 한다.", None),  # 해석을 감싼 꼴은 규범이 아니다(1.5 검토 P2)
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
    c = claim("C1", "inference", "F1 Score 는 목표 0.8 에 못 미친다")
    v = judge(c, ans(expr="{metric.M2.value}", expr_b="0.8", relation=relation, inference_basis="data"))
    assert v.verdict == verdict and v.detail["as"] == "comparison"


def test_inference_restating_client_input_is_judged_as_name():
    c = claim("C1", "inference", "평가 데이터는 학습 데이터와 다른 출처이다")
    ok = judge(c, ans(name_field="client.test_data.학습 데이터와의 관계", named="다른 출처", inference_basis="client_input"))
    bad = judge(c, ans(name_field="client.test_data.학습 데이터와의 관계", named="같은 출처", inference_basis="client_input"))
    assert (ok.verdict, bad.verdict) == ("match", "inference_violation")  # 문장에 없는 이름은 받지 않는다(1.5)
    assert bad.detail["remap_refused"] == "unanchored"


@pytest.mark.parametrize("a", [
    ans(inference_basis="data"),  # ③ 대조 칸 없음
    ans(expr="{없는.칸}", expr_b="{metric.M2.value}", relation="lt", inference_basis="data"),  # 대조할 수 없음
    ans(resolution="no_data", expr="{metric.M2.value}", expr_b="0.8", relation="lt"),  # resolved 가 아니면 옮기지 않음
])
def test_unhedged_inference_without_usable_mapping_still_violates(a):
    v = judge(claim("C1", "inference", "이 모델은 새 지점 자료에서도 잘 맞을 것이다"), a)  # 인과 낱말 없는 단정
    assert v.verdict == "inference_violation"


async def test_agent_schema_offers_not_claim_and_prompt_explains_three_ways():
    schema = fa.answer_schema(["C1"])
    assert "not_claim" in schema["properties"]["claims"]["items"]["properties"]["resolution"]["enum"]
    prompt = criteria.load_text("prompts/faithfulness.md")
    for word in ("not_claim", "①", "②", "③", "전체 표본 수가 아니다", "이름에 붙은 숫자", "판정 기준값 칸(rule.*", "facts.declared_average 는"):
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



# ── 1.1 검토 반영(P1·P2) ──────────────────────────────────────────────


@pytest.mark.parametrize("quote, verdict", [("클래스 7 예측은 130건으로 실제 건수를 웃돈다", "match"),
                                            ("클래스 7 예측은 150건으로 실제 건수를 웃돈다", "violation")])
def test_number_answered_as_comparison_still_checks_its_quantity(quote, verdict):
    v = judge(claim("C1", "number", quote), ans(expr="{facts.prediction_bias.7.predicted}",
                                                expr_b="{facts.prediction_bias.7.actual}", relation="gt"))
    assert v.verdict == verdict


def test_causal_inference_is_not_remapped_to_a_true_comparison():
    c = claim("C1", "inference", "클래스 7 의 성능이 낮은 것은 표본이 적기 때문이다")
    v = judge(c, ans(expr="{distribution.class_distribution.7}", expr_b="{distribution.class_distribution.8}", relation="lt"))
    assert v.verdict == "inference_violation" and v.detail["remap_refused"] == "causal"


@pytest.mark.parametrize("quote, verdict", [
    ("경계 오류는 두 확률이 비슷한 경우를 뜻한다", "not_claim"),
    ("클래스 7 의 성능이 낮은 것은 표본이 적기 때문이다", "inference_violation"),  # 뜻풀이 어미 없음
    ("이 결과는 표본이 적기 때문임을 뜻한다", "inference_violation"),  # 인과 낱말
    ("오류 40건은 두 확률이 비슷한 경우를 뜻한다", "inference_violation"),  # 숫자가 있는 문장은 이 데이터에 대한 말
])
def test_not_claim_is_accepted_only_for_definitions(quote, verdict):
    piece = Piece("slot:7.1", "7.1", "5.2", "ai", quote + ".")
    v = fa.judge_answer(claim("C1", "inference", quote), ans(resolution="not_claim"), SOURCES, [], [], piece)
    assert v.verdict == verdict


@pytest.mark.parametrize("actual, named, want", [
    ("fp", "거짓 음성이 거짓 양성보다 더 위험하다", None),  # 같은 칸 두 코드 — 예전엔 코드 글자 순서로 맞다
    ("fp", "FN 이 FP 보다 위험하다", None),
    ("all_equal", "모든 클래스가 동등하지 않다", False),  # 실제 값을 부정
    ("realtime", "실시간이 아닌 배치로 쓴다", False),
    ("batch", "실시간이 아닌 배치로 쓴다", True),
    ("fp", "실시간 서비스에서 FP 가 더 위험하다", True),  # 다른 칸 낱말은 보지 않는다
    ("all_equal", "두 오류의 비용은 동등하다고 보고 다수 클래스를 우선", False),
    ("FAIL", "종합 판정은 '최종 불합격'", True),
    ("PASS", "종합 판정은 '최종 불합격'", False),
])
def test_name_matching_uses_its_own_code_group_ties_and_negation(actual, named, want):
    assert fa.name_matches(actual, named) is want


def test_ambiguous_name_is_unverifiable_not_match():
    v = judge(claim("C1", "name", "FN 이 FP 보다 위험하다"),
              ans(name_field="facts.declarations.risk_direction", named="FN 이 FP 보다 위험하다"))
    assert v.verdict == "unverifiable"


def test_inference_remapped_to_name_records_it():
    c = claim("C1", "inference", "평가 데이터는 학습 데이터와 다른 출처이다")
    v = judge(c, ans(name_field="client.test_data.학습 데이터와의 관계", named="다른 출처"))
    assert v.detail["as"] == "name" and fa._count([v])["from_inference"] == 1


@pytest.mark.parametrize("own, table, text, counted", [
    (0, "0", "3", True), (1, "1", "4", True),  # 건수 칸 0·1 은 비율이 아니다
    (0.95, "95%", "0.5", True),  # 1 이하는 비율의 표기일 수 있는 크기 — 넘기지 않고 센다
    (0.95, "95%", "2", False),
])
def test_ratio_rule_only_for_real_ratio_fields(own, table, text, counted):
    found, _, _ = co.value_mismatches({"f": [(table, TABLE), (text, AI)]}, {"f": own})
    assert bool(found) is counted


@pytest.mark.parametrize("sentence, quote, reason", [
    ("라벨 오류 때문에 성능이 낮으며, 그 밖의 영향은 단정할 수 없다.", "라벨 오류 때문에 성능이 낮으며", None),
    ("바로 배포해야 하며 성능 문제는 없다.", "성능 문제는 없다", None),
    ("그래서 운영 자료를 대표하는지는 단정할 수 없다.", "운영 자료를 대표하는지는 단정할 수 없다", "scope"),
])
def test_normative_and_scope_words_count_only_in_the_quoted_clause(sentence, quote, reason):
    assert ex._not_inference(ex._clause_of(quote, sentence)) == reason


@pytest.mark.parametrize("text, want", [("레이블 14개 가운데 3개", "14"), ("클래스 3개 모두", "3"), ("클래스 3 은 약하다", None)])
def test_count_units_make_a_number_after_a_name_word_a_quantity(text, want):
    assert first_quantity(text) == want


def test_extract_instructions_carry_added_markers():
    assert "기 쉽" in ex.claim_rules_text()["inference_markers"]


def test_scorecard_reports_not_claim_and_remapped_counts():
    from app.judge import aggregate

    d = judge(claim("C1", "inference", "경계 오류는 두 확률이 비슷한 경우를 뜻한다"), ans(resolution="not_claim"))
    r = judge(claim("C2", "inference", "F1 Score 는 목표 0.8 에 못 미친다"), ans(expr="{metric.M2.value}", expr_b="0.8", relation="lt"))
    faith = fa.FaithResult([d, r], fa._count([d, r]))
    cons = M(counts={"checked": 0, "contradictions": 0, "by_type": {}}, contradictions=[])
    cov = M(counts={"applicable": 0, "described": 0, "not_applicable": 0}, items=[])
    rea = M(counts={"rules": 0, "violations": 0})
    card = aggregate.scorecard(faith, cons, cov, rea)["faithfulness"]
    assert (card["not_claim"], card["from_inference"], card["checked"]) == (1, 1, 1)


def test_extraction_filter_is_wired_to_the_quoted_clause():
    # 배선: 추출 결과 받기(_accept)가 절 범위로 거른다 — 앞 절의 '…해야 하며' 때문에 뒤 절의 단정이 버려지지 않는다
    piece = Piece("slot:8.2", "8.2", "5.1", "ai", "바로 배포해야 하며 성능 문제는 없다. 대표하는지는 단정할 수 없다.")
    raw = [{"kind": "inference", "quote": "성능 문제는 없다", "piece_id": piece.id},
           {"kind": "inference", "quote": "대표하는지는 단정할 수 없다", "piece_id": piece.id}]
    rejected: list[dict] = []
    got = ex._accept(raw, {piece.id: piece}, {}, [], 0, rejected)
    assert [c.quote for c in got] == ["성능 문제는 없다"] and [r["reason"] for r in rejected] == ["scope"]


@pytest.mark.parametrize("gap_high, verdict", [(0.75, "match"), (0.65, "violation")])
def test_interpretation_of_average_gap_is_checked_against_rule_threshold(gap_high, verdict):
    # 평균 차이를 풀이한 문장(1.1 사용자 결정): '크다'를 판정 기준값(0.1)과 비교 — 실제 차이가 작으면 위반으로 잡힌다
    fs = {**FS, "facts": {**FS["facts"], "average_gap": {"kind": "macro_vs_micro", "macro_f1": 0.5, "other_f1": 0.5,
                                                         "difference": round(gap_high - 0.6, 3)}}}
    src = Sources(fs, [], {}, CLIENT)
    assert "크다" in src.lookup("rule.imbalance_gap_threshold")["note"]
    q = "두 평균의 차이는 클래스 사이 성능 편차가 크다는 것을 보여 준다"
    v = fa.judge_answer(claim("C1", "inference", q),
                        ans(expr="{facts.average_gap.difference}", expr_b="{rule.imbalance_gap_threshold}", relation="ge"),
                        src, [], [], Piece("slot:7.1", "7.1", "5.2", "ai", q + "."))
    assert v.verdict == verdict


# ── 1.2(V2 1.1 1회차에서 막힌 것) ────────────────────────────────────


@pytest.mark.parametrize("sentence, reason", [
    ("자료가 한쪽으로 몰리는지 매주 확인할 것을 권고한다.", "normative"),
    ("이 오류가 적다고 모든 예측을 믿어도 된다는 뜻은 아니다.", "scope"),
    ("이 값이 낮다고 쓸모없다는 의미는 아니다.", "scope"),
])
def test_v12_recommendation_and_denial_endings(sentence, reason):
    assert ex._not_inference(sentence) == reason


def test_v12_text_field_lookup_carries_note():
    fs = {**FS, "facts": {**FS["facts"], "declared_average": {"class_priority": "larger_first", "average": "weighted", "f1": 0.5}}}
    got = Sources(fs, [], {}, CLIENT).lookup("facts.declared_average.average")
    assert got["value"] == "weighted" and "성적서 지표" in got["note"] and "평균이 아니다" in got["note"]


@pytest.mark.parametrize("avg, verdict", [("매크로 평균", "match"), ("가중 평균", "violation")])
def test_v12_metric_average_kind_is_checked_by_equal_values(avg, verdict):
    fs = {**FS, "class_averages": [{"label": "매크로 평균", "precision": 0.5, "recall": 0.5, "f1": 0.61, "support": 400},
                                   {"label": "가중 평균", "precision": 0.5, "recall": 0.5, "f1": 0.66, "support": 400}]}
    src = Sources(fs, [], {}, CLIENT)
    c = claim("C1", "inference", "이 성적서의 F1 은 클래스별 값을 같은 무게로 평균한 매크로 평균이다")
    v = fa.judge_answer(c, ans(expr="{metric.M2.value}", expr_b="{class_averages." + avg + ".f1}", relation="eq"), src, [], [], PIECE)
    assert v.verdict == verdict


@pytest.mark.parametrize("resolution", ["not_claim", "resolved", "no_data"])
def test_v12_inference_detail_records_agent_resolution(resolution):
    v = judge(claim("C1", "inference", "경계 오류는 두 확률이 비슷한 경우를 뜻한다"), ans(resolution=resolution))
    assert v.detail["resolution"] == resolution


# ── 1.3(V2 1.2 2회차) ────────────────────────────────────────────────


@pytest.mark.parametrize("actual, named, want", [
    ("default", "핵심 지표는 분류 유형의 기본값이다", True),
    ("client", "핵심 지표는 분류 유형의 기본값이다", False),
    ("client", "의뢰자가 지정한 핵심 지표", True),
    ("no_distortion", "불균형은 있으나 성능 왜곡은 관찰되지 않았다", True),
    ("affected", "불균형은 있으나 성능 왜곡은 관찰되지 않았다", False),  # 긴 이름이 짧은 이름을 덮는다
    ("affected", "불균형 때문에 지표 왜곡이 관찰되었다", True),
    ("not_provided", "학습 분포가 제공되지 않아 비교하지 않았다", True),
    ("no_target", "핵심 지표 목표값이 없어 판정 불가", True),
    ("weighted", "의뢰자 관점에 맞춘 마이크로 평균", False),
])
def test_v13_every_named_code_field_has_korean_names(actual, named, want):
    assert fa.name_matches(actual, named) is want


def test_v13_code_groups_cover_every_labelled_code_once():
    names = criteria.load_json("names.json")
    grouped = [c for g in names["code_groups"] for c in g]
    assert sorted(grouped) == sorted(names["code_labels"]) and len(grouped) == len(set(grouped))


@pytest.mark.parametrize("sentence, agent, verdict", [
    ("결정 경계 근처의 오류는 두 확률이 가까운 경우로, 모델이 두 클래스를 가르지 못했음을 뜻한다", "resolved", "not_claim"),
    ("결정 경계 근처의 오류는 두 확률이 가까운 경우로, 모델이 두 클래스를 가르지 못했음을 뜻한다", "not_claim", "not_claim"),
    ("이는 모델이 두 클래스를 가르지 못했음을 뜻한다", "resolved", "inference_violation"),  # 앞 말을 해석
    ("이는 모델이 두 클래스를 가르지 못했음을 뜻한다", "not_claim", "inference_violation"),
])
def test_v13_definition_is_decided_by_code_whatever_the_agent_says(sentence, agent, verdict):
    piece = Piece("slot:7.1", "7.1", "5.2", "ai", sentence + ".")
    v = fa.judge_answer(claim("C1", "inference", sentence), ans(resolution=agent), SOURCES, [], [], piece)
    assert v.verdict == verdict


# ── 1.4(V2 1.3 3회차) ────────────────────────────────────────────────


@pytest.mark.parametrize("quote, value, verdict", [
    ("가장 약한 클래스는 매크로 F1 보다 0.1 이상 낮다", 0.51, "match"),  # 경계를 지킴
    ("가장 약한 클래스는 매크로 F1 보다 0.1 이상 낮다", 0.1, "match"),
    ("가장 약한 클래스는 매크로 F1 보다 0.1 이상 낮다", 0.04, "violation"),  # 경계를 어김
    ("차이는 0.1 미만이다", 0.04, "match"),
    ("차이는 0.1 미만이다", 0.2, "violation"),
    ("차이는 0.1 이다", 0.51, "violation"),  # 경계 말이 없으면 같아야 한다
    ("비율은 5% 이하이다", 0.03, "match"),
    ("비율은 5% 이상이다", 0.08, "match"),  # 백분율 경계는 데이터를 백분율로도 본다
])
def test_v14_bound_numbers_are_checked_as_bounds(quote, value, verdict):
    fs = {**FS, "facts": {**FS["facts"], "weakest_class": {"label": "7", "f1": 0.1, "gap_from_macro": value}}}
    src = Sources(fs, [], {}, CLIENT)
    v = fa.judge_answer(claim("C1", "number", quote), ans(expr="{facts.weakest_class.gap_from_macro}"), src, [], [], PIECE)
    assert v.verdict == verdict


# ── 1.5 재확인(민서 결정 2026-10-31) ──────────────────────────────────

R_PIECES = [Piece("slot:8.2", "8.2", "5.1", "ai", "가상 결론. F1 Score 가 낮다. 이 결과로는 목표 수준에 이르렀다고 보기 어렵다."),
            Piece("slot:7.1", "7.1", "5.2", "ai", "가상 분석. 성능이 낮은 것은 라벨 품질 때문이다."),
            Piece("slot:9:6.1", "9", "6.1", "ai", "가상 권고 칸. 재확인과 상관없는 글이다.")]
GOAL = Claim("C1", "inference", "목표 수준에 이르렀다고 보기 어렵다", "slot:8.2", "8.2", "5.1", "ai", "", "", "", "", 0)
CAUSE = Claim("C2", "inference", "성능이 낮은 것은 라벨 품질 때문이다", "slot:7.1", "7.1", "5.2", "ai", "", "", "", "", 0)


def first_pass(*cids):
    return F.json_response({"claims": [{"claim_id": c, **ans()} for c in cids]})


def second(cid, **kw):
    return F.json_response({"claims": [{"claim_id": cid, **ans(**kw)}]})


async def run_recheck(claims, *replies):
    fake = F.FakeOpenAI(list(replies))
    return fake, await fa.judge_faithfulness(fake, R_PIECES, claims, SOURCES, [], [])


@pytest.mark.parametrize("relation, verdict", [("lt", "match"), ("gt", "violation")])
async def test_recheck_turns_blocking_inference_into_code_checked_comparison(relation, verdict):
    fake, res = await run_recheck([GOAL], first_pass("C1"),
                                  second("C1", expr="{metric.M2.value}", expr_b="{metric.M2.threshold}", relation=relation))
    v = res.verdicts[0]
    assert v.verdict == verdict and v.detail["recheck"]["first"] == "inference_violation" and res.counts["rechecked"] == 1
    assert len(fake.calls) == 2 and "faithfulness_recheck" in json.dumps(fake.calls[1]["text"], ensure_ascii=False)
    # 재확인 호출도 기록에 든다 — 장부 비용(core.JudgeRun.cost_usd)이 이 기록을 더한다
    assert [c["purpose"] for c in res.calls] == ["faithfulness", "faithfulness_recheck"]


async def test_recheck_sends_only_pieces_of_rechecked_claims():
    fake, _ = await run_recheck([GOAL], first_pass("C1"), second("C1"))
    sent = json.dumps(fake.calls[1]["input"], ensure_ascii=False)
    assert "목표 수준에 이르렀다고" in sent and "재확인과 상관없는" not in sent and "라벨 품질" not in sent
    assert "다시 묻는다" not in fake.calls[0]["instructions"] and "하나씩 다시 본다" in fake.calls[1]["instructions"]


async def test_recheck_without_mapping_keeps_the_block():
    _, res = await run_recheck([GOAL], first_pass("C1"), second("C1", inference_basis="data"))
    assert [v.verdict for v in res.verdicts] == ["inference_violation"] and len(res.violations()) == 1


async def test_recheck_failure_keeps_first_verdict_without_failing_the_module():
    _, res = await run_recheck([GOAL], first_pass("C1"), F.response([F.message_item("x", refusal=True)]))
    assert res.failure is None and res.verdicts[0].verdict == "inference_violation"
    assert res.verdicts[0].detail["recheck"] == {"failure": "refusal"}


async def test_causal_sentence_is_not_rechecked():
    fake, res = await run_recheck([CAUSE], first_pass("C2"))
    assert len(fake.calls) == 1 and res.verdicts[0].verdict == "inference_violation" and res.counts["rechecked"] == 0


async def test_no_recheck_when_nothing_blocks():
    fake, res = await run_recheck([GOAL], F.json_response({"claims": [{"claim_id": "C1", **ans(
        expr="{metric.M2.value}", expr_b="{metric.M2.threshold}", relation="lt")}]}))
    assert len(fake.calls) == 1 and res.counts["rechecked"] == 0 and res.verdicts[0].verdict == "match"


@pytest.mark.parametrize("sentence, reason", [
    ("이 점검은 낮은 성능의 원인을 확정하려는 것이 아니라 후보를 좁힌다.", "scope"),
    ("임계값을 0.3 으로 둔 것은 거짓 음성을 줄이기 위한 것이다.", None),  # 의도 단정 — '위한 것이다'로 버리지 않는다(1.5 검토)
])
def test_v15_purpose_and_not_intended_endings(sentence, reason):
    assert ex._not_inference(sentence) == reason


# ── 1.5 검토 반영 — 문장과 상관없는 대조로 통과시키지 않는다 ─────────


@pytest.mark.parametrize("quote, a", [
    ("이 모델은 현장에 그대로 배포해도 안정적으로 동작한다",
     ans(expr="{n_samples}", expr_b="{facts.error_totals.misclassified}", relation="gt")),  # 상관없는 참인 비교
    ("이 모델은 다른 지점에서도 잘 통한다",
     ans(name_field="client.test_data.학습 데이터와의 관계", named="다른 출처")),  # 칸 값을 베낀 이름
    ("F1 Score 가 높아 그대로 써도 된다", ans(expr="{metric.M2.value}", expr_b="0.1", relation="gt")),  # 문장에 없는 숫자
    ("F1 Score 가 높아 그대로 써도 된다", ans(expr="{metric.M2.value}", expr_b="{metric.M2.threshold}", relation="lt")),  # 기준 낱말 없음
])
def test_unanchored_remap_is_refused_and_blocks(quote, a):
    v = judge(claim("C1", "inference", quote), a)
    assert v.verdict == "inference_violation" and v.detail["remap_refused"] == "unanchored"


def test_anchor_context_reaches_two_sentences_back_but_not_three():
    text = "F1 Score 는 0.61 이다. 가상 문장. 다른 문장. 따라서 판정은 그대로다."
    piece = Piece("slot:8.2", "8.2", "5.1", "ai", text)
    c = Claim("C1", "inference", "따라서 판정은 그대로다", piece.id, "8.2", "5.1", "ai", "", "", "", "", 0)
    assert "F1" not in fa.context_of(c.quote, piece)
    piece2 = Piece("slot:8.2", "8.2", "5.1", "ai", "F1 Score 는 0.61 이다. 다른 문장. 따라서 판정은 그대로다.")
    assert "F1 Score" in fa.context_of(c.quote, piece2)


def test_anchor_context_includes_the_previous_sentence():
    piece = Piece("slot:8.2", "8.2", "5.1", "ai", "F1 Score 는 0.61 이다. 따라서 판정은 그대로다.")
    c = Claim("C1", "inference", "따라서 판정은 그대로다", piece.id, "8.2", "5.1", "ai", "", "", "", "", 0)
    v = fa.judge_answer(c, ans(expr="{metric.M2.value}", expr_b="0.61", relation="eq"), SOURCES, [], [], piece)
    assert v.verdict == "match"


def test_short_label_anchor_needs_word_boundary():
    assert fa.field_anchored("per_class.7.f1", "클래스 7 의 F1", SOURCES)
    assert not fa.field_anchored("per_class.7.f1", "표본 170건", SOURCES)


@pytest.mark.parametrize("sentence", ["재현율이 낮은 것은 라벨 오류에서 비롯된다", "성능 저하는 표본 부족에 기인한다",
                                      "조명 변화가 정확도를 떨어뜨렸다", "오류는 장비 차이에 의한 것이다"])
def test_more_causal_words_block_remap(sentence):
    v = judge(claim("C1", "inference", sentence), ans(expr="{metric.M2.value}", expr_b="{metric.M2.threshold}", relation="lt"))
    assert v.detail["causal"] is True and v.verdict == "inference_violation"


@pytest.mark.parametrize("sentence, quote", [
    ("원인을 확정하려는 것이 아니라 클래스 7 의 낮은 재현율은 라벨 오류에서 비롯된다.", "클래스 7 의 낮은 재현율은 라벨 오류에서 비롯된다"),
    ("다수 클래스만 맞히는 것은 정확도를 높이려는 것으로 이 모델은 현장에 쓸 수 없다.", "이 모델은 현장에 쓸 수 없다"),
])
def test_assertion_after_not_intended_clause_is_kept(sentence, quote):
    assert ex._not_inference(ex._clause_of(quote, sentence)) is None


async def test_recheck_failure_costs_zero_and_is_counted():
    from app.judge import aggregate

    fake = F.FakeOpenAI([first_pass("C1"), TimeoutError("느림")])
    res = await fa.judge_faithfulness(fake, R_PIECES, [GOAL], SOURCES, [], [])
    assert res.verdicts[0].verdict == "inference_violation" and res.counts["recheck_failed"] == 1
    failed = [c for c in res.calls if c["purpose"] == "faithfulness_recheck"]
    assert failed and all(c["cost_usd"] == 0.0 and c["cost_unknown"] for c in failed)
    card = aggregate.scorecard(res, M(counts={"checked": 0, "contradictions": 0, "by_type": {}}, contradictions=[]),
                               M(counts={"applicable": 0, "described": 0, "not_applicable": 0}, items=[]),
                               M(counts={"rules": 0, "violations": 0}))["faithfulness"]
    assert (card["rechecked"], card["recheck_failed"], card["recheck_changed"]) == (1, 1, 0)


async def test_recheck_keeps_first_detail_sums_tools_and_uses_short_deadline():
    t = [100.0]
    fake = F.FakeOpenAI([first_pass("C1"),
                         F.response([F.function_call_item("lookup_value", {"field": "metric.M2.threshold"})]),
                         second("C1", expr="{metric.M2.value}", expr_b="{metric.M2.threshold}", relation="lt")])
    res = await fa.judge_faithfulness(fake, R_PIECES, [GOAL], SOURCES, [], [], deadline=1000.0, clock=lambda: t[0])
    v = res.verdicts[0]
    assert v.verdict == "match" and v.detail["recheck"]["first_detail"]["resolution"] == "resolved"
    assert res.tool_calls == 1 and res.counts["recheck_changed"] == 1
    # 재확인 호출의 시간 한도는 지금 + 60초(전체 마감 1000 이 아니라)
    assert fake.options[-1]["timeout"] <= fa.RECHECK_SECONDS + 1e-6


def test_every_field_family_has_an_anchor():
    # 칸이 근거 표에 없으면 그 칸으로 옮긴 대조는 늘 거부된다 — 맞는 서술이 막힌다(1.5: facts.declared_average 가 빠져 있었다).
    # 조회 자료의 모든 칸이 근거 낱말을 하나 이상 갖는지 본다(시험용 사실 시트로).
    from fnmatch import fnmatch

    t = criteria.load_json("remap_anchors.json")
    fs = {**FS, "class_averages": [{"label": "가중 평균", "precision": 0.5, "recall": 0.5, "f1": 0.6, "support": 400}],
          "confusion": {"labels": ["7", "8"], "matrix": [[90, 10], [20, 280]]},
          "facts": {**FS["facts"], "declared_average": {"class_priority": "larger_first", "average": "weighted", "f1": 0.5},
                    "precision_recall_gap": {"precision": 0.5, "recall": 0.4}, "n_classes": 2, "positive_class": "8"}}
    src = Sources(fs, [], {"confusion": {"total": 400}}, CLIENT)

    def has_anchor(n):
        return (any(fnmatch(n, p) for p in t["threshold_fields"]) or n.startswith("rule.") or n.startswith(("metric.", "benchmark."))
                or any(n.startswith(p) for p in t["label_fields"] + t["pair_fields"]) or n.startswith("class_averages.")
                or any(fnmatch(n, p) for p, w in t["patterns"].items() if w))
    missing = sorted(n for n in [*src.numbers, *src.texts] if not has_anchor(n))
    assert missing == [], missing


def test_declared_average_remap_is_anchored_by_priority_words():
    q = "다수 클래스를 우선하는 관점에서도 판정 결과는 달라지지 않는다"
    piece = Piece("slot:8.2", "8.2", "5.1", "ai", "그에 맞춘 평균 F1 은 0.5 이다. F1 Score 의 목표에 못 미친다. 따라서 " + q + ".")
    fs = {**FS, "facts": {**FS["facts"], "declared_average": {"class_priority": "larger_first", "average": "weighted", "f1": 0.5}}}
    v = fa.judge_answer(Claim("C1", "inference", q, piece.id, "8.2", "5.1", "ai", "", "", "", "", 0),
                        ans(expr="{facts.declared_average.f1}", expr_b="{metric.M2.threshold}", relation="lt"),
                        Sources(fs, [], {}, CLIENT), [], [], piece)
    assert v.verdict == "match"


def test_pair_field_is_anchored_by_either_class_name():
    assert fa.field_anchored("confusion.7>8", "실제 7 을 8 로 본 것이 20건이다", SOURCES)
    assert not fa.field_anchored("confusion.7>8", "이 모델은 안정적이다", SOURCES)


# ── 1.6(V2 1.5 R-0038 — 숫자가 여럿인 문장에서 다른 숫자를 대조) ─────

M_PIECE = Piece("slot:7.1", "7.1", "5.2", "ai", "가상. F1 Score 는 지금 0.61 에서 그 임계값에서는 0.8 이 된다.")
TWO = Claim("C1", "number", "F1 Score 는 지금 0.61 에서 그 임계값에서는 0.8 이 된다", M_PIECE.id, "7.1", "5.2", "ai",
            "metric.M2.value", "", "0.8", "", 0)


def test_v16_listing_tells_which_number_to_check():
    assert "대조할 숫자 0.8" in fa._listing([TWO])
    assert "대조할 숫자" not in fa._listing([claim("C2", "inference", "가상 문장")])


def test_v16_value_of_another_number_in_the_sentence_is_misaimed():
    v = fa.judge_answer(TWO, ans(expr="{metric.M2.value}"), SOURCES, [], [], M_PIECE)  # 0.61 의 칸
    assert v.verdict == "violation" and v.detail["misaimed"] is True
    v = fa.judge_answer(TWO, ans(expr="{metric.M2.threshold}"), SOURCES, [], [], M_PIECE)  # 0.8 — 맞음
    assert v.verdict == "match"
    wrong = Claim("C1", "number", "F1 Score 는 0.7 이다", M_PIECE.id, "7.1", "5.2", "ai", "", "", "0.7", "", 0)
    v = fa.judge_answer(wrong, ans(expr="{metric.M2.value}"), SOURCES, [], [], M_PIECE)  # 진짜 틀린 숫자
    assert v.verdict == "violation" and not v.detail.get("misaimed")


async def test_v16_misaimed_number_is_rechecked_and_code_judges_again():
    fake = F.FakeOpenAI([F.json_response({"claims": [{"claim_id": "C1", **ans(expr="{metric.M2.value}")}]}),
                         F.json_response({"claims": [{"claim_id": "C1", **ans(expr="{metric.M2.threshold}")}]})])
    res = await fa.judge_faithfulness(fake, [M_PIECE], [TWO], SOURCES, [], [])
    assert res.verdicts[0].verdict == "match" and res.counts["rechecked"] == 1
    assert "대조할 숫자 0.8" in json.dumps(fake.calls[1]["input"], ensure_ascii=False)


async def test_v16_misaimed_twice_stays_blocked_and_real_wrong_number_is_not_rechecked():
    fake = F.FakeOpenAI([F.json_response({"claims": [{"claim_id": "C1", **ans(expr="{metric.M2.value}")}]}),
                         F.json_response({"claims": [{"claim_id": "C1", **ans(expr="{metric.M2.value}")}]})])
    res = await fa.judge_faithfulness(fake, [M_PIECE], [TWO], SOURCES, [], [])
    assert res.verdicts[0].verdict == "violation"
    wrong = Claim("C1", "number", "F1 Score 는 0.7 이다", M_PIECE.id, "7.1", "5.2", "ai", "", "", "0.7", "", 0)
    fake = F.FakeOpenAI([F.json_response({"claims": [{"claim_id": "C1", **ans(expr="{metric.M2.value}")}]})])
    res = await fa.judge_faithfulness(fake, [M_PIECE], [wrong], SOURCES, [], [])
    assert len(fake.calls) == 1 and res.verdicts[0].verdict == "violation"


def test_v16_prompts_say_check_the_listed_number():
    assert "대조할 숫자" in criteria.load_text("prompts/faithfulness.md")
    assert "대조할 숫자" in criteria.load_text("prompts/faithfulness_recheck.md")


# ── 1.7(V2 1.6 R-0039 — '판정의 여유가 작다') ─────────────────────────


def _m(mid, status, near):
    return M(metric_id=mid, display_name=mid, value=0.98, threshold=0.97, status=status, is_core=True, near_target=near)


@pytest.mark.parametrize("polarity, quote, near, want", [
    ("negative", "Recall 은 판정의 여유가 작다", True, "ok"),
    ("negative", "Recall 은 판정의 여유가 작다", False, "violation"),  # 여유가 넉넉한데 작다고 함
    ("positive", "Recall 은 판정 여유가 충분하다", False, "ok"),
    ("positive", "Recall 은 판정 여유가 충분하다", True, "violation"),
    ("negative", "Recall 은 목표에 근접해 겨우 넘었다", True, "ok"),
    ("negative", "Recall 성능이 미흡하다", True, "violation"),  # 여유 낱말 없음 — 보통 규칙(목표 충족인데 부정)
])
def test_v17_margin_degree_is_judged_by_near_target(polarity, quote, near, want):
    from app.judge.criteria.rules_degree import degree_verdict

    assert degree_verdict(polarity, "M3", quote, [_m("M3", "pass", near)], []) == want


def test_v17_margin_without_flag_falls_back_to_usual_rule():
    from app.judge.criteria.rules_degree import degree_verdict

    m = M(metric_id="M3", display_name="Recall", value=0.98, threshold=0.97, status="pass", is_core=True)
    assert degree_verdict("negative", "M3", "판정의 여유가 작다", [m], []) == "violation"
    failed = M(metric_id="M3", display_name="Recall", value=0.9, threshold=0.97, status="fail", is_core=True)
    assert degree_verdict("negative", "M3", "판정의 여유가 작다", [failed], []) == "ok"  # 미달 지표에 부정은 보통 규칙대로 허용


# ── 1.8(V2 1.7 R-0041 — 지표 상태 pass ↔ '충족') ─────────────────────


@pytest.mark.parametrize("actual, named, want", [
    ("pass", "Hamming Loss 와 Jaccard Index 는 모두 목표를 충족했다", True),
    ("fail", "모두 목표를 충족했다", False),
    ("fail", "목표를 충족하지 못했다", True),
    ("pass", "목표를 충족하지 못했다", False),
    ("pass", "목표를 충족하지 않았다", False),
    ("info", "목표값이 없는 참고 지표", True),
    ("argmax", "가장 큰 클래스를 예측으로 골랐다", True),
    ("threshold", "레이블마다 결정 임계값을 적용했다", False),
    ("threshold", "확률에 결정 임계값을 적용해 예측을 만들었다", True),
])
def test_v18_metric_status_and_derivation_codes(actual, named, want):
    assert fa.name_matches(actual, named) is want


def test_v18_every_enumerated_str_field_in_schema_is_named():
    # 스키마에서 값 목록(Literal 이거나 설명의 'a | b')을 가진 칸의 값이 이름 표에 모두 있다 — 1.3 은 Literal 만 보아 놓쳤다.
    # 서술에 이름으로 나오지 않는 칸(문제 목록·단위·경계 메모)만 경로로 뺀다.
    import re
    import typing

    from pydantic import BaseModel

    from app.narrative import schemas as S

    labels = criteria.load_json("names.json")["code_labels"]
    not_named = {".problems.kind", ".problems.priority", ".problems.recommend_section", ".error_ratios.unit",
                 ".error_ratios.boundary_note", ".facts.error_totals.unit", ".facts.prediction_bias.unit"}
    missing, seen = [], set()

    def walk(model, path):
        if model in seen:
            return
        seen.add(model)
        for name, f in model.model_fields.items():
            here = f"{path}.{name}"
            vals = []
            for t in [f.annotation, *typing.get_args(f.annotation)]:
                if typing.get_origin(t) is typing.Literal:
                    vals += list(typing.get_args(t))
            m = re.fullmatch(r"\s*(\w+(?:\s*\|\s*\w+)+)\s*(?:\(.*\))?\s*", f.description or "")
            if m:
                vals += [v.strip() for v in m.group(1).split("|")]
            if here not in not_named:
                missing.extend(f"{here}={v}" for v in vals if isinstance(v, str) and v not in labels)

            def inner(t):
                if isinstance(t, type) and issubclass(t, BaseModel):
                    walk(t, here)
                for a in typing.get_args(t) or []:
                    inner(a)
            inner(f.annotation)
    walk(S.FactSheet, "")
    assert missing == [], missing


# ── 1.9(V2 1.8 R-0043) ───────────────────────────────────────────────

N_PIECE = Piece("slot:9:6.2", "9", "6.2", "ai", "가상. 이번 시험에서 클래스 7로 예측된 표본이 실제 클래스 7 표본보다 많았다.")


@pytest.mark.parametrize("a, verdict", [
    (ans(expr="{facts.prediction_bias.7.predicted}"), "unverifiable"),  # 이름 속 숫자 '7' 은 수량이 아니다
    (ans(expr="{facts.prediction_bias.7.predicted}", expr_b="{facts.prediction_bias.7.actual}", relation="gt"), "match"),
    (ans(expr="{facts.prediction_bias.7.predicted}", expr_b="{facts.prediction_bias.7.actual}", relation="lt"), "violation"),
])
def test_v19_number_quote_that_is_only_a_class_name(a, verdict):
    c = Claim("C1", "number", "7", N_PIECE.id, "9", "6.2", "ai", "", "", "7", "", 0)
    assert fa.judge_answer(c, a, SOURCES, [], [], N_PIECE).verdict == verdict


def test_v19_count_after_name_is_still_a_quantity():
    piece = Piece("slot:7.1", "7.1", "5.2", "ai", "가상. 평가한 클래스 2개 가운데 하나가 약하다.")
    c = Claim("C1", "number", "2", piece.id, "7.1", "5.2", "ai", "", "", "2", "", 0)
    assert fa.printed_number(c, piece) == "2"


@pytest.mark.parametrize("expr, want", [
    ("metric.M2.value", "{metric.M2.value}"),
    ("metric.M2.value - metric.M2.threshold", "{metric.M2.value} - {metric.M2.threshold}"),
    ("{metric.M2.value}", "{metric.M2.value}"),
    ("0.8", "0.8"),
    ("없는.칸", "없는.칸"),
])
def test_v19_bare_field_names_are_braced(expr, want):
    assert fa.braced(expr, SOURCES) == want


def test_v19_recheck_answer_without_braces_is_still_judged():
    c = claim("C1", "inference", "F1 Score 는 목표 수준에 이르렀다고 보기 어렵다")
    v = judge(c, ans(expr="metric.M2.value", expr_b="metric.M2.threshold", relation="lt"))
    assert v.verdict == "match" and v.detail["expr"] == "{metric.M2.value}"


def test_v19_number_that_names_a_field_label_is_not_a_quantity_even_without_sentence():
    piece = Piece("slot:9:6.2", "9", "6.2", "ai", "가상 0.7 과 7건. 클래스 7 예측이 실제보다 많았다.")  # '7' 이 여러 번
    c = Claim("C1", "number", "7", piece.id, "9", "6.2", "ai", "", "", "7", "", 0)
    a = ans(expr="{facts.prediction_bias.7.predicted}", expr_b="{facts.prediction_bias.7.actual}", relation="gt")
    assert fa.judge_answer(c, a, SOURCES, [], [], piece).verdict == "match"
    wrong = Claim("C1", "number", "150건", piece.id, "9", "6.2", "ai", "", "", "150", "", 0)  # 진짜 수량은 그대로 대조
    assert fa.judge_answer(wrong, a, SOURCES, [], [], piece).verdict == "violation"


# ── 1.10(V2 1.9 R-0044) ──────────────────────────────────────────────


@pytest.mark.parametrize("actual, named, want", [
    ("not_provided", "같은 분포인지는 이 성적서에서 확인할 수 없다", None),  # 이름 표의 말이 없음 — 검증 불가
    ("not_provided", "학습 분포가 제공되지 않아 비교하지 않았다", True),
    ("compared", "학습 분포가 제공되지 않아 비교하지 않았다", False),  # 다른 값의 이름 — 위반
    ("batch", "batch 방식", True),  # 코드 글자 그대로 쓰면 맞다
])
def test_v110_code_value_without_any_named_label_is_unverifiable(actual, named, want):
    assert fa.name_matches(actual, named) is want


def test_v110_unverifiable_name_does_not_block():
    v = judge(claim("C1", "name", "같은 분포인지는 확인할 수 없다"),
              ans(name_field="facts.distribution_shift_status", named="확인할 수 없다"))
    assert v.verdict == "unverifiable"


# ── 1.11(검증 추출 미달 분석 — 되풀이 인용) ────────────────────────────


def test_v111_repeated_quote_is_kept_as_many_times_as_it_appears():
    piece = Piece("slot:7.1", "7.1", "5.2", "ai", "가상 FP 는 1건이다. 가상 FN 도 1건이다. 표본 171건.")
    raw = [{"kind": "number", "quote": "1건", "piece_id": piece.id, "number_text": "1"} for _ in range(3)]
    rejected: list[dict] = []
    got = ex._accept(raw, {piece.id: piece}, {}, [], 0, rejected)
    assert len(got) == 2  # '1건' 은 두 번(171건 안의 1건은 세지 않는다) — 세 번째는 버림
    assert [r["reason"] for r in rejected] == ["duplicate"]  # 버린 것도 기록에 남는다
    again = ex._accept(raw[:1], {piece.id: piece}, {}, got, 1, [])  # 자기 점검 회차에서 또 내도 넘치지 않는다
    assert again == []


@pytest.mark.parametrize("quote, text, n", [("1건", "1건과 1건, 171건", 2), ("0.984", "0.984 와 0.9841", 1), ("없음", "글", 1)])
def test_v111_occurrences(quote, text, n):
    assert ex.occurrences(quote, text) == n


# ── 1.12(1.11 R-0051 분석 — 권고 문장·기준선 숫자, 위험 방향 지표 칸) ──


@pytest.mark.parametrize("item, sentence, reason", [
    ({"kind": "number", "number_text": "0.970"}, "권고: 매주 Recall 이 목표 0.970 아래로 내려가는지 점검한다.", "recommendation"),
    ({"kind": "name"}, "권고: 표본을 늘려 다시 확인한다.", "recommendation"),
    ({"kind": "number", "number_text": "0.1"}, "두 값의 차이는 0.1 미만이다.", "threshold"),
    ({"kind": "number", "number_text": "95%"}, "Recall 의 95% 신뢰구간은 0.91~0.99 이다.", "threshold"),
    ({"kind": "number", "number_text": "10%p"}, "비율 차가 10%p 미만이다.", "threshold"),
    ({"kind": "number", "number_text": "0.352"}, "차이는 0.352 이상이다.", None),  # 기준값이 아닌 숫자는 주장
    ({"kind": "number", "number_text": "0.1"}, "F1 은 0.1 로 낮다.", None),  # 경계 말이 없으면 주장
    ({"kind": "number", "number_text": "0.970"}, "Recall 은 목표 0.970 을 충족했다.", None),  # 권고 문장이 아님
])
def test_v112_recommendation_and_threshold_numbers_are_not_claims(item, sentence, reason):
    quote = item.get("number_text") or "표본"
    assert ex._not_a_claim(item, quote, sentence) == reason


def test_v112_filters_are_wired_into_accept():
    piece = Piece("slot:9:6.2", "9", "6.2", "ai", "권고: 목표 0.970 아래로 내려가는지 점검한다. 차이는 0.1 미만이다. F1 은 0.61 이다.")
    raw = [{"kind": "number", "quote": q, "piece_id": piece.id, "number_text": q} for q in ("0.970", "0.1", "0.61")]
    rejected: list[dict] = []
    got = ex._accept(raw, {piece.id: piece}, {}, [], 0, rejected)
    assert [c.quote for c in got] == ["0.61"] and [r["reason"] for r in rejected] == ["recommendation", "threshold"]


def test_v112_core_metric_fits_declared_risk_via_risk_side_name():
    fs = {**FS, "f1_threshold_gain": {"risk_side": {"side": "fp", "metric_id": "M2", "display_name": "Precision"}}}
    src = Sources(fs, [], {}, CLIENT)
    assert "위험" in src.lookup("f1_threshold_gain.risk_side.display_name")["note"]
    q = "Precision 이 핵심 지표로 지정된 것은 이 용도와 부합한다"
    piece = Piece("slot:8.2", "8.2", "5.1", "ai", q + ".")
    ok = fa.judge_answer(claim("C1", "inference", q), ans(name_field="f1_threshold_gain.risk_side.display_name", named="Precision"),
                         src, [], [], piece)
    assert ok.verdict == "match"
