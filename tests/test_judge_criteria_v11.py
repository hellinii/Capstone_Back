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
    # 1.21: 실패하면 한 번 더 묻고, 그래도 실패하면 첫 판정(막음) 그대로
    refuse = F.response([F.message_item("x", refusal=True)])
    fake, res = await run_recheck([GOAL], first_pass("C1"), refuse, F.response([F.message_item("x", refusal=True)]))
    assert res.failure is None and res.verdicts[0].verdict == "inference_violation" and len(fake.calls) == 3
    assert res.verdicts[0].detail["recheck"] == {"failure": "refusal"}


async def test_recheck_is_retried_once_after_a_failure():
    # 1.21(V2 1.20 R-0060 — 재확인이 4턴 안에 답을 못 내 맞는 문장이 막혔다): 실패 뒤 한 번 더 물어 대조를 받으면 코드가 판정
    retry = F.json_response({"claims": [{"claim_id": "C1", **ans(expr="{metric.M2.value}", expr_b="{metric.M2.threshold}",
                                                                 relation="lt")}]})
    fake, res = await run_recheck([GOAL], first_pass("C1"), F.response([F.message_item("x", refusal=True)]), retry)
    assert len(fake.calls) == 3 and res.verdicts[0].verdict == "match" and res.counts["recheck_retried"] == 1


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
    # 기준선 숫자는 1.13 부터 _not_a_claim 이 아니라 drop_covered_thresholds 가 본다(비교 주장이 있을 때만 버림)
    ({"kind": "number", "number_text": "0.1"}, "두 값의 차이는 0.1 미만이다.", None),
    ({"kind": "number", "number_text": "0.352"}, "차이는 0.352 이상이다.", None),  # 기준값이 아닌 숫자는 주장
    ({"kind": "number", "number_text": "0.1"}, "F1 은 0.1 로 낮다.", None),  # 경계 말이 없으면 주장
    ({"kind": "number", "number_text": "0.970"}, "Recall 은 목표 0.970 을 충족했다.", None),  # 권고 문장이 아님
])
def test_v112_recommendation_and_threshold_numbers_are_not_claims(item, sentence, reason):
    quote = item.get("number_text") or "표본"
    assert ex._not_a_claim(item, quote, sentence) == reason


def test_v112_recommendation_filter_is_wired_into_accept():
    piece = Piece("slot:9:6.2", "9", "6.2", "ai", "권고: 목표 0.970 아래로 내려가는지 점검한다. 차이는 0.1 미만이다. F1 은 0.61 이다.")
    raw = [{"kind": "number", "quote": q, "piece_id": piece.id, "number_text": q} for q in ("0.970", "0.1", "0.61")]
    rejected: list[dict] = []
    got = ex._accept(raw, {piece.id: piece}, {}, [], 0, rejected)
    assert [c.quote for c in got] == ["0.1", "0.61"] and [r["reason"] for r in rejected] == ["recommendation"]


# ── 1.13(V2 1.12 R-0052) ─────────────────────────────────────────────

T_PIECE = Piece("slot:7.2", "7.2", "5.3", "ai", "불균형 비율은 3.0 이다. 두 값 모두 0.1 미만이다. 비율 차가 10%p 미만이다.")


def _c(cid, kind, quote, number_text=""):
    return Claim(cid, kind, quote, T_PIECE.id, "7.2", "5.3", "ai", "", "", number_text, "", 0)


def test_v113_threshold_number_dropped_only_when_a_comparison_covers_it():
    claims = [_c("C1", "number", "0.1", "0.1"), _c("C2", "comparison", "두 값 모두 0.1 미만이다"),
              _c("C3", "number", "10%p", "10%p"), _c("C4", "number", "3.0", "3.0")]
    rejected: list[dict] = []
    kept = ex.drop_covered_thresholds(claims, {T_PIECE.id: T_PIECE}, rejected)
    # 0.1 은 비교가 담고 있어 버림, 10%p 는 비교가 없어 남김(사람 정답지의 비교 주장과 짝), 3.0 은 기준선이 아님
    assert [c.id for c in kept] == ["C2", "C3", "C4"] and [r["reason"] for r in rejected] == ["threshold"]


@pytest.mark.parametrize("quote, want", [("0.1", True), ("10%p", True), ("3.0", False)])
def test_v113_is_threshold_number(quote, want):
    assert ex.is_threshold_number(_c("C1", "number", quote, quote), T_PIECE.text) is want


def test_v113_by_sentence_marks_sentences_without_claims():
    # 1.17: 1.13 의 '빈 문장 목록'을 문장마다 이미 뽑은 주장을 붙인 목록으로 바꿈 — 빈 문장은 여전히 드러난다
    sents = ex.sentences_of([T_PIECE])
    text = ex.by_sentence(sents, [_c("C1", "number", "3.0", "3.0")])
    assert "[s1] (slot:7.2) 불균형 비율은 3.0 이다.\n    이미 뽑은 것: number: 3.0" in text
    assert "[s2] (slot:7.2) 두 값 모두 0.1 미만이다.\n    이미 뽑은 것: (아직 없음)" in text
    full = ex.by_sentence(sents, [_c("C1", "number", "3.0"), _c("C2", "number", "0.1"), _c("C3", "number", "10%p")])
    assert "(아직 없음)" not in full


async def test_v113_self_check_round_shows_empty_sentences():
    item = {"kind": "number", "quote": "3.0", "piece_id": T_PIECE.id, "number_text": "3.0", "field": "", "field_b": "", "polarity": ""}
    sents = [{"sid": "s1", "n_claims": 1, "none_reason": ""}, {"sid": "s2", "n_claims": 0, "none_reason": "기준 설명"},
             {"sid": "s3", "n_claims": 0, "none_reason": "기준 설명"}]
    fake = F.FakeOpenAI([F.json_response({"sentences": sents, "claims": [item]}), F.json_response({"claims": []})])
    res = await ex.extract_claims(fake, [T_PIECE], {})
    assert res.failure is None and len(fake.calls) == 2
    second = fake.calls[1]["instructions"]
    assert "두 값 모두 0.1 미만이다.\n    이미 뽑은 것: (아직 없음)" in second
    assert "불균형 비율은 3.0 이다.\n    이미 뽑은 것: number: 3.0" in second


def test_v112_core_metric_fits_declared_risk_via_risk_side_name():
    fs = {**FS, "f1_threshold_gain": {"risk_side": {"side": "fp", "metric_id": "M2", "display_name": "Precision"}}}
    src = Sources(fs, [], {}, CLIENT)
    assert "위험" in src.lookup("f1_threshold_gain.risk_side.display_name")["note"]
    q = "Precision 이 핵심 지표로 지정된 것은 이 용도와 부합한다"
    piece = Piece("slot:8.2", "8.2", "5.1", "ai", q + ".")
    ok = fa.judge_answer(claim("C1", "inference", q), ans(name_field="f1_threshold_gain.risk_side.display_name", named="Precision"),
                         src, [], [], piece)
    assert ok.verdict == "match"


# ── 1.14(V2 1.13 R-0053 — '다른 지표는 판정 여유가 충분하여') ──────────


@pytest.mark.parametrize("flags, polarity, want", [
    ((True, False), "positive", "target_unclear"),  # 섞임 — 무엇을 가리키는지 모름
    ((True, False), "negative", "target_unclear"),
    ((False, False), "positive", "ok"), ((False, False), "negative", "violation"),
    ((True, True), "negative", "ok"), ((True, True), "positive", "violation"),
])
def test_v114_margin_without_metric_uses_all_judged_metrics(flags, polarity, want):
    from app.judge.criteria.rules_degree import degree_verdict

    ms = [M(metric_id=f"M{i}", display_name=f"M{i}", value=0.98, threshold=0.97, status="pass", is_core=i == 0, near_target=f)
          for i, f in enumerate(flags)]
    quote = "판정 여유가 충분하다" if polarity == "positive" else "판정 여유가 작다"
    assert degree_verdict(polarity, "", quote, ms, []) == want


# ── 1.15(V2 1.14 R-0054 — 추출 재현율) ───────────────────────────────


def test_v115_extract_prompt_names_often_missed_claims():
    text = ex.instructions({})
    for word in ("가장 높은·낮은·많은", "핵심 지표인 Recall", "판정 여유", "관찰되지 않았다"):
        assert word in text


def test_v115_sentence_list_is_not_cut_short():
    long = Piece("slot:9:6.2", "9", "6.2", "ai", " ".join(f"가상 문장 {i} 이다." for i in range(120)))
    sents = ex.sentences_of([long])
    assert len(sents) == 120 and "[s120] (slot:9:6.2) 가상 문장 119 이다." in ex.by_sentence(sents, [])
    assert "[s120] (slot:9:6.2) 가상 문장 119 이다." in ex.render_sentences(sents)


def test_v115_extraction_effort_stays_low_for_time():
    # medium 을 검토했으나 추출이 이미 약 85초(채점 전체 약 150초) — 두 배가 되면 발급 시간 한도에 걸린다(1.15 에서 되돌림)
    assert criteria.load_json("model.json")["extract_effort"] == "low"


# ── 1.16(V2 1.15 R-0056 — '수행 안 함' ↔ '수행하지 않았다') ─────────────


@pytest.mark.parametrize("actual, named, want", [
    ("수행 안 함 (의뢰자 제공: 그대로 학습했다.)", "편향 감소 노력을 수행하지 않았음", True),
    ("수행 안 함", "편향 감소 노력은 수행하지 못했다", True),
    ("수행 안 함", "편향 감소 노력을 수행했다", False),  # 부정 없음 — 반대 말
    ("측정 안 함", "일치도를 측정하지 않았다", True),
    ("측정 안 함", "일치도를 측정했다", False),
])
def test_v116_negated_client_option(actual, named, want):
    assert fa.client_text_matches(actual, named) is want


# ── 1.18(V2 1.17 R-0058 — 넓힌 추출이 뽑은 단위 이름·'○○되지 않음' 선택지) ─────────


@pytest.mark.parametrize("actual, named, want", [
    ("label_cell", "합친 수는 사람 단위가 아니라 레이블 칸(사람 수 × 레이블 수)", True),
    ("label_cell", "합계의 단위는 표본 단위이다", False),  # 다른 코드의 이름만
    ("sample", "합계의 단위는 표본 단위이다", True),
    ("sample", "모두 레이블 칸으로 센 값", False),
])
def test_v118_error_unit_code_names(actual, named, want):
    assert fa.name_matches(actual, named) is want


@pytest.mark.parametrize("actual, named, want", [
    ("제공되지 않음", "학습 데이터의 건수도 제공하지 않았다", True),
    ("제공되지 않음", "학습 데이터의 건수는 제공되지 않았다", True),
    ("제공되지 않음", "학습 데이터의 건수는 제공되지 않아", True),  # R-0058 C56 — 1.18 첫 고침이 이 꼴을 깼다(부정 목록에 '되지 않')
    ("제공되지 않음", "학습 데이터의 건수를 제공했다", False),  # 부정 없음 — 반대 말
    ("기록하지 않음", "보정 내역은 기록하지 않았다", True),
    ("수행 안 함", "편향 감소 노력을 수행하지 않았음", True),  # 1.16 그대로
])
def test_v118_negated_client_option_variants(actual, named, want):
    assert fa.client_text_matches(actual, named) is want


# ── 1.19(V2 1.18 R-0059 — 일반 말 '약한 레이블', 칸 이름을 부정한 의뢰자 선택지) ─────────


@pytest.mark.parametrize("named, want", [
    ("약한 레이블", None),  # 어떤 클래스도 지목하지 않음 — 검증 불가(위반 아님)
    ("가장 약한 클래스 7", "match"),
    ("가장 약한 클래스 8", "violation"),  # 다른 클래스를 지목하면 여전히 위반
])
def test_v119_generic_class_phrase_is_not_a_violation(named, want):
    fs = {**FS, "facts": {**FS["facts"], "weakest_class": {"label": "7", "f1": 0.3}}}
    src = Sources(fs, [], {}, CLIENT)
    assert fa.class_labels(src) == {"7", "8"} and src.texts["facts.weakest_class.label"] == "7"
    q = named + "의 부진"
    v = fa.judge_answer(claim("C1", "name", q), ans(name_field="facts.weakest_class.label", named=named), src, [], [],
                        Piece(PIECE.id, PIECE.slot, PIECE.place, PIECE.source, q + "."))
    assert v.verdict == want if want else v.verdict not in ("violation", "match")


@pytest.mark.parametrize("actual, named, label, want", [
    ("수행 안 함 (의뢰자 제공: 그대로 학습했다.)", "편향 감소 노력이 없었던 상태", "편향 감소 노력", True),
    ("수행 안 함", "편향 감소 노력을 기울였다", "편향 감소 노력", False),  # 칸 이름만 있고 부정이 없으면 반대 말
    ("수행 안 함", "편향 감소 노력이 없었던 상태", "", None),  # 칸 이름을 모르면 전처럼 글자 겹침으로(여기선 판단 못 함)
])
def test_v119_negated_option_by_field_label(actual, named, label, want):
    got = fa.client_text_matches(actual, named, label)
    assert got is want if want is not None else got is not True


def test_v119_field_label_reaches_client_match_through_judge():
    # 배선: 사실 일치가 의뢰자 칸 이름('편향 감소 노력')을 client_text_matches 에 넘긴다
    src = Sources(FS, [], {}, {"training_data": {"편향 감소 노력": "수행 안 함 (의뢰자 제공: 그대로 학습했다.)"}})
    q = "편향 감소 노력이 없었던 상태"
    v = fa.judge_answer(claim("C1", "name", q), ans(name_field="client.training_data.편향 감소 노력", named=q), src, [], [],
                        Piece(PIECE.id, PIECE.slot, PIECE.place, PIECE.source, q + "에서 학습했다."))
    assert v.verdict == "match"


# ── 1.21(V2 1.20 R-0060 — 가능성 절·계산 방식 풀이·클래스 문장의 정도 표현·재확인 넓힘, 마지막 기회) ─────────


def _judge_in(c, a, text, src=SOURCES, metrics=()):
    return fa.judge_answer(c, a, src, list(metrics), [], Piece(PIECE.id, PIECE.slot, PIECE.place, PIECE.source, text))


@pytest.mark.parametrize("text, quote, want", [
    ("한 주문에 상품이 여럿일 수 있어 비율을 모두 더하면 전체를 넘을 수 있다.", "비율을 모두 더하면 전체를 넘을 수 있다", "unverifiable"),
    ("비율을 모두 더하면 전체를 넘는다.", "비율을 모두 더하면 전체를 넘는다", "violation"),  # 단정이면 그대로 위반
])
def test_v121_possibility_clause_is_not_refuted_by_one_value(text, quote, want):
    a = ans(expr="{n_samples}", expr_b="{n_samples}", relation="gt")  # 400 > 400 아님
    assert _judge_in(claim("C1", "comparison", quote), a, text).verdict == want


def test_v121_denied_possibility_is_still_an_assertion():
    # '가능성은 낮다'는 단정 — 틀리면 막혀야 한다(판정 여유 문장)
    a = ans(expr="{n_samples}", expr_b="{n_samples}", relation="gt")
    text = "표본 수가 기준을 넘을 가능성은 낮다."
    assert _judge_in(claim("C1", "comparison", "표본 수가 기준을 넘을 가능성은 낮다"), a, text).verdict == "violation"


def test_v121_possibility_does_not_hide_a_true_match():
    a = ans(expr="{n_samples}", expr_b="1", relation="gt")
    text = "표본 수는 1 보다 클 수 있다."
    assert _judge_in(claim("C1", "comparison", "표본 수는 1 보다 클 수 있다"), a, text).verdict == "match"


@pytest.mark.parametrize("sentence, want", [
    ("가중 평균은 모든 매장의 주문을 합쳐 계산하므로 주문이 많은 매장의 영향이 크다.", True),
    ("단순 평균은 매장마다 같은 무게로 평균하므로 작은 매장의 몫도 같다.", True),
    ("이 결과는 매장별 값을 합쳐 계산하므로 0.71 이 나왔다.", False),  # 숫자가 있으면 이 데이터에 대한 말
    ("이는 모든 매장을 합쳐 계산하므로 큰 매장의 영향이 크다는 뜻이다.", False),  # 앞 말을 가리키며 시작 — 해석
])
def test_v121_method_explanation_counts_as_definition(sentence, want):
    assert fa.is_definition(sentence) is want
    assert fa.is_definition(sentence, "주문이 많은 매장의 영향이 크다") is want


def test_v121_method_explanation_inference_is_not_blocked():
    text = "가중 평균은 모든 매장의 주문을 합쳐 계산하므로 주문이 많은 매장의 영향이 크다."
    v = _judge_in(claim("C1", "inference", "주문이 많은 매장의 영향이 크다"), ans(inference_basis="data"), text)
    assert v.verdict == "not_claim"


@pytest.mark.parametrize("text, want", [
    ("가장 약한 클래스는 F1 0.05 인 클래스 7 이며, 평균보다 0.1 이상 낮아 운영 위험으로 다룬다.", "target_unclear"),
    ("전체 성능은 운영 위험이 크다.", "violation"),  # 클래스를 말하지 않으면 판정 지표로 — 핵심 지표가 목표를 넘었는데 '위험'
    ("전체 성능은 운영 위험이 크며 클래스 7 만 예외이다.", "violation"),  # 클래스가 뒤에 곁들여 나와도 첫 절이 전체를 말하면 그대로
])
def test_v121_degree_in_a_class_sentence_is_not_judged_against_core_metrics(text, want):
    m = M(metric_id="M2", display_name="F1 Score", value=0.9, threshold=0.8, status="pass", is_core=True, near_target=False)
    c = Claim("C1", "degree", "운영 위험", PIECE.id, PIECE.slot, PIECE.place, PIECE.source, "", "", "", "negative", 0)
    assert _judge_in(c, ans(metric_id=""), text, metrics=[m]).verdict == want


def test_v121_planted_degree_flip_still_blocks():
    # 결함 D01 꼴(목표 미달인데 '양호') — 가능성 낱말도 클래스 이름도 없으니 1.21 규칙에 걸리지 않고 위반
    m = M(metric_id="M1", display_name="Accuracy", value=0.5, threshold=0.9, status="fail", is_core=True, near_target=False)
    c = Claim("C1", "degree", "양호한 수준", PIECE.id, PIECE.slot, PIECE.place, PIECE.source, "", "", "", "positive", 0)
    assert _judge_in(c, ans(metric_id="M1"), "Accuracy 는 0.5 로 양호한 수준이다.", metrics=[m]).verdict == "violation"


@pytest.mark.parametrize("label, text, want", [
    ("7", "클래스 7 의 F1 은 0.05 이다", True), ("7", "F1 은 0.07 이다", False), ("7", "클래스 77 이다", False),
    ("Class1", "Class14 는 13건이다", False), ("Class1", "Class1 은 13건이다", True),
])
def test_v121_class_mention_boundaries(label, text, want):
    assert fa._mentions_label(label, text) is want


# 재확인은 넓히지 않는다(1.21-dev 에서 넓혔다가 독립 검토 P0 로 되돌림) — 이름·비교·정도 위반은 그대로 막는다
NAME_PIECE = [Piece("slot:7.1", "7.1", "5.2", "ai", "가상 분석. 의뢰자가 밝힌 사용 방식은 배치 방식이다.")]
NAME_CLAIM = Claim("C1", "name", "사용 방식은 배치 방식", "slot:7.1", "7.1", "5.2", "ai", "", "", "", "", 0)
MISREAD = dict(name_field="facts.declarations.usage_mode", named="배치 방식")  # 실제 realtime — 진짜 틀림


async def _run_name(*replies, pieces=NAME_PIECE, c=NAME_CLAIM):
    fake = F.FakeOpenAI(list(replies))
    return fake, await fa.judge_faithfulness(fake, pieces, [c], SOURCES, [], [])


async def test_v121_fact_violations_are_not_rechecked():
    # 재확인은 넓히지 않는다 — 1.24 보기 고르기가 '배치'(1번)를 고르면 실제 realtime 과 달라 그대로 위반
    pick = F.json_response({"items": [{"claim_id": "C1", "choice": "A"}]})
    fake, res = await _run_name(second("C1", **MISREAD), pick)
    assert len(fake.calls) == 2 and res.verdicts[0].verdict == "violation" and res.counts["rechecked"] == 0
    assert res.counts["name_checked"] == 1 and fake.calls[1]["instructions"].startswith("너는 시험성적서 서술의 구절이")


@pytest.mark.parametrize("text, quote, polarity, want", [
    # 독립 검토 P1 — 흔한 단정 어미 '할 수 있다·쓸 수 있는'은 가능성이 아니다(결함 '양호'가 빠져나가지 않게)
    ("F1 Score 는 0.61 로 양호한 수준이라 할 수 있다.", "양호한 수준", "positive", "violation"),
    ("운영에 쓸 수 있는 양호한 수준이다.", "양호한 수준", "positive", "violation"),
])
def test_v121_assertive_su_itda_is_not_possibility(text, quote, polarity, want):
    m = M(metric_id="M2", display_name="F1 Score", value=0.61, threshold=0.8, status="fail", is_core=True, near_target=False)
    c = Claim("C1", "degree", quote, PIECE.id, PIECE.slot, PIECE.place, PIECE.source, "", "", "", polarity, 0)
    assert _judge_in(c, ans(metric_id="M2"), text, metrics=[m]).verdict == want


def test_v121_class_first_clause_does_not_exempt_a_whole_performance_judgement():
    # 독립 검토 P1 — '가장 약한 클래스는 … 클래스 7 이며, 전체 성능은 … 미흡하다'의 '미흡'은 전체 판단 — 판정 지표로 대조
    m = M(metric_id="M2", display_name="F1 Score", value=0.9, threshold=0.8, status="pass", is_core=True, near_target=False)
    c = Claim("C1", "degree", "미흡하다", PIECE.id, PIECE.slot, PIECE.place, PIECE.source, "", "", "", "negative", 0)
    text = "가장 약한 클래스는 F1 0.05 인 클래스 7 이며, 전체 성능은 운영에 쓰기에 미흡하다."
    assert _judge_in(c, ans(metric_id=""), text, metrics=[m]).verdict == "violation"


def test_v121_method_marker_does_not_hide_a_model_claim():
    # 독립 검토 — '…합쳐 계산하므로 이 모델은 운영에서도 안정적으로 동작한다'는 계산 방식 풀이가 아니라 단정형 추론
    text = "가중 평균은 모든 매장의 주문을 합쳐 계산하므로 이 모델은 운영에서도 안정적으로 동작한다."
    v = _judge_in(claim("C1", "inference", "이 모델은 운영에서도 안정적으로 동작한다"), ans(inference_basis="data"), text)
    assert v.verdict == "inference_violation"


async def test_v121_recheck_has_room_for_more_tool_turns():
    # R-0060 C119: 재확인이 도구를 여러 번 부르다 4턴에서 끊겼다 — 5턴 도구 뒤 답해도 받는다
    tool = lambda i: F.response([F.reasoning_item(), F.function_call_item("lookup_value", {"field": "n_samples"}, f"call_{i}")])  # noqa: E731
    final = second("C1", expr="{metric.M2.value}", expr_b="{metric.M2.threshold}", relation="lt")
    fake, res = await run_recheck([GOAL], first_pass("C1"), *[tool(i) for i in range(5)], final)
    assert len(fake.calls) == 7 and res.verdicts[0].verdict == "match" and res.counts["recheck_retried"] == 0



@pytest.mark.parametrize("text, quote, polarity, metric_id, want", [
    # 클래스 문장에서 첫 답이 핵심 지표를 집어도, 절이 그 지표 이름을 말하지 않으면 대조하지 않는다(R-0060 C138 꼴)
    ("가장 약한 클래스는 F1 0.05 인 클래스 7 이며, 평균보다 0.1 이상 낮아 운영 위험으로 다룬다.", "운영 위험", "negative", "M2", "target_unclear"),
    # 절이 지표 이름을 말하면 그대로 대조 — 결함 꼴('클래스 7 이며, F1 Score 는 … 양호하다')이 빠져나가지 않게
    ("가장 약한 클래스는 클래스 7 이며, F1 Score 는 0.61 로 양호하다.", "양호하다", "positive", "M2", "violation"),
    ("가장 약한 클래스는 클래스 7 이며, 핵심 지표도 양호하다.", "양호하다", "positive", "M2", "violation"),  # '지표'는 전체 낱말
])
def test_v121_class_sentence_with_a_chosen_metric(text, quote, polarity, metric_id, want):
    m = M(metric_id="M2", display_name="F1 Score", value=0.61, threshold=0.8, status="fail", is_core=True, near_target=False)
    if polarity == "negative":
        m = M(metric_id="M2", display_name="F1 Score", value=0.9, threshold=0.8, status="pass", is_core=True, near_target=False)
    c = Claim("C1", "degree", quote, PIECE.id, PIECE.slot, PIECE.place, PIECE.source, "", "", "", polarity, 0)
    assert _judge_in(c, ans(metric_id=metric_id), text, metrics=[m]).verdict == want



# ── 1.22 신뢰구간 수준 숫자 ──────────────────────────────────────────


@pytest.mark.parametrize("quote, number_text, want", [
    ("95%", "95%", True),  # '95% 신뢰구간' — 방법 설정
    ("0.917", "0.917", False),  # 구간 끝값은 데이터
    ("90%", "90%", False),  # 뒤에 '신뢰구간'이 없으면 아님
])
def test_v122_ci_level_number_is_dropped(quote, number_text, want):
    piece = Piece("slot:8.2", "8.2", "5.1", "ai", "지표의 95% 신뢰구간은 0.917~0.997 이고 비율은 90% 이다.")
    c = Claim("C1", "number", quote, piece.id, "8.2", "5.1", "ai", "", "", number_text, "", 0)
    rejected = []
    kept = ex.drop_covered_thresholds([c], {piece.id: piece}, rejected)
    assert (kept == []) is want and ([r["reason"] for r in rejected] == ["ci_level"]) is want


# ── 1.23 문장 꼴로 정하는 '주장 아님'(정밀도 — V2 18회차 사람 정답 0개 잃음) ─────────


LABS = {"7", "8", "Class9", "benign"}


@pytest.mark.parametrize("kind, sentence, want", [
    ("name", "3.2절의 권고는 표본 보강(P2)과 재평가(P3)를 다룬다.", "section_label"),
    ("number", "3.2절의 권고는 표본 0.5 기준(P1)을 다룬다.", "section_label"),
    ("name", "정확 일치율은 주문의 모든 품목을 맞힌 비율이고 오차율은 틀린 칸의 비율이다.", "metric_definition"),
    ("name", "가중 평균은 모든 매장의 주문을 합쳐 계산하므로 주문이 많은 매장의 영향이 크다.", "method_sentence"),
    # 남겨야 하는 꼴(사람 정답과 같은 꼴)
    ("name", "가중 평균은 매장마다 같은 무게로 계산되므로, Class9 처럼 낮은 매장의 부진은 이 평균에 반영된다.", None),  # 숫자 든 이름
    ("name", "가중 평균은 매장마다 같은 무게로 계산되므로, benign 처럼 낮은 클래스의 부진은 이 평균에 반영된다.", None),  # 숫자 없는 이름
    ("name", "가중 평균은 매장마다 같은 무게로 계산되므로, benignity 가 큰 매장의 몫이 크다.", "method_sentence"),  # 이름 경계
    ("comparison", "이 값은 목표보다 낮은 값이다.", None),  # 비교 낱말
    ("name", "정확 일치율은 0.41 로 모든 품목을 맞힌 비율이다.", None),  # 숫자
    ("number", "정확 일치율은 모든 품목을 맞힌 비율이다.", None),  # 숫자 주장은 건드리지 않음(절 꼬리표만 예외)
    ("degree", "클래스 7 은 같은 무게로 평균하므로 부진이 그대로 보인다.", None),  # 숫자 이름 클래스
])
def test_v123_sentence_shapes_that_are_not_claims(kind, sentence, want):
    assert ex.not_a_claim_sentence(kind, sentence, LABS) == want


def test_v123_catalog_labels():
    cat = {n: None for n in ("per_class.Class9.f1", "per_class.7.recall", "distribution.class_distribution.8", "metric.M2.value")}
    assert ex.catalog_labels(cat) == {"Class9", "7", "8"}


async def test_v123_filtered_claims_are_recorded_and_not_refilled():
    piece = Piece("slot:7.1", "7.1", "5.2", "ai", "3.2절의 권고는 표본 보강(P2)을 다룬다. 재평가가 필요하다.")
    item = {"kind": "name", "quote": "표본 보강(P2)", "piece_id": piece.id, "field": "", "field_b": "", "number_text": "", "polarity": ""}
    sents = [{"sid": "s1", "n_claims": 1, "none_reason": ""}, {"sid": "s2", "n_claims": 0, "none_reason": "규범"}]
    fake = F.FakeOpenAI([F.json_response({"claims": [item], "sentences": sents}), F.json_response({"claims": []})])
    res = await ex.extract_claims(fake, [piece], {})
    assert res.claims == [] and [r["reason"] for r in res.rejected] == ["section_label"]
    assert res.sentence_checks["gaps"] == {} and res.sentence_checks["lost"] == 0 and len(fake.calls) == 2  # 본 것으로 친다



# ── 1.24 이름 주장 보기 고르기·넓은 말·레이블 전반 근거 ─────────────────────────────


def test_v124_broader_code_name_matches_narrower_actual():
    assert fa.name_matches("threshold_per_label", "확률에 결정 임계값을 적용해 만든 것") is True  # 넓은 말 ⊃ 좁은 실제 값
    assert fa.name_matches("threshold", "레이블마다 임계값을 적용했다") is False  # 반대 방향은 틀림
    assert fa.name_matches("argmax", "확률에 결정 임계값을 적용해 만든 것") is False


async def test_v124_name_choice_fixes_a_paraphrase_the_table_missed():
    # 첫 답이 이름을 '배치 방식'으로 옮겨 위반 → 보기 고르기가 구절 뜻대로 2) 실시간을 고르면 실제 realtime 과 같아 맞음
    piece = [Piece("slot:7.1", "7.1", "5.2", "ai", "가상 분석. 의뢰자는 응답을 그때그때 바로 돌려주는 방식으로 쓴다고 밝혔다.")]
    c = Claim("C1", "name", "응답을 그때그때 바로 돌려주는 방식", "slot:7.1", "7.1", "5.2", "ai", "", "", "", "", 0)
    pick = F.json_response({"items": [{"claim_id": "C1", "choice": "B"}]})
    fake, res = await _run_name(second("C1", **MISREAD), pick, pieces=piece, c=c)
    assert res.verdicts[0].verdict == "match" and res.verdicts[0].detail["name_choice"] == {"kind": "code", "choice": "B"}
    listing = fake.calls[1]["input"][0]["content"]
    assert "A) 배치 / 일괄 처리 / 일괄로 처리" in listing and "B) 실시간 / 대화형" in listing and "X) 특정하지 않음" in listing
    assert "Y) 보기 값을 부정함" in listing
    assert "realtime" not in listing and "의뢰자 글" not in listing  # 실제 값은 보여 주지 않는다


@pytest.mark.parametrize("kind, opts, choice, want", [
    ("code", ["batch", "realtime"], "B", "match"),  # 실제 realtime
    ("code", ["batch", "realtime"], "A", "violation"),  # 다른 값 — 진짜 틀림
    ("code", ["batch", "realtime"], "X", "unverifiable"),  # 특정 안 함·여럿
    ("code", ["batch", "realtime"], "Y", "violation"),  # 보기 값을 부정·보기에 없음
    ("code", ["batch", "realtime"], "C", "violation"),  # 보기 밖 글자 — 위반 그대로
    ("code", ["batch", "realtime"], "2", "violation"),  # 숫자로 답하면 받지 않는다
    ("code", ["batch", "realtime"], None, "violation"),
    ("client", [], "맞음", "match"), ("client", [], "관계없음", "unverifiable"),
    ("client", [], "어긋남", "violation"), ("client", [], "모름", "violation"),
])
def test_v124_apply_name_choice(kind, opts, choice, want):
    v = fa.ClaimVerdict("C1", "name", "violation", "7.1", "slot:7.1", "q", {"data": "realtime"})
    fa.apply_name_choice(v, kind, opts, choice)
    assert v.verdict == want


def test_v124_broader_choice_counts_as_match():
    v = fa.ClaimVerdict("C1", "name", "violation", "7.1", "slot:7.1", "q", {"data": "threshold_per_label"})
    fa.apply_name_choice(v, "code", ["threshold", "argmax", "threshold_per_label"], "A")  # '임계값 적용'(넓은 말)
    assert v.verdict == "match"
    v = fa.ClaimVerdict("C1", "name", "violation", "7.1", "slot:7.1", "q", {"data": "threshold"})
    fa.apply_name_choice(v, "code", ["threshold", "argmax", "threshold_per_label"], "C")  # 실제는 하나인데 '레이블마다'
    assert v.verdict == "violation"


def test_v124_name_options_by_field_kind():
    src = Sources({**FS, "per_class": {"7": {"f1": 0.3}, "8": {"f1": 0.9}}}, [], {}, CLIENT)
    mk = lambda field, data: fa.ClaimVerdict("C1", "name", "violation", "", "", "q", {"name_field": field, "data": data})  # noqa: E731
    assert fa.name_options(mk("facts.declarations.usage_mode", "realtime"), src, []) == ("code", ["batch", "realtime"])
    assert fa.name_options(mk("facts.weakest_class.label", "7"), src, [])[0] == "label"
    m = M(metric_id="M2", display_name="F1 Score")
    assert fa.name_options(mk("facts.x.display_name", "F1 Score"), src, [m]) == ("metric", ["F1 Score"])
    assert fa.name_options(mk("client.test_data.학습 데이터와의 관계", "다른 출처"), src, []) == ("client", [])
    assert fa.name_options(mk("facts.something_free_text", "아무 글"), src, []) is None  # 보지 않음 — 위반 그대로


async def test_v124_client_contradiction_still_blocks_and_consistent_passes():
    src_client = {"training_data": {"수집 주기": "매주 새 목록을 받는다 (의뢰자 제공: 주 1회 새 목록을 한꺼번에 돌린다.)"}}
    src = Sources(FS, [], {}, src_client)
    piece = [Piece("slot:7.1", "7.1", "5.2", "ai", "가상 분석. 새 목록은 하루 한 번 들어온다.")]
    c = Claim("C1", "name", "새 목록은 하루 한 번 들어온다", "slot:7.1", "7.1", "5.2", "ai", "", "", "", "", 0)
    first = second("C1", name_field="client.training_data.수집 주기", named="하루 한 번 들어온다")
    for answer, want in (("어긋남", "violation"), ("맞음", "match")):
        fake = F.FakeOpenAI([first, F.json_response({"items": [{"claim_id": "C1", "choice": answer}]})])
        res = await fa.judge_faithfulness(fake, piece, [c], src, [], [])
        assert res.verdicts[0].verdict == want
        assert "의뢰자 글: 매주 새 목록을 받는다" in fake.calls[1]["input"][0]["content"]


async def test_v124_name_choice_failure_keeps_violation():
    refuse = lambda: F.response([F.message_item("x", refusal=True)])  # noqa: E731
    fake, res = await _run_name(second("C1", **MISREAD), refuse(), refuse())
    assert len(fake.calls) == 3 and res.verdicts[0].verdict == "violation"
    assert res.verdicts[0].detail["name_choice"]["failure"] is True


def test_v124_generic_label_words_anchor_class_fields():
    # 1.24 검토 P1 로 좁힘 — 클래스 칸 차이('-')·같은 지표·레이블 전반 낱말·차이 낱말이 다 있을 때만
    diff = {"expr": "{per_class.7.f1} - {per_class.8.f1}", "expr_b": "0", "relation": "ne"}
    assert fa.mapping_anchored(diff, "일부 범주를 놓치는 정도가 레이블마다 다르므로", SOURCES)
    assert not fa.mapping_anchored(diff, "일부 범주를 놓치는 정도가 다르므로", SOURCES)  # 레이블 전반 낱말 없음
    assert not fa.mapping_anchored(diff, "클래스별 성능이 고르게 높다", SOURCES)  # 빼기 식이어도 차이 낱말이 없으면 아님
    gt = {"expr": "{per_class.8.f1}", "expr_b": "{per_class.7.f1}", "relation": "gt"}
    assert not fa.mapping_anchored(gt, "클래스별 성능이 고르게 높아 배포해도 안정적이다", SOURCES)  # 차이 낱말·빼기 없음
    assert not fa.mapping_anchored(gt, "클래스별 성능 차이가 작아 안정적이다", SOURCES)  # 빼기 없는 비교
    mixed = {"expr": "{per_class.7.f1} - {per_class.8.recall}", "expr_b": "0", "relation": "ne"}
    assert not fa.mapping_anchored(mixed, "클래스마다 성능이 다르다", SOURCES)  # 다른 지표끼리


async def test_v124_number_violation_is_never_name_checked():
    # 숫자 값이 클래스 이름과 같아도('3' 클래스와 값 3) 숫자 위반은 보기 고르기에 들지 않는다 — 고르는 것만으로 풀리지 않게
    src = Sources({**FS, "per_class": {"3": {"support": 7}, "7": {"support": 9}}}, [], {}, CLIENT)
    piece = [Piece("slot:7.1", "7.1", "5.2", "ai", "가상 분석. 클래스 3 의 표본은 3건이다.")]
    c = Claim("C1", "number", "3건", "slot:7.1", "7.1", "5.2", "ai", "", "", "3", "", 0)
    fake = F.FakeOpenAI([second("C1", expr="{per_class.3.support}")])  # 실제 7 — 클래스 이름 '7' 과 같은 값
    res = await fa.judge_faithfulness(fake, piece, [c], src, [], [])
    assert res.verdicts[0].verdict == "violation" and len(fake.calls) == 1 and res.counts["name_checked"] == 0



def test_v124_numeric_class_names_use_letter_options():
    # 검토 P1: 클래스 '0'~'9' 와 보기 번호가 겹쳤다 — 이제 글자 보기, 숫자 답은 받지 않는다
    src = Sources({**FS, "per_class": {str(i): {"f1": 0.5} for i in range(10)}}, [], {}, CLIENT)
    mk = fa.ClaimVerdict("C1", "name", "violation", "", "", "q", {"name_field": "facts.weakest_class.label", "data": "1"})
    kind, opts = fa.name_options(mk, src, [])
    assert kind == "label" and opts[:3] == ["0", "1", "2"]
    for choice, want in (("C", "violation"), ("2", "violation"), ("B", "match")):  # C = '2' (틀림), '2' 숫자 답, B = '1'
        v = fa.ClaimVerdict("C1", "name", "violation", "", "", "q", {"data": "1"})
        fa.apply_name_choice(v, kind, opts, choice)
        assert v.verdict == want, choice


@pytest.mark.parametrize("marked, choice, want", [
    (False, "X", "inference_violation"), (True, "X", "inference_ok"),  # 검토 P1: 옮긴 추론은 어미 판정으로
    (False, "B", "match"),
])
def test_v124_inference_mapped_as_name_falls_back_to_marker(marked, choice, want):
    v = fa.ClaimVerdict("C1", "inference", "violation", "", "", "q", {"data": "realtime", "marked": marked, "as": "name"})
    fa.apply_name_choice(v, "code", ["batch", "realtime"], choice)
    assert v.verdict == want
    v = fa.ClaimVerdict("C1", "inference", "violation", "", "", "q", {"data": "x", "marked": False, "as": "name"})
    fa.apply_name_choice(v, "client", [], "관계없음")
    assert v.verdict == "inference_violation"


def test_v124_negated_actual_is_not_sent_to_choice():
    # 검토 P1: 실제 합격인데 '합격하지 않았다' — 보기 고르기에서 빼고 그대로 위반
    assert fa.negates_actual("PASS", "최종 합격하지 않았다") and not fa.negates_actual("PASS", "최종 합격이다")
    v = fa.ClaimVerdict("C1", "name", "violation", "", "", "q",
                        {"name_field": "facts.verdict_detail.verdict", "data": "PASS", "named": "최종 합격하지 않았다"})
    assert fa.name_options(v, SOURCES, []) is None



# ── 1.25 묻는 말로 끝나는 구절(V2 1.24 R-0065) ───────────────────────────


@pytest.mark.parametrize("kind, quote, want", [
    ("comparison", "목표 0.80 아래로 떨어지는지", "question"),
    ("name", "같은 분포인지", "question"),
    ("degree", "표본이 충분한지", "question"),
    ("comparison", "목표를 넘을지", "question"),
    ("name", "다른 지점에서도 유지되는지는 이번 시험으로 확인되지 않았다", None),  # 끝이 단정 — 그대로
    ("comparison", "목표 0.80 보다 낮다", None),
    ("number", "0.80 인지", None),  # 숫자 주장은 건드리지 않음
])
def test_v125_question_ending_is_not_a_claim(kind, quote, want):
    assert ex.is_question(kind, quote) == want


async def test_v125_question_claim_is_dropped_and_counted_as_looked():
    piece = Piece("slot:9:6.2", "9", "6.2", "ai", "매주 Recall 을 산출하고 목표 0.80 아래로 떨어지는지 점검하기를 권한다.")
    item = {"kind": "comparison", "quote": "목표 0.80 아래로 떨어지는지", "piece_id": piece.id, "field": "", "field_b": "",
            "number_text": "", "polarity": ""}
    fake = F.FakeOpenAI([F.json_response({"claims": [item], "sentences": [{"sid": "s1", "n_claims": 1, "none_reason": ""}]}),
                         F.json_response({"claims": []})])
    res = await ex.extract_claims(fake, [piece], {})
    assert res.claims == [] and [r["reason"] for r in res.rejected] == ["question"] and res.sentence_checks["lost"] == 0
