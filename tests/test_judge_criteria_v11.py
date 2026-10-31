"""tests/test_judge_criteria_v11.py — 심판 기준 1.1 고침(criteria/CHANGELOG.md "## 1.1-dev" 1~5).

V2 1차(재현성 5회 모두 발급 불가)의 심판 쪽 오판을 고친 것마다 맞는 쪽과 틀린 쪽을 함께 본다(고침이 너무 넓어 진짜 위반까지
통과시키지 않게). 문장은 모두 지어낸 것이다 — 봉인 서술·판 3 발급본·예시본 글을 쓰지 않는다(검증 README 오염 방지 규칙 2·3).
"""
from __future__ import annotations

import pytest

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
    r = judge(claim("C2", "inference", "가중 평균으로 보아도 목표에 못 미친다"), ans(expr="{metric.M2.value}", expr_b="0.8", relation="lt"))
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
    fs = {**FS, "per_class": [{"label": "7", "f1": gap_high, "precision": 0.5, "recall": 0.5, "support": 100},
                              {"label": "8", "f1": 0.6, "precision": 0.5, "recall": 0.5, "support": 300}]}
    src = Sources(fs, [], {}, CLIENT)
    assert "크다" in src.lookup("rule.imbalance_gap_threshold")["note"]
    c = claim("C1", "inference", "두 평균이 벌어진 것은 클래스마다 성능이 고르지 않다는 뜻이다")
    v = fa.judge_answer(c, ans(expr="{per_class.7.f1} - {per_class.8.f1}", expr_b="{rule.imbalance_gap_threshold}", relation="ge"),
                        src, [], [], PIECE)
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
