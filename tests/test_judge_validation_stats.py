"""tests/test_judge_validation_stats.py — 검증 통계 순수 함수(app/judge/validation/stats.py, 계획 17-2).

공개 예제 정답: Shrout & Fleiss(1979) 표 2(대상 6 × 평가자 4) — ICC(2,1) = ICC(A,1) = 0.29, 95% 신뢰구간 [0.019, 0.761]
(McGraw & Wong 1996 Case 2A 식), 평균 제곱 BMS 11.24 · JMS 32.49 · EMS 1.02.
심판 결과는 실제 모듈의 결과 꼴(FaithResult 등)과 aggregate.scorecard 로 만든다 — core.py 의 키가 바뀌면 여기서 빨개진다.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.judge import aggregate
from app.judge.consistency import ConsistencyResult, Contradiction
from app.judge.core import JudgeRun
from app.judge.coverage import CoverageResult, ItemResult
from app.judge.faithfulness import ClaimVerdict, FaithResult, _count
from app.judge.readability import Finding, ReadabilityResult
from app.judge.validation import stats as S

SF_TABLE2 = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]


# ---------------------------------------------------------------- 합성 심판 결과

def run(*, matched=10, violations=0, contradictions=(), described=(True, True, True), findings=(), flags=(),
        gate=None, viol_quote="정확도는 0.9이다.") -> JudgeRun:
    verdicts = [ClaimVerdict(f"C{i}", "number", "match", "7.1", "p1", f"값 {i}") for i in range(matched)]
    verdicts += [ClaimVerdict(f"V{i}", "number", "violation", "7.1", "p1", viol_quote) for i in range(violations)]
    faith = FaithResult(verdicts, _count(verdicts))
    cons_list = [Contradiction("logic_1", ["p1", "p2"], ["8.1", "8.2"], ["ai", "ai"], list(q)) for q in contradictions]
    cons = ConsistencyResult(cons_list, {"checked": 5, "contradictions": len(cons_list), "by_type": {}})
    items = [ItemResult(f"I{i}", d, "p1", "7.1", "분석-1", "") for i, d in enumerate(described, 1)]
    cov = CoverageResult(items, [], {"applicable": len(items), "described": sum(described), "not_applicable": 0})
    fnd = [Finding("romanization", "p1", "7.1", t) for t in findings]
    flg = [Finding("llm_hard_to_read", "p1", "7.1", t) for t in flags]
    rea = ReadabilityResult(fnd, flg, {"rules": 2, "violations": len(fnd), "llm_flags": len(flg)})
    card = aggregate.scorecard(faith, cons, cov, rea)
    record = {"faithfulness": faith.as_record(), "consistency": cons.as_record(), "coverage": cov.as_record(),
              "readability": rea.as_record()}
    return JudgeRun(gate or aggregate.gate(faith, cons), None, scorecard=card, record=record)


def failed() -> JudgeRun:
    return JudgeRun(None, "timeout", record={"extract": {}})


def scores(f=1.0, cov=1.0, cons=0.0, r=0.0) -> dict:
    return {"faithfulness": f, "coverage": cov, "consistency": cons, "readability": r}


# ---------------------------------------------------------------- ICC(A,1)

def test_icc_matches_shrout_fleiss_table2():
    res = S.icc_a1(SF_TABLE2)
    assert round(res["icc"], 3) == 0.290
    assert (round(res["ci_low"], 3), round(res["ci_high"], 3)) == (0.019, 0.761)
    assert (res["n_targets"], res["n_raters"], res["zero_variance"], res["note"]) == (6, 4, False, "")
    assert set(res) == {"icc", "ci_low", "ci_high", "n_targets", "n_raters", "zero_variance", "note"}


def test_mean_squares_match_shrout_fleiss_table2():
    msr, msc, mse = S._mean_squares(np.asarray(SF_TABLE2, dtype=float))
    assert (round(msr, 2), round(msc, 2), round(mse, 2)) == (11.24, 32.49, 1.02)


def test_icc_hand_example_and_rater_bias_lowers_absolute_agreement():
    # 손 계산: [[1,2],[3,4]] → MSR 4, MSC 1, MSE 0 → 4 / (4 + 0 + 2·1/2) = 0.8
    assert S.icc_a1([[1, 2], [3, 4]])["icc"] == pytest.approx(0.8)
    # 회차끼리 완전히 같으면 1(잔차 0, 평가자 차 0), 신뢰구간도 1
    same = S.icc_a1([[1, 1], [2, 2], [3, 3]])
    assert (same["icc"], same["ci_low"], same["ci_high"], same["zero_variance"]) == (1.0, 1.0, 1.0, False)


def test_icc_ci_level_and_bounds():
    wide = S.icc_a1(SF_TABLE2, level=0.99)
    narrow = S.icc_a1(SF_TABLE2)
    assert wide["ci_low"] < narrow["ci_low"] < narrow["icc"] < narrow["ci_high"] < wide["ci_high"]


def test_icc_zero_variance_rule():
    res = S.icc_a1([[3, 3, 3], [3, 3, 3]])
    assert res["zero_variance"] is True and res["icc"] is None and "분산 0" in res["note"]
    # 대상 1개·회차 1회면 분산 0 규칙을 쓰지 않는다 — 계산 불가(J10 검토)
    for m in ([[0.0]], [[0.0, 0.0, 0.0]], [[0.0], [0.0]]):
        r = S.icc_a1(m)
        assert r["zero_variance"] is False and r["icc"] is None and "계산 불가" in r["note"], m


def test_reproducibility_one_report_is_not_computable_not_passed():
    one = {"A": [scores(f=1.0), scores(f=1.0), scores(f=1.0)]}
    res = S.reproducibility(one)
    assert all(a["passed"] is None for a in res["axes"].values()) and res["axes_passed"] is None
    assert S.tri([True, None]) is None and S.tri([None, False]) is False and S.tri([True, True]) is True


def test_icc_edges():
    assert S.icc_a1([])["note"] == "빈 입력" and S.icc_a1([])["icc"] is None
    one = S.icc_a1([[1, 2, 3]])  # 대상 1개, 흔들림 있음 → 계산 불가
    assert one["icc"] is None and one["zero_variance"] is False and one["note"]
    single_rater = S.icc_a1([[1], [2]])
    assert single_rater["icc"] is None and single_rater["n_raters"] == 1
    with pytest.raises(ValueError):
        S.icc_a1([[1, None], [2, 3]])
    with pytest.raises(ValueError):
        S.icc_a1([1, 2, 3])


# ---------------------------------------------------------------- 축 점수

def test_axis_scores_from_real_scorecard_keys():
    r = run(matched=3, violations=1, contradictions=[("a", "b")], described=(True, False, True, True), findings=("worse",),
            flags=("x",))
    got = S.axis_scores(r)
    assert got == {"faithfulness": 0.75, "coverage": 0.75, "consistency": 1.0, "readability": 1.0}  # llm_flags 는 넣지 않음
    as_dict = S.axis_scores({"gate": r.gate, "scorecard": r.scorecard, "record": r.record})  # 장부·파일에 적힌 꼴도 같다
    assert as_dict == got


def test_axis_scores_none_when_missing_or_zero_denominator():
    assert S.axis_scores(failed()) == {a: None for a in S.AXES}
    assert S.axis_scores(None) == {a: None for a in S.AXES}
    assert S.axis_scores(run(matched=0, described=()))["faithfulness"] is None  # checked 0
    assert S.axis_scores(run(matched=0, described=()))["coverage"] is None  # applicable 0
    card = run().scorecard
    del card["readability"]["violations"]
    assert S.axis_scores({"scorecard": card})["readability"] is None


# ---------------------------------------------------------------- 게이트 재현성

def test_gate_stability():
    assert S.gate_stability(["pass"] * 3) is True
    assert S.gate_stability(["blocked"] * 5, expected_runs=5) is True
    assert S.gate_stability(["pass", "pass", "blocked"]) is False
    assert S.gate_stability(["pass", None, "pass"]) is False
    assert S.gate_stability([None, None]) is False
    assert S.gate_stability(["pass"] * 2, expected_runs=3) is False
    assert S.gate_stability([]) is False


# ---------------------------------------------------------------- 재현성

def test_reproducibility_axes_gates_spread():
    m = [[0.9, 0.8, 0.85], [0.5, 0.6, 0.55], [0.7, 0.7, 0.75]]
    runs = {f"R{i}": [run(matched=0) for _ in row] for i, row in enumerate(m)}
    for rep, row in zip(runs, m):
        for r, v in zip(runs[rep], row):
            r.scorecard["faithfulness"].update(checked=100, matched=round(v * 100))
    res = S.reproducibility(runs, expected_runs=3)
    assert res["axes"]["faithfulness"]["icc"] == pytest.approx(S.icc_a1(m)["icc"])
    assert res["axes"]["faithfulness"]["passed"] is (S.icc_a1(m)["icc"] >= 0.75)
    for axis in ("consistency", "readability"):  # 모두 0 → 분산 0 규칙으로 통과
        assert res["axes"][axis]["zero_variance"] and res["axes"][axis]["passed"]
    assert res["axes"]["coverage"]["zero_variance"]
    assert res["n_runs"] == 3 and res["gates_passed"] is True
    assert res["spread"]["R0"]["faithfulness"] == pytest.approx(0.1)


def test_reproducibility_threshold_is_an_argument():
    m = [[0.9, 0.8, 0.85], [0.5, 0.6, 0.55], [0.7, 0.7, 0.75]]
    icc = S.icc_a1(m)["icc"]
    runs = {f"R{i}": [scores(f=v) for v in row] for i, row in enumerate(m)}
    assert S.reproducibility(runs, threshold=icc - 0.01)["axes"]["faithfulness"]["passed"] is True
    assert S.reproducibility(runs, threshold=icc + 0.01)["axes"]["faithfulness"]["passed"] is False


def test_reproducibility_failed_run_drops_report_and_breaks_gate():
    runs = {"A": [run(), run(), failed()], "B": [run(), run(), run()]}
    res = S.reproducibility(runs)
    assert res["axes"]["faithfulness"]["dropped"] == ["A"] and res["axes"]["faithfulness"]["passed"] is False
    assert "A" in res["axes"]["faithfulness"]["note"]
    assert res["gates"]["A"]["stable"] is False and res["gates"]["B"]["stable"] is True and res["gates_passed"] is False


def test_reproducibility_edges():
    with pytest.raises(ValueError):
        S.reproducibility({})
    with pytest.raises(ValueError):
        S.reproducibility({"A": [run()], "B": [run(), run()]})
    with pytest.raises(ValueError):
        S.reproducibility({"A": []})


def test_spread():
    assert S.spread([1, 3, 2]) == 2.0
    assert S.spread([0.5]) == 0.0
    assert S.spread([]) is None and S.spread([1, None]) is None


# ---------------------------------------------------------------- 결함 탐지

PLANTED = "Recall 은 Precision 보다 0.015 높다."


def test_defect_detected_faithfulness_and_consistency():
    hit = run(violations=1, viol_quote="0.015 높다")
    miss = run(violations=1, viol_quote="다른 문장의 0.3")
    assert S.defect_detected(hit, "faithfulness", sentence=PLANTED) is True
    assert S.defect_detected(miss, "F", sentence=PLANTED) is False
    matched_only = run(matched=1)
    matched_only.record["faithfulness"]["verdicts"][0]["quote"] = "0.015 높다"
    assert S.defect_detected(matched_only, "faithfulness", sentence=PLANTED) is False  # 그 문장을 짚었어도 부합은 탐지 아님
    assert S.defect_detected(run(contradictions=[("앞 문장", PLANTED)]), "consistency", sentence=PLANTED) is True
    assert S.defect_detected(run(contradictions=[("앞 문장", "뒤 문장")]), "Cons", sentence=PLANTED) is False
    assert S.defect_detected(failed(), "faithfulness", sentence=PLANTED) is False


def test_defect_detected_readability_and_coverage():
    s = "오류 방향은 worse 쪽이다."
    assert S.defect_detected(run(findings=("worse",)), "readability", sentence=s) is True
    assert S.defect_detected(run(flags=(s,)), "R", sentence=s) is True
    assert S.defect_detected(run(findings=("better",)), "readability", sentence=s) is False
    assert S.defect_detected(run(described=(True, False)), "coverage", item_id="I2") is True
    assert S.defect_detected(run(described=(True, True)), "Cov", item_id="I2") is False
    assert S.defect_detected(run(described=(True,)), "coverage", item_id="I2") is False  # 항목이 없음(해당 없음) ≠ 탐지
    with pytest.raises(ValueError):
        S.defect_detected(run(), "coverage")
    with pytest.raises(ValueError):
        S.defect_detected(run(), "faithfulness", sentence="  ")
    with pytest.raises(ValueError):
        S.defect_detected(run(), "speed", sentence="x")


def test_touches_truncated_quote():
    long_line = "가" * 115 + " 첫 문장 끝. " + PLANTED
    cut = long_line[:120]
    assert S.touches(long_line, PLANTED) is True
    assert S.touches(cut + PLANTED[:3], PLANTED) is False  # 겹침이 OVERLAP_MIN 보다 짧다
    assert S.touches("앞말 " + PLANTED[:12], PLANTED) is True  # 잘린 끝이 문장 앞과 12자 겹침
    assert S.touches(PLANTED[-12:] + " 뒷말", PLANTED) is True
    assert S.touches("", PLANTED) is False


def test_detection_passed():
    full = {"F1": {"axis": "F", "hits": [True] * 3}, "C1": {"axis": "Cons", "hits": [True] * 3}}
    others = {f"V{i}": {"axis": "Cov", "hits": [True, True, False]} for i in range(6)}
    res = S.detection_passed(full | others)
    assert res["passed"] and res["others_needed"] == 5
    one_miss = full | {"F1": {"axis": "F", "hits": [True, True, False]}} | others
    assert S.detection_passed(one_miss)["gate_axes_ok"] is False
    weak = dict(others, V0={"axis": "R", "hits": [True, False, False]}, V1={"axis": "R", "hits": [False] * 3})
    assert S.detection_passed(full | weak)["others_ok"] is False  # 6개 중 4개만
    weak1 = dict(others, V0={"axis": "R", "hits": [True, False, False]})
    assert S.detection_passed(full | weak1)["others_ok"] is True  # 6개 중 5개
    # 줄인 V2(결함마다 1회, 필수 내용·표기 3개): ceil(3 × 5/6) = 3 → 모두
    small = {"V1": {"axis": "Cov", "hits": [True]}, "V2": {"axis": "R", "hits": [True]}, "V3": {"axis": "R", "hits": [False]}}
    assert S.detection_passed(small)["others_needed"] == 3 and not S.detection_passed(small)["passed"]
    assert S.detection_passed(small, other_min_defect_share=2 / 3)["passed"]
    with pytest.raises(ValueError):
        S.detection_passed({})
    with pytest.raises(ValueError):
        S.detection_passed({"F1": {"axis": "F", "hits": []}})


# ---------------------------------------------------------------- 독립성·sham

def test_independence_uses_base_spread_and_skips_defect_axis():
    base = [scores(f=0.90, cov=0.80), scores(f=0.92, cov=0.85), scores(f=0.91, cov=0.80)]
    inside = [scores(f=0.60, cov=0.84)]  # 사실 일치(결함 축)는 크게 떨어져도 상관없다
    res = S.independence_ok(base, inside, "faithfulness")
    assert res["ok"] and "faithfulness" not in res["axes"]
    assert res["axes"]["coverage"]["spread"] == pytest.approx(0.05)
    outside = [scores(f=0.60, cov=0.70)]
    assert S.independence_ok(base, outside, "F")["ok"] is False
    # 코드로만 정해지는 점수는 흔들림 폭 0 → 작은 변화도 실패(Q11 이 미리 알린 성질)
    assert S.independence_ok([scores()] * 3, [scores(r=1.0)], "coverage")["ok"] is False


def test_independence_mode_mean_vs_first():
    base = [scores(cov=0.80), scores(cov=0.90), scores(cov=0.85)]  # 평균 0.85, 1회째 0.80, 폭 0.10
    other = [scores(cov=0.94)]
    assert S.independence_ok(base, other, "F", mode="mean")["ok"] is True  # |0.94 − 0.85| = 0.09
    assert S.independence_ok(base, other, "F", mode="first")["ok"] is False  # |0.94 − 0.80| = 0.14
    with pytest.raises(ValueError):
        S.independence_ok(base, other, "F", mode="median")


def test_sham_checks_all_axes_and_missing_scores():
    base = [scores(), scores(), scores()]
    assert S.sham_ok(base, [scores()])["ok"] is True
    res = S.sham_ok(base, [scores(f=0.9)])
    assert res["ok"] is False and set(res["axes"]) == set(S.AXES)
    assert S.sham_ok(base, [failed()])["ok"] is False  # 실패 회차는 확인 불가 → 통과 아님
    assert S.sham_ok([run()] * 2, [run()])["ok"] is True  # 심판 결과 꼴도 받는다
    with pytest.raises(ValueError):
        S.sham_ok([], [scores()])


def test_summarize():
    assert S.summarize([1, 2, 6]) == 3.0 and S.summarize([1, 2, 6], "first") == 1.0
    with pytest.raises(ValueError):
        S.summarize([])


# ---------------------------------------------------------------- 일치율

def test_agreement_rate_and_human_rule():
    assert S.agreement_rate(["a", "b", "c", "d"], ["a", "b", "x", "d"]) == 0.75
    with pytest.raises(ValueError):
        S.agreement_rate([], [])
    with pytest.raises(ValueError):
        S.agreement_rate(["a"], ["a", "b"])
    assert S.human_agreement_ok(0.80, 0.85) is True  # 정확히 5%p 낮음 → 통과
    assert S.human_agreement_ok(0.79, 0.85) is False
    assert S.human_agreement_ok(0.79, 0.85, margin=0.10) is True
    assert math.isclose(S.HUMAN_MARGIN, 0.05)


# ---------------------------------------------------------------- 일치도 계수(agreement.py, 계획 17-7)

from app.judge.validation import agreement as A  # noqa: E402

N = None
# Krippendorff(2011) "Computing Krippendorff's Alpha-Reliability" — 평가자 4 × 단위 12, 빈칸 있음: 명목 0.743, 구간 0.849
K2011 = [[1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
         [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
         [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
         [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N]]
# 같은 글의 두 평가자 이진 예: 0.095
K2011_BINARY = [[0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]]
# 두 평가자 50항목: 예/예 20, 예/아니오 5, 아니오/예 10, 아니오/아니오 15 → 관측 0.7, κ 우연 0.5 → κ 0.4
# AC1 손 계산: π_예 = (25+30)/100 = 0.55 → 우연 = 2·0.55·0.45 / 1 = 0.495 → (0.7 − 0.495)/0.505 = 0.40594
RATER_A = ["y"] * 25 + ["n"] * 25
RATER_B = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15


def test_krippendorff_published_examples():
    assert round(A.krippendorff_alpha(K2011), 3) == 0.743
    assert round(A.krippendorff_alpha(K2011, "interval"), 3) == 0.849
    assert round(A.krippendorff_alpha(K2011_BINARY), 3) == 0.095


def test_krippendorff_edges():
    assert A.krippendorff_alpha([]) is None
    assert A.krippendorff_alpha([[1, N], [N, 2]]) is None  # 값이 둘인 단위가 없다
    assert A.krippendorff_alpha([[3, 3], [3, 3]]) is None  # 변이 없음
    assert A.krippendorff_alpha([[1, 2, 3], [1, 2, 3]]) == 1.0
    with pytest.raises(ValueError):
        A.krippendorff_alpha(K2011, "ordinal")
    with pytest.raises(ValueError):
        A.krippendorff_alpha([[1, 2], [1]])


def test_cohen_kappa_and_gwet_ac1():
    assert A.cohen_kappa(RATER_A, RATER_B) == pytest.approx(0.4)
    # 주변 비율이 서로 다른 손 계산: A y3·n1, B y1·n3, 관측 0.5, 우연 = .75·.25 + .25·.75 = .375 → .125/.625 = 0.2
    assert A.cohen_kappa(["y", "y", "y", "n"], ["y", "n", "n", "n"]) == pytest.approx(0.2)
    assert A.gwet_ac1(RATER_A, RATER_B) == pytest.approx(0.205 / 0.505)
    # 한쪽으로 쏠린 자료(18/20 일치, 둘 다 y 19·n 1): κ 는 낮게, AC1 은 높게(역설)
    # κ 우연 = (19·19 + 1·1)/400 = 0.905 → (0.9 − 0.905)/0.095. AC1 우연 = 2·0.95·0.05 = 0.095 → (0.9 − 0.095)/0.905
    a = ["y"] * 18 + ["n", "y"]
    b = ["y"] * 18 + ["y", "n"]
    assert A.cohen_kappa(a, b) == pytest.approx(-0.005 / 0.095)
    assert A.gwet_ac1(a, b) == pytest.approx(0.805 / 0.905)
    # 관측되지 않은 범주를 categories 로 주면 q = 3: 우연 = 0.095 / 2 = 0.0475
    assert A.gwet_ac1(a, b, categories=["y", "n", "?"]) == pytest.approx(0.8525 / 0.9525)


def test_kappa_ac1_edges():
    assert A.cohen_kappa([], []) is None and A.gwet_ac1([], []) is None
    assert A.cohen_kappa(["y", "y"], ["y", "y"]) is None  # 우연 일치 1
    assert A.gwet_ac1(["y", "y"], ["y", "y"]) is None  # 범주 1개
    assert A.cohen_kappa(["y", N, "n"], ["y", "n", "n"]) == 1.0  # 빈칸 쌍은 뺀다
    with pytest.raises(ValueError):
        A.cohen_kappa(["y"], ["y", "n"])


def test_zero_denominator_is_noted_apart_from_failed_runs():
    """사실 일치 checked 0(주장 없음)이 모든 회차면 '해당 없음 — 결정 대기'로 적는다. 판정은 그대로 미달(닫힌 쪽, J10 인계)."""
    def card(checked):
        return {"gate": "pass", "scorecard": {"faithfulness": {"checked": checked, "matched": checked},
                                              "coverage": {"applicable": 1, "described": 1},
                                              "consistency": {"contradictions": 0}, "readability": {"violations": 0}}}
    runs = {"A": [card(0), card(0)], "B": [card(2), card(2)], "C": [card(0), {"gate": None}]}
    f = S.reproducibility(runs)["axes"]["faithfulness"]
    assert f["dropped"] == ["A", "C"] and f["not_applicable"] == ["A"] and f["passed"] is False
    assert "분모 0(해당 없음 — 결정 대기): A" in f["note"]
