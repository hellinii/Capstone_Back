"""판정자 평가 실험 집계 순수 함수(app/judge/validation/evaluation.py, 사전 등록 v1 §6) 시험.

합성 문서·합성 채점 결과만 쓴다(실호출·봉인 서술 없음). 이빨: 경계값(허용 폭 딱 맞음/조금 넘음), 실패 회차를 미탐지로 세지 않음,
C01 두 축 따로, 판정표에 새는 것 종류마다, 빈 묶음 막기(검사 전에 대상이 비지 않았는지 먼저 본다).
"""
from __future__ import annotations

import math

import pytest

from app.judge.reading import Piece
from app.judge.validation import evaluation as E

TEXT_71 = "문장 가는 0.91 이다. 문장 나는 FN 43건이다. 문장 다는 43건과 같다."
TEXT_82 = "총평 하나다. 총평 둘은 양호하다."


def pieces(t71: str = TEXT_71, t82: str = TEXT_82):
    return [
        Piece("slot:7.1", "7.1", "5.2", "ai", t71),
        Piece("slot:8.2", "8.2", "5.1", "template", t82),
        Piece("m:fixed", "8.2", "5.1", "fixed", "고정 문구 0.91 이다."),
        Piece("rec:6.2:P1:0", "9", "6.2", "ai", "권고: 임계값을 다시 고른다 / 기대 효과: 놓침이 준다"),
    ]


def index(t71: str = TEXT_71, t82: str = TEXT_82) -> E.DocIndex:
    return E.DocIndex.from_pieces(pieces(t71, t82))


def verdict(cid, quote, verdict="violation", kind="number", piece="slot:7.1", **detail):
    return {"claim_id": cid, "kind": kind, "verdict": verdict, "slot": "7.1", "piece_id": piece, "quote": quote,
            "detail": detail}


def contradiction(kind, qa, qb, pa="slot:7.1", pb="slot:8.2", **detail):
    return {"type": kind, "pieces": [pa, pb], "slots": ["7.1", "8.2"], "sources": ["ai", "template"], "quotes": [qa, qb],
            "detail": detail}


def claim(cid, quote, kind="number", piece="slot:7.1", field="", field_b=""):
    return {"id": cid, "kind": kind, "quote": quote, "piece_id": piece, "slot": "7.1", "field": field, "field_b": field_b}


def run(gate="blocked", verdicts=(), contradictions=(), claims=(), card=None, failure=None):
    card = card or {"faithfulness": {"checked": 10, "matched": 9}, "coverage": {"applicable": 10, "described": 8},
                    "consistency": {"contradictions": len(contradictions)}, "readability": {"violations": 0}}
    return {"gate": gate, "failure": failure, "scorecard": card if gate else None,
            "record": {"extract": {"claims": list(claims), "failure": None},
                       "faithfulness": {"verdicts": list(verdicts)},
                       "consistency": {"contradictions": list(contradictions)}}}


# ── 6.1 문장 대조 ──


def test_index_numbers_only_narrative_sentences():
    idx = index()
    assert [s.text for s in idx.sentences][:3] == ["문장 가는 0.91 이다.", "문장 나는 FN 43건이다.", "문장 다는 43건과 같다."]
    assert {s.source for s in idx.sentences} == {"ai", "template"}  # 고정 문구는 빠진다
    assert len(idx.sentences) == 6 and [s.id for s in idx.sentences] == list(range(1, 7))


def test_locate_same_piece_first_then_whole_doc_and_not_outside_pieces():
    idx = index()
    assert idx.locate("0.91", "slot:7.1") == [1]
    assert idx.locate("43건", "slot:7.1") == [2, 3]  # 같은 조각에서 여러 문장에 닿으면 모두(stats.touches 그대로)
    assert idx.locate("총평 둘은", "slot:7.1") == [5]  # 같은 조각에 없으면 문서 전체
    assert idx.locate("총평 둘은") == [5]  # 조각을 모르면 문서 전체
    assert idx.locate("0.91", "m:fixed") == []  # 서술 밖(고정 문구) 조각의 인용은 서술 문장에 붙이지 않는다
    assert idx.locate("", "slot:7.1") == []


def test_injected_sentences_replace_insert_split_and_swap():
    base = "가는 하나다. 나는 둘이다. 다는 셋이다."
    assert E.injected_sentences(base, "가는 하나다. 나는 넷이다. 다는 셋이다.") == [(1, "나는 넷이다.")]
    assert E.injected_sentences(base, "가는 하나다. 나는 둘이다. 더한 문장이다. 다는 셋이다.") == [(2, "더한 문장이다.")]
    assert E.injected_sentences(base, "가는 하나다. 나는 둘. 이다. 다는 셋이다.") == [(1, "나는 둘."), (2, "이다.")]
    assert E.injected_sentences(base, "다는 셋이다. 나는 둘이다. 가는 하나다.") == [(0, "다는 셋이다."), (2, "가는 하나다.")]
    assert E.injected_sentences(base, "나는 둘이다. 가는 하나다. 다는 셋이다.") == [(0, "나는 둘이다."), (1, "가는 하나다.")]
    assert E.injected_sentences(base, base) == []


def test_injected_ids_by_position_and_in_recommendation_rows():
    new = "문장 가는 0.91 이다. 문장 나는 FN 43건이다. 바뀐 문장이다."
    idx = index(t71=new)
    inj = E.injected_sentences(TEXT_71, new)
    assert E.injected_ids(idx, "interpretation.confusion_analysis", inj) == [3]
    assert E.injected_ids(idx, "recommendations.0.action", [(0, "임계값을 다시 고른다")]) == [6]
    with pytest.raises(ValueError):
        E.injected_ids(idx, "interpretation.confusion_analysis", [(2, "없는 문장이다.")])


# ── 6.2 탐지 ──


def test_detection_gate_axis_out_of_place_and_failed():
    idx = index()
    inj = [2]
    hit = run(verdicts=[verdict("C1", "FN 43건", piece="slot:7.1")])
    d = E.detect(hit, idx, inj, ["faithfulness"])
    assert d["status"] == "탐지" and d["gate_detected"] is True and d["axes"] == {"faithfulness": True}
    assert d["out_of_place"] == []
    # 위치 불일치: 다른 문장의 위반만 — 게이트 차단이지만 미탐지, 위치 불일치 표시 1
    other = run(verdicts=[verdict("C1", "0.91 이다", piece="slot:7.1")])
    d = E.detect(other, idx, inj, ["faithfulness"])
    assert d["status"] == "미탐지" and d["gate_detected"] is False and len(d["out_of_place"]) == 1
    # 축 수준: 일관성 모순으로만 잡힘 → 게이트 탐지, 기대 축(충실성) 미탐지
    cons = run(contradictions=[contradiction("analysis_vs_finding", "FN 43건이다", "총평 둘은 양호하다")])
    d = E.detect(cons, idx, inj, ["faithfulness"])
    assert d["gate_detected"] is True and d["axes"] == {"faithfulness": False}
    # 실패 회차는 미탐지가 아니라 실패
    d = E.detect(run(gate=None, failure="timeout"), idx, inj, ["faithfulness"])
    assert d["status"] == "실패" and d["gate_detected"] is None and d["axes"] == {"faithfulness": None}
    assert E.detect(None, idx, inj, ["faithfulness"])["status"] == "미실행"
    # 통과 회차(표시 없음)는 미탐지
    assert E.detect(run(gate="pass"), idx, inj, ["faithfulness"])["gate_detected"] is False
    # 위반이 아닌 판정(match·inference_ok)은 표시가 아니다
    ok = run(gate="pass", verdicts=[verdict("C1", "FN 43건", verdict="match")])
    assert E.detect(ok, idx, inj, ["faithfulness"])["gate_detected"] is False


def test_c01_axes_are_counted_separately():
    idx = index()
    inj = [2]
    both = run(verdicts=[verdict("C1", "FN 43건")],
               contradictions=[contradiction("value_mismatch", "43건", "0.91", pa="slot:7.1", pb="slot:7.1", field="x")])
    d = E.detect(both, idx, inj, ["faithfulness", "consistency"])
    assert d["axes"] == {"faithfulness": True, "consistency": True}
    only_cons = run(contradictions=[contradiction("value_mismatch", "FN 43건", "0.91", pa="slot:7.1", pb="slot:7.1")])
    d = E.detect(only_cons, idx, inj, ["faithfulness", "consistency"])
    assert d["gate_detected"] is True and d["axes"] == {"faithfulness": False, "consistency": True}


def test_detect_refuses_empty_injected():
    with pytest.raises(ValueError):
        E.detect(run(), index(), [], ["faithfulness"])


def test_two_run_class_and_failures_not_missed():
    assert E.two_run_class([True, True]) == "두 회차 모두 탐지"
    assert E.two_run_class([True, False]) == "한 회차만 탐지"
    assert E.two_run_class([False, False]) == "두 회차 모두 미탐지"
    assert E.two_run_class([True, None]) == "한 회차 탐지 · 한 회차 실패"
    assert E.two_run_class([False, None]) == "한 회차 미탐지 · 한 회차 실패"
    assert E.two_run_class([None, None]) == "두 회차 모두 실패"
    assert E.two_run_class([True]).startswith("미실행")


def test_wilson_18_of_18_lower_bound_is_082():
    w = E.wilson(18, 18)
    assert math.isclose(w["low"], 0.8241, abs_tol=5e-5) and round(w["low"], 2) == 0.82 and w["high"] == 1.0
    assert E.wilson(0, 0)["low"] is None


# ── 6.6 불변성·6.7 독립성 ──


def _scored(gate, faith=0.9, cov=0.8, cons=0, rea=0):
    card = {"faithfulness": {"checked": 100, "matched": round(faith * 100)}, "coverage": {"applicable": 100, "described": round(cov * 100)},
            "consistency": {"contradictions": cons}, "readability": {"violations": rea}}
    return {"gate": gate, "scorecard": card, "record": {}}


def test_tolerance_boundary_exact_passes_and_epsilon_over_fails():
    base = [0.90, 0.92]
    assert E.within(0.97, base, "faithfulness")[0] is True  # 최댓값 + 0.05 딱
    assert E.within(0.85, base, "faithfulness")[0] is True  # 최솟값 − 0.05 딱
    assert E.within(0.97 + 1e-6, base, "faithfulness")[0] is False
    assert E.within(0.85 - 1e-6, base, "faithfulness")[0] is False
    assert E.within(3, [1, 2], "consistency")[0] is True and E.within(4, [1, 2], "consistency")[0] is False
    assert E.within(0, [1, 2], "readability")[0] is True and E.within(-0.01, [1, 2], "readability")[0] is False
    assert E.within(None, [None, None], "coverage") == (True, "모두 점수 없음")
    assert E.within(None, [0.5, 0.6], "coverage")[0] is None and E.within(0.5, [None, 0.6], "coverage")[0] is None


def test_invariance_rules_including_original_split():
    base = [_scored("pass", faith=0.90), _scored("pass", faith=0.92)]
    same = [_scored("pass", faith=0.97), _scored("pass", faith=0.85)]
    assert E.invariance(base, same)["status"] == "불변"
    out = E.invariance(base, [_scored("pass", faith=0.98), _scored("pass")])
    assert out["status"] == "불변 아님" and out["axes"]["faithfulness"]["ok"] is False
    out = E.invariance(base, [_scored("blocked"), _scored("pass")])
    assert out["status"] == "불변 아님" and out["decision"] is False
    split = E.invariance([_scored("pass"), _scored("blocked")], same)
    assert split["status"] == "판단 불가(원본 갈림)" and split["invariant"] is None
    failed = E.invariance(base, [_scored("pass"), {"gate": None, "failure": "timeout"}])
    assert failed["status"] == "판단 불가(변형 실패)"
    assert E.invariance(base, [_scored("pass")])["status"] == "판단 불가(회차 부족)"


def test_independence_marks_allowed_secondary_axis():
    base = [_scored("pass", cons=0), _scored("pass", cons=0)]
    defect = [_scored("blocked", cons=3), _scored("blocked", cons=0)]
    rows = E.independence(base, defect, ["faithfulness"], ["consistency"])
    assert set(rows) == {"coverage", "consistency", "readability"}
    assert rows["consistency"]["status"] == "허용" and rows["coverage"]["status"] == "범위 안"
    rows = E.independence(base, defect, ["faithfulness"], [])
    assert rows["consistency"]["status"] == "벗어남"
    rows = E.independence(base, [_scored("blocked"), {"gate": None}], ["faithfulness"], [])
    assert all(r["ok"] is None for r in rows.values())


# ── 6.5 재현성 ──


def test_jaccard_both_empty_is_one():
    assert E.jaccard(set(), set()) == 1.0
    assert E.jaccard({1}, set()) == 0.0
    assert E.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_reproducibility_groups_and_excludes_failed_docs():
    idx = index()
    flag = run(verdicts=[verdict("C1", "FN 43건")])
    flag2 = run(verdicts=[verdict("C1", "FN 43건"), verdict("C2", "0.91 이다")])
    per = {"a": [run(gate="pass"), run(gate="pass")], "b": [flag, flag2], "c": [flag, run(gate="pass")],
           "d": [flag, run(gate=None, failure="timeout")]}
    out = E.reproducibility(per, {k: idx for k in per})
    assert out["excluded"] == {"d": ["blocked", None]}
    assert out["docs"]["a"]["group"] == "두 회차 모두 통과" and out["docs"]["a"]["jaccard"] == 1.0
    assert out["docs"]["b"]["group"] == "두 회차 모두 차단" and out["docs"]["b"]["exact"] is False
    assert out["docs"]["b"]["jaccard"] == pytest.approx(0.5)
    assert out["docs"]["c"]["group"] == "갈림" and out["docs"]["c"]["same_decision"] is False
    assert out["overall"]["n"] == 3 and out["overall"]["decision_agreement"] == pytest.approx(2 / 3)
    assert out["groups"]["갈림"]["n"] == 1


# ── 6.3 미탐지 원인·6.8 추출률 ──


def test_extraction_kind_mapping_follows_prereg():
    assert E.RELATED_KINDS["F01"] == E.RELATED_KINDS["F02"] == E.RELATED_KINDS["C01"] == ("number",)
    assert E.RELATED_KINDS["F03"] == E.RELATED_KINDS["F04"] == ("name",)
    assert E.RELATED_KINDS["F05"] == E.RELATED_KINDS["F06"] == ("comparison", "number")
    assert E.RELATED_KINDS["F07"] == ("degree",) and E.RELATED_KINDS["F08"] == E.RELATED_KINDS["F09"] == ("inference",)
    assert all(E.RELATED_KINDS[f"C{i:02d}"] is None for i in range(2, 10))
    assert len(E.EXTRACTION_IDS) == 10 and set(E.RELATED_KINDS) == set(E.DEFECT_IDS)
    idx = index()
    inj = [2]
    degree_only = run(claims=[claim("c1", "FN 43건", kind="degree")])
    assert E.extraction_hit("F01", degree_only, idx, inj) is False  # 숫자 결함에 정도 주장은 아니다
    assert E.extraction_hit("F07", degree_only, idx, inj) is True
    assert E.extraction_hit("C04", degree_only, idx, inj) is True  # C02~C09 는 어떤 종류든
    comparison = run(claims=[claim("c1", "FN 43건", kind="number")])
    assert E.extraction_hit("F05", comparison, idx, inj) is True  # 비교·방향은 number 도
    assert E.extraction_hit("F01", run(claims=[claim("c1", "0.91 이다")]), idx, inj) is False  # 다른 문장
    assert E.extraction_hit("F01", None, idx, inj) is None
    assert E.rate([True, False, None]) == {"k": 1, "n": 2, "rate": 0.5, "no_record": 1}


def test_miss_cause_pipeline_order():
    idx = index()
    inj = [2]
    nothing = run(gate="pass")
    assert E.miss_cause("F01", nothing, idx, inj, "confusion_counts.1.fn")["cause"] == "추출 누락"
    wrong_field = run(gate="pass", claims=[claim("c1", "FN 43건", field="confusion_counts.1.fp")])
    assert E.miss_cause("F01", wrong_field, idx, inj, "confusion_counts.1.fn")["cause"] == "필드 오지정"
    wrong_target = run(gate="pass", claims=[claim("c1", "FN 43건", field="confusion_counts.1.fn")],
                       verdicts=[verdict("c1", "FN 43건", verdict="match", expr="{confusion_counts.0.fn}")])
    out = E.miss_cause("F01", wrong_target, idx, inj, "confusion_counts.1.fn")
    assert out["cause"] == "대상 오답" and out["claims"][0]["answered_fields"] == ["confusion_counts.0.fn"]
    allowed = run(gate="pass", claims=[claim("c1", "FN 43건", field="confusion_counts.1.fn")],
                  verdicts=[verdict("c1", "FN 43건", verdict="match", expr="{confusion_counts.1.fn}", data=43)])
    out = E.miss_cause("F01", allowed, idx, inj, ["confusion_counts.1.fn"])
    assert out["cause"] == "규칙 허용" and out["claims"][0]["verdict"] == "match"
    # 근거 필드가 없으면(추론 결함) 2·3단계는 건너뛰고 규칙 허용
    inference = run(gate="pass", claims=[claim("c1", "FN 43건", kind="inference")],
                    verdicts=[verdict("c1", "FN 43건", verdict="inference_ok", kind="inference")])
    assert E.miss_cause("F08", inference, idx, inj, "")["cause"] == "규칙 허용"
    assert E.miss_cause("C05", allowed, idx, inj)["cause"] == "모순 미검출"
    # metric_id 로 답한 대상도 근거 필드와 맞춘다
    degree = run(gate="pass", claims=[claim("c1", "FN 43건", kind="degree", field="metric.M1.value")],
                 verdicts=[verdict("c1", "FN 43건", verdict="match", kind="degree", metric_id="M1")])
    assert E.miss_cause("F07", degree, idx, inj, "metric.M1.value")["cause"] == "규칙 허용"


# ── 6.9 숫자 토큰 ──


def test_numeric_tokens_exclusions():
    text = "F1 Score 0.85, Class9 와 M4 는 빼고 95% 신뢰구간과 4.4절, 표 3, 2,000회, 3번째는 뺀다. 목표 0.900 과 10%p 와 1,234.5건은 센다."
    assert E.numeric_tokens(text) == ["0.85", "0.900", "10%", "1,234.5"]
    assert E.numeric_tokens("95% 신뢰구간", exclude=False) == ["95%"]


def test_numeric_recall_counts_tokens_in_same_sentence_number_claims():
    idx = index()
    r = run(claims=[claim("c1", "0.91 이다"), claim("c2", "43건", kind="degree")])
    out = E.numeric_recall(r, idx)
    assert out["tokens"] == 3 and out["matched"] == 1  # 0.91 만(43건 둘은 숫자 주장이 아님)
    assert {m["token"] for m in out["missed"]} == {"43"}


# ── 베이스라인 ──


def test_baseline1_sentence_vs_section_level():
    found = [{"section": "7.1", "check": "numbers", "value": "44", "sentence_ids": [2]}]
    assert E.baseline1_detection(found, [2], ["7.1"])["status"] == "탐지"
    sec = [{"section": "7.1", "check": "d24_missing", "value": None, "sentence_ids": []}]
    assert E.baseline1_detection(sec, [2], ["7.1"])["status"] == "칸 단위 검출"
    assert E.baseline1_detection(sec, [2], ["8.2"])["status"] == "미탐지"
    elsewhere = [{"section": "7.1", "check": "numbers", "value": "0.91", "sentence_ids": [1]}]
    out = E.baseline1_detection(elsewhere, [2], ["7.1"])
    assert out["status"] == "미탐지" and out["elsewhere"] == elsewhere


def test_baseline2_detection_by_quote_sentence():
    idx = index()
    b2 = {"decision": "block", "issues": [{"axis": "faithfulness", "type": "숫자", "quotes": ["FN 43건이다"], "reason": "x"}]}
    out = E.baseline2_detection(b2, idx, [2])
    assert out["status"] == "탐지" and out["issues"] == 1
    assert E.baseline2_detection(b2, idx, [5])["status"] == "미탐지"
    assert E.baseline2_detection(None, idx, [5])["status"] == "미실행"


# ── 6.4 판정표 ──


def _entries():
    idx = index()
    a = [E.locate_flag(f, idx) for f in E.flags_of(run(verdicts=[verdict("C12", "FN 43건", field_x=1, expr="{x}", data=44)],
                                                     claims=[claim("C12", "FN 43건", field="x")]))]
    b = [E.locate_flag(f, idx) for f in E.flags_of(run(verdicts=[verdict("C7", "FN 43건", expr="{x}", data=44)],
                                                     contradictions=[contradiction("analysis_vs_finding", "문장 다는", "총평 둘은")]))]
    entries = E.sheet_entries("E-0001", "pima_strong", "원본", 1, a, idx)
    entries += E.sheet_entries("E-0007", "F01", "결함본", 2, b, idx)
    return entries, idx


def test_sheet_groups_same_sentence_kind_field_and_hides_sources():
    entries, idx = _entries()
    assert len(entries) == 3
    rows, key, items = E.build_sheet(entries, seed=5)
    assert len(rows) == 2  # 같은 문장·종류·필드의 위반 둘이 한 문항으로
    assert all(len(r) == len(E.SHEET_COLUMNS) for r in rows) and {r[0] for r in rows} == {"J001", "J002"}
    grouped = next(i for i in items if len(i["members"]) == 2)
    assert {m["run_id"] for m in grouped["members"]} == {"E-0001", "E-0007"}
    assert len(key) == 3 and {k[3] for k in key} == {"원본", "결함본"}
    allowed = [s.text for s in idx.sentences] + ["44"]
    assert rows and E.sheet_leaks(rows, doc_names=["pima_strong", "F01"], allowed_texts=allowed) == []
    pair_row = next(r for r in rows if r[1].startswith("모순"))
    assert pair_row[2] == "문장 다는 43건과 같다." and pair_row[3] == "총평 둘은 양호하다."
    assert pair_row[4] == "문장 나는 FN 43건이다." and pair_row[5] == ""  # 앞뒤 문장(같은 칸)
    # 시드가 같으면 같은 번호, 다르면 섞임이 바뀔 수 있다
    assert E.build_sheet(entries, seed=5)[0] == rows


@pytest.mark.parametrize("cell, what", [
    ("pima_strong 의 문장", "문서 이름"), ("yeast_strong", "문서 이름"), ("dev_doc_x", "문서 이름"),
    ("F07 결함", "결함·변형 ID"), ("V12", "결함·변형 ID"), ("C03", "결함·변형 ID"),
    ("E-0012", "실행 번호"), ("R-0003", "실행 번호"),
    ("2회차", "회차"), ("eval_r1", "회차"), ("r2", "회차"), ("round 1", "회차"),
    ("원본에서", "출처 낱말"), ("변형", "출처 낱말"), ("결함본", "출처 낱말"), ("base", "출처 낱말"), ("베이스라인", "출처 낱말"),
])
def test_leak_checker_catches_each_forbidden_class(cell, what):
    entries, idx = _entries()
    rows, _, _ = E.build_sheet(entries, seed=5)
    assert rows
    bad = [tuple(cell if c == 6 else v for c, v in enumerate(rows[0])), *rows[1:]]
    found = E.sheet_leaks(bad, doc_names=["dev_doc_x"], allowed_texts=[s.text for s in idx.sentences])
    assert any(what in p for p in found), found


def test_leak_checker_allows_source_words_inside_document_sentences_only():
    sentence = "원본 데이터의 변형 비율은 낮다."
    row = ("J001", "사실 일치 위반", sentence, "", "", "", "주장 종류 숫자", "", "", "", "")
    assert E.sheet_leaks([row], allowed_texts=[sentence]) == []
    assert E.sheet_leaks([row], allowed_texts=[]) != []
    tool = ("J001", "사실 일치 위반", sentence, "", "", "", "원본 문서", "", "", "", "")
    assert E.sheet_leaks([tool], allowed_texts=[sentence]) != []


def test_tally_overblocking_rules():
    key = [{"문항": "J001", "실행 번호": "E-1", "출처": "원본", "축": "faithfulness"},
           {"문항": "J002", "실행 번호": "E-2", "출처": "변형", "축": "faithfulness"},
           {"문항": "J003", "실행 번호": "E-2", "출처": "변형", "축": "consistency"},
           {"문항": "J004", "실행 번호": "E-3", "출처": "원본", "축": "faithfulness"},
           {"문항": "J005", "실행 번호": "E-9", "출처": "결함본", "축": "faithfulness"},
           {"문항": "J006", "실행 번호": "E-4", "출처": "원본", "축": "faithfulness"}]
    answers = {"J001": "판정자 오판", "J002": "판정자 오판", "J003": "실제 오류", "J004": "판정자 오판", "J005": "판정자 오판",
               "J006": ""}
    runs = {"E-1": {"doc": "a", "source": "원본", "gate": "blocked", "checked": 100},
            "E-2": {"doc": "V01", "source": "변형", "gate": "blocked", "checked": 100},
            "E-3": {"doc": "a", "source": "원본", "gate": "blocked", "checked": 50},
            "E-4": {"doc": "b", "source": "원본", "gate": "blocked", "checked": 50},
            "E-5": {"doc": "b", "source": "원본", "gate": "pass", "checked": 100},
            "E-9": {"doc": "F01", "source": "결함본", "gate": "blocked", "checked": 100}}
    out = E.tally_overblocking(key, answers, runs)
    assert out["runs"]["E-1"]["status"] == "과잉 차단" and out["runs"]["E-3"]["status"] == "과잉 차단"
    assert out["runs"]["E-2"]["status"] == "자연 발생 오류 검출" and out["runs"]["E-4"]["status"] == "판정 미완"
    assert out["counts"]["과잉 차단"] == 2 and out["counts"]["통과"] == 1 and out["counts"]["차단"] == 4
    assert out["claim_level"] == {"wrong_faithfulness_violations": 3, "checked": 400, "rate": 3 / 400}
    assert out["defect_out_of_place"]["판정자 오판"] == 1
    with pytest.raises(ValueError):
        E.tally_overblocking(key, {"J001": "모름"}, runs)


def test_prereg_tables_are_complete():
    assert len(E.PREREG_DEFECTS) == 18 and len(E.VARIANT_IDS) == 12
    assert sum(d.axis == "faithfulness" for d in E.PREREG_DEFECTS.values()) == 9
    for rep in E.EVAL_REPORTS:
        mine = [d for d in E.PREREG_DEFECTS.values() if d.report == rep]
        assert len(mine) == 6 and sum(d.axis == "faithfulness" for d in mine) == 3
    assert E.PREREG_DEFECTS["C01"].expected_axes == ("faithfulness", "consistency")
    assert E.ROUND_SEEDS == {1: 20261105, 2: 20261106} and (E.BUDGET_USD, E.BUDGET_RUNS) == (30.0, 110)
    assert E.SHEET_COLUMNS == ("문항", "표시 종류", "표시된 문장", "짝 문장(모순일 때)", "앞 문장", "뒤 문장", "판정자 사유",
                               "대상 필드", "필드 값", "판정(실제 오류/판정자 오판)", "판정 사유(필드 값과 함께 한 줄)")
