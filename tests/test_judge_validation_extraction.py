"""tests/test_judge_validation_extraction.py — 추출 재현율·정밀도 짝짓기(app/judge/validation/extraction.py, 계획 17-4).

시험 자료는 합성 문장이다(봉인 서술·판 3 발급본을 쓰지 않는다). 표 꼴은 annotation.py 의 열 이름을 그대로 쓴다.
"""
from __future__ import annotations

import pytest

from app.judge.extract import Claim
from app.judge.validation import extraction as X
from app.judge.validation.annotation import CLAIM_COLUMNS, SENTENCE_COLUMNS

S1 = "Precision 0.969, Recall 0.984로 Recall이 0.015 높다."
S2 = "오류는 클래스 3에 몰려 있다."
S3 = "전체 정확도는 0.912이다."
SENTENCES = [  # annotation.sentence_rows 꼴(튜플)
    (1, "p71", "7.1", "분석-1", "ai", S1),
    (2, "p71", "7.1", "분석-1", "ai", S2),
    (3, "p81", "8.1", "소견-1", "template", S3),
]


def gold_row(no, kind, quote):
    return dict(zip(CLAIM_COLUMNS, (str(no), kind, quote, "")))


def pred(kind, quote, piece="p71", slot="7.1", number_text="", rnd=0):
    return Claim("", kind, quote, piece, slot, "분석-1", "ai", "", "", number_text, "", rnd)


def gold(*rows):
    return X.gold_from_csv([gold_row(*r) for r in rows], SENTENCES)


def preds(*claims):
    return X.pred_from_extract(list(claims), SENTENCES)


# ---------------------------------------------------------------- 표 읽기

def test_gold_from_csv_reads_sentence_table_and_spans():
    g = gold((1, "number", "0.969"), (3, "number", "0.912"))
    assert g[0]["slot"] == "7.1" and g[0]["sentence"] == 1 and g[0]["span"] == [S1.index("0.969"), S1.index("0.969") + 5]
    assert g[1]["slot"] == "8.1" and g[1]["piece"] == "p81"
    as_dicts = [dict(zip(SENTENCE_COLUMNS, (str(r[0]), *r[1:]))) for r in SENTENCES]  # csv.DictReader 꼴(번호가 글)
    assert X.gold_from_csv([gold_row(1, "number", "0.969")], as_dicts)[0]["sentence"] == 1
    assert gold((1, "name", "없는 구절"))[0]["span"] is None
    with pytest.raises(ValueError):
        gold((9, "number", "0.1"))


def test_pred_from_extract_locates_sentence():
    p = preds(pred("number", "0.912", piece="p81", slot="8.1"), pred("degree", "몰려 있다"),
              pred("comparison", "Recall이 0.015 높다. 오류는 클래스", rnd=1), pred("name", "여기 없음"))
    assert (p[0]["sentence"], p[1]["sentence"]) == (3, 2) and p[0]["span"] is not None
    assert p[2]["sentence"] == 1 and p[2]["span"] is None and p[2]["round"] == 1  # 문장 경계를 넘은 인용
    assert p[3]["sentence"] is None
    rec = {"id": "C9", "kind": "number", "quote": "0.969", "piece_id": "p71", "slot": "7.1", "number_text": "0.969", "round": 2}
    assert X.pred_from_extract([rec], SENTENCES)[0]["id"] == "C9"  # as_record 의 dict 꼴도 받는다


def test_numbers():
    assert X.numbers("0.969, 1,234 그리고 −0.5") == {0.969, 1234.0, -0.5}
    assert X.numbers("F1 Score 와 Class9") == set()


# ---------------------------------------------------------------- 짝짓기 조건 다섯

def test_condition_same_slot():
    g = gold((1, "number", "0.969"))
    p = preds(pred("number", "0.969"))
    assert len(X.match_claims(g, p)["pairs"]) == 1
    p[0]["slot"] = "7.2"
    assert X.match_claims(g, p)["pairs"] == []


def test_condition_same_sentence_and_span_overlap():
    g = gold((1, "degree", "Recall이 0.015 높다"))
    p = preds(pred("degree", "0.015 높다"))
    assert len(X.match_claims(g, p)["pairs"]) == 1
    other = preds(pred("degree", "Precision 0.969"))  # 같은 문장, 다른 범위
    assert X.match_claims(g, other)["pairs"] == []
    elsewhere = preds(pred("degree", "몰려 있다"))  # 같은 칸, 다른 문장
    assert X.match_claims(gold((1, "degree", "몰려")), elsewhere)["pairs"] == []
    no_span = gold((1, "degree", "적어 둔 구절이 문장에 없음"))  # 범위를 모르면 문장만 본다
    assert len(X.match_claims(no_span, other)["pairs"]) == 1
    lost = preds(pred("degree", "여기 없음"))
    lost[0]["slot"] = "7.1"
    assert X.match_claims(gold((1, "degree", "높다")), lost)["pairs"] == []  # 문장을 못 찾은 추출은 짝 없음


def test_condition_number_value_and_kind():
    g = gold((1, "number", "0.984"))
    assert len(X.match_claims(g, preds(pred("number", "Recall 0.984", number_text="0.984")))["pairs"]) == 1
    wrong = preds(pred("number", "Recall 0.984", number_text="0.948"))  # 범위는 겹치지만 값이 다름
    assert X.match_claims(g, wrong)["pairs"] == []
    as_comparison = preds(pred("comparison", "Recall 0.984"))  # 종류만 다르면 짝(민서 결정 2026-10-31 — 이름표 차이)
    assert len(X.match_claims(g, as_comparison)["pairs"]) == 1
    assert len(X.match_claims(gold((1, "comparison", "Recall 0.984")), preds(pred("number", "0.984")))["pairs"]) == 1


def test_condition_kind_no_longer_blocks_a_pair():
    """V1 결정 10③(2026-10-29)은 숫자·비교 주장은 종류가 같아야 짝이었다. 민서 결정(2026-10-31, 결과를 본 뒤 — 까닭은 검증
    README 기록): 같은 자리를 짚었는데 종류만 다르면 이름표 차이라 짝으로 센다. 같은 종류면 덧점으로 먼저 짝지어진다."""
    g = gold((1, "comparison", "Recall이 0.015 높다"))
    assert len(X.match_claims(g, preds(pred("comparison", "Recall이 0.015 높다")))["pairs"]) == 1
    assert len(X.match_claims(g, preds(pred("degree", "Recall이 0.015 높다")))["pairs"]) == 1
    assert len(X.match_claims(gold((2, "degree", "몰려 있다")), preds(pred("comparison", "몰려 있다")))["pairs"]) == 1
    assert X.STRICT_KINDS == ()
    # 같은 종류 후보가 있으면 그쪽과 짝(덧점) — 다른 종류 후보는 남는다
    both = preds(pred("degree", "Recall이 0.015 높다"), pred("comparison", "Recall이 0.015 높다"))
    m = X.match_claims(g, both)
    assert m["pairs"][0][1]["kind"] == "comparison" and [p["kind"] for p in m["unmatched_pred"]] == ["degree"]


def test_gold_has_no_target_column_so_target_condition_never_runs():
    """V1 결정 10②: 정답지 claims.csv 에 '대상' 열이 없다 — 정답 쪽 target 은 늘 빈칸이라 추출 쪽 target 이 무엇이든
    조건 ④는 돌지 않는다(짝은 칸·문장 겹침·숫자 값·종류·일대일로만)."""
    assert "대상" not in CLAIM_COLUMNS
    g = gold((1, "number", "0.969"))
    assert g[0]["target"] == ""
    p = preds(pred("number", "0.969"))
    p[0]["target"] = "recall"
    assert len(X.match_claims(g, p)["pairs"]) == 1


def test_condition_target():
    g = gold((1, "number", "0.969"))
    p = preds(pred("number", "0.969"))
    g[0]["target"], p[0]["target"] = "precision", "recall"
    assert X.match_claims(g, p)["pairs"] == []
    p[0]["target"] = "precision"
    assert len(X.match_claims(g, p)["pairs"]) == 1
    p[0]["target"] = ""  # 한쪽만 있으면 비교하지 않는다
    assert len(X.match_claims(g, p)["pairs"]) == 1


def test_one_to_one_is_optimal_not_greedy():
    # 정답 G1(넓은 비교 구절)은 추출 A·B 둘 다와 짝이 될 수 있고, G2 는 A 하고만 된다.
    # 탐욕(첫 정답부터 더 많이 겹치는 것)은 G1–A 를 골라 G2 를 놓친다. 최적은 G1–B, G2–A 로 둘 다.
    g = gold((1, "comparison", "Recall 0.984로 Recall이 0.015 높다"), (1, "comparison", "Recall 0.984로"))
    p = preds(pred("comparison", "Recall 0.984로 Recall이"), pred("comparison", "0.015 높다"))
    m = X.match_claims(g, p)
    assert len(m["pairs"]) == 2 and m["unmatched_gold"] == [] and m["unmatched_pred"] == []
    assert {(a["id"], b["quote"]) for a, b in m["pairs"]} == {("G1", "0.015 높다"), ("G2", "Recall 0.984로 Recall이")}


def test_one_to_one_duplicate_predictions():
    g = gold((1, "number", "0.969"))
    p = preds(pred("number", "0.969"), pred("number", "Precision 0.969", number_text="0.969"))
    m = X.match_claims(g, p)
    assert len(m["pairs"]) == 1 and len(m["unmatched_pred"]) == 1  # 한 정답에 추출 하나만


def test_tie_prefers_same_kind_and_same_kind_option():
    g = gold((2, "degree", "몰려 있다"))
    p = preds(pred("inference", "몰려 있다"), pred("degree", "몰려 있다"))
    m = X.match_claims(g, p)
    assert m["pairs"][0][1]["kind"] == "degree"
    assert X.match_claims(g, preds(pred("inference", "몰려 있다")), same_kind=True)["pairs"] == []
    assert len(X.match_claims(g, preds(pred("inference", "몰려 있다")))["pairs"]) == 1


# ---------------------------------------------------------------- 재현율·정밀도

def test_recall_precision_and_edges():
    g = gold((1, "number", "0.969"), (1, "number", "0.984"), (3, "number", "0.912"))
    p = preds(pred("number", "0.969"), pred("number", "0.984"), pred("degree", "몰려 있다"))
    rp = X.recall_precision(X.match_claims(g, p))
    assert rp == {"recall": 2 / 3, "precision": 2 / 3, "n_gold": 3, "n_pred": 3, "n_pairs": 2}
    assert X.recall_precision(X.match_claims([], p))["recall"] is None
    assert X.recall_precision(X.match_claims(g, []))["precision"] is None
    empty = X.match_claims([], [])
    assert empty["pairs"] == [] and X.recall_precision(empty)["n_pairs"] == 0


def test_by_round_cumulative():
    g = gold((1, "number", "0.969"), (1, "number", "0.984"), (3, "number", "0.912"))
    p = preds(pred("number", "0.969", rnd=0), pred("number", "0.984", rnd=1),
              pred("number", "0.912", piece="p81", slot="8.1", rnd=2))
    br = X.by_round(g, p)
    assert [br[k]["recall"] for k in (0, 1, 2)] == [1 / 3, 2 / 3, 1.0]
    assert [br[k]["n_pred"] for k in (0, 1, 2)] == [1, 2, 3]


def test_runs_distribution():
    g = gold((1, "number", "0.969"), (1, "number", "0.984"))
    runs = [preds(pred("number", "0.969")), preds(pred("number", "0.969"), pred("number", "0.984")),
            preds(pred("number", "0.969"), pred("number", "0.984", rnd=1), pred("degree", "몰려 있다", rnd=2))]
    d = X.runs_distribution(g, runs)
    assert d["recall"] == {"min": 0.5, "median": 1.0, "max": 1.0, "n": 3}
    assert d["precision"]["min"] == pytest.approx(2 / 3) and d["precision"]["max"] == 1.0
    assert d["by_round"][0]["recall"] == {"min": 0.5, "median": 0.5, "max": 1.0, "n": 3}
    assert len(d["runs"]) == 3
    with pytest.raises(ValueError):
        X.runs_distribution(g, [])
    assert X.distribution([None, None]) == {"min": None, "median": None, "max": None, "n": 0}


# ---------------------------------------------------------------- 사람이 놓친 주장

def test_missed_rows_and_confirmed_additions():
    g = gold((1, "number", "0.969"))
    p = preds(pred("number", "0.969"), pred("degree", "몰려 있다"), pred("name", "여기 없음"))
    m = X.match_claims(g, p)
    rows = X.missed_rows(m)
    assert rows and len(rows) == 2 and all(len(r) == len(X.MISSED_COLUMNS) for r in rows)
    assert rows[0][1:5] == (2, "7.1", "degree", "몰려 있다") and rows[0][5] == ""
    assert rows[1][1] == ""  # 문장을 못 찾음 → 사람이 적는다
    judged = [dict(zip(X.MISSED_COLUMNS, r)) for r in rows]
    judged[0][X.MISSED_COLUMNS[5]] = "예"
    judged[1][X.MISSED_COLUMNS[5]] = "아니오"
    added = X.confirmed_additions(judged)
    assert len(added) == 1 and added[0]["인용 구절"] == "몰려 있다" and added[0]["메모"] == X.ADDED_MEMO
    regold = X.gold_from_csv([gold_row(1, "number", "0.969"), *added], SENTENCES)
    assert regold[1]["added"] is True and regold[0]["added"] is False
    assert X.recall_precision(X.match_claims(regold, p))["n_pairs"] == 2
    judged[1][X.MISSED_COLUMNS[5]] = "예"
    with pytest.raises(ValueError):  # 문장번호 없이 보탤 수 없다
        X.confirmed_additions(judged)
    judged[1][X.MISSED_COLUMNS[5]] = "글쎄"
    with pytest.raises(ValueError):
        X.confirmed_additions(judged)
    assert X.missed_rows(X.match_claims([], [])) == []


# ── 2026-10-31 세션 7: 짧은 숫자 인용의 문장 위치 ──────────────────────

S2 = [(1, "slot:7.1", "7.1", "5.2", "ai", "가상 표본 171건 중 3건이 틀렸다."),
      (2, "slot:7.1", "7.1", "5.2", "ai", "가상 FP 는 2건, FN 은 1건이다."),
      (3, "slot:7.1", "7.1", "5.2", "ai", "가상 클래스 3 은 약하고 클래스 3 의 표본은 3건이다.")]


def _pred(quote, kind="number", number_text=""):
    return X.pred_from_extract([Claim("C1", kind, quote, "slot:7.1", "7.1", "5.2", "ai", "", "", number_text, "", 0)], S2)[0]


def test_number_quote_is_not_found_inside_another_number():
    assert X._span("1건", "가상 표본 171건 중 3건이 틀렸다.") is None
    assert X._span("1건", "가상 FP 는 2건, FN 은 1건이다.") is not None
    assert X._span("3", "가상 0.3 과 3.5") is None and X._span("3", "가상 3 과 4") == [3, 4]
    p = _pred("1건", number_text="1")
    assert p["sentence"] == 2 and set(p["candidates"]) == {2}


def test_quote_in_several_sentences_pairs_with_gold_sentence():
    p = _pred("3건", number_text="3")
    assert set(p["candidates"]) == {1, 3}
    gold = X.gold_from_csv([{"문장번호": "3", "종류": "number", "인용 구절": "3건", "메모": ""}],
                           [dict(zip(SENTENCE_COLUMNS, (str(r[0]), *r[1:]))) for r in S2])
    m = X.match_claims(gold, [p])
    assert len(m["pairs"]) == 1 and m["unmatched_gold"] == [] and m["unmatched_pred"] == []
