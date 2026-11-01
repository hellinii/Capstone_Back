"""tests/test_judge_consistency.py — 일관성(app/judge/consistency.py, 심판 라운드 J3, 계획 11-3)."""
from __future__ import annotations

import json
from types import SimpleNamespace as M

from app.judge import consistency as co
from app.judge.extract import Claim
from app.judge.reading import Piece
from tests import judge_fakes as F

FACT_MARK = "ZQX사실표식"
PIECES = [
    Piece("slot:7.1", "7.1", "5.2", "ai", "FN 은 43건이다. 가장 약한 클래스는 0 이다."),
    Piece("slot:8.2", "8.2", "5.1", "ai", "종합 판정은 불합격이다. 보완은 필요하지 않다."),
    Piece("slot:9:6.2", "9", "6.2", "ai", "Recall 개선을 위해 재학습을 권한다."),
    Piece("m:4.4-counts", "7.1", "4.4", "table", "클래스 | FN\n1 | 43", (("43", "confusion_counts.1.fn"),)),
    Piece("m:4.2-table", "8", "4.2", "table", "종합 판정 | 충족\n불합격 | 4개 중 0개",
          (("4", "verdict.judged_count"), ("0", "verdict.passed_count"))),
]
FS = M(verdict="FAIL", metrics=[M(display_name="Accuracy", threshold=0.9, status="fail", value=0.5, f=FACT_MARK)])


def num(cid, text, fld, piece="slot:7.1"):
    p = next(p for p in PIECES if p.id == piece)
    return Claim(cid, "number", text, piece, p.slot, p.place, p.source, fld, "", text, "", 0)


def check(t, present=False, qa="", pa="", qb="", pb=""):
    return {"type": t, "present": present, "quote_a": qa, "piece_a": pa, "quote_b": qb, "piece_b": pb}


def none_found():
    return F.json_response({"checks": [check(t) for t in ("verdict_vs_recommendation", "weakness_vs_recommendation",
                                                            "analysis_vs_finding")]})


async def run(claims, *replies, pieces=PIECES):
    fake = F.FakeOpenAI(list(replies))
    return fake, await co.judge_consistency(fake, pieces, claims, FS)


async def test_value_group_match_and_mismatch_by_code():
    _, ok = await run([num("C1", "43", "confusion_counts.1.fn")], none_found())
    assert ok.contradictions == [] and ok.counts["value_groups_checked"] == 1
    _, bad = await run([num("C1", "44", "confusion_counts.1.fn")], none_found())
    c = bad.contradictions[0]
    assert (c.type, c.pieces, c.sources, c.quotes) == ("value_mismatch", ["slot:7.1", "m:4.4-counts"], ["ai", "table"], ["44", "43"])
    assert bad.counts["contradictions"] == 1 and bad.counts["checked"] == 1 + 3


async def test_alias_fields_group_together():
    claims = [num("C1", "5", "facts.verdict_detail.judged_count", piece="slot:8.2")]  # 표는 별칭 verdict.judged_count
    _, res = await run(claims, none_found())
    assert res.contradictions[0].detail["field"] == "facts.verdict_detail.judged_count"


async def test_single_entry_groups_are_not_checked():
    _, res = await run([], none_found())
    assert res.counts["value_groups_checked"] == 0 and res.counts["checked"] == 3


async def test_logic_contradiction_with_citations():
    reply = F.json_response({"checks": [
        check("verdict_vs_recommendation", True, "종합 판정은 불합격이다", "slot:8.2", "보완은 필요하지 않다", "slot:8.2"),
        check("weakness_vs_recommendation"), check("analysis_vs_finding")]})
    _, res = await run([], reply)
    c = res.contradictions[0]
    assert (c.type, c.slots) == ("verdict_vs_recommendation", ["8.2", "8.2"])


async def test_uncited_is_asked_again_once_then_failure():
    bad = F.json_response({"checks": [check("verdict_vs_recommendation", True, "없는 구절", "slot:8.2", "", ""),
                                      check("weakness_vs_recommendation"), check("analysis_vs_finding")]})
    again_bad = F.json_response({"checks": [check("verdict_vs_recommendation", True, "", "", "", "")]})
    fake, res = await run([], bad, again_bad)
    assert len(fake.calls) == 2 and res.failure == "uncited"
    assert "verdict_vs_recommendation" in fake.calls[1]["instructions"].split("다시 답하라:")[1]


async def test_uncited_then_withdrawn_is_fine():
    bad = F.json_response({"checks": [check("verdict_vs_recommendation", True, "", "", "", ""),
                                      check("weakness_vs_recommendation"), check("analysis_vs_finding")]})
    withdrawn = F.json_response({"checks": [check("verdict_vs_recommendation", False)]})
    _, res = await run([], bad, withdrawn)
    assert res.failure is None and res.contradictions == []


async def test_missing_type_answer_is_bad_output():
    _, res = await run([], F.json_response({"checks": [check("verdict_vs_recommendation")]}))
    assert res.failure == "bad_output"


async def test_llm_gets_text_and_verdict_only():
    fake, _ = await run([num("C1", "43", "confusion_counts.1.fn")], none_found())
    sent = json.dumps(fake.calls, ensure_ascii=False)
    assert FACT_MARK not in sent  # 사실 시트 원본은 가지 않는다
    assert "confusion_counts.1.fn" not in sent  # 주장 목록·칸 이름도 가지 않는다
    assert "m:4.4-counts" in sent and "종합 판정: FAIL" in sent and "기준 미달 지표: Accuracy" in sent
    assert fake.calls[0]["model"] == "gpt-6.1-sol"


async def test_misassigned_claim_number_is_skipped_but_non_data_number_counts():
    data = {"confusion_counts.1.fn": 43, "facts.ftg.best_recall": 44}
    claims = [num("C1", "44", "confusion_counts.1.fn")]  # 44 는 다른 칸 값과 맞다 — 칸을 잘못 짚음
    fake = F.FakeOpenAI([none_found()])
    res = await co.judge_consistency(fake, PIECES, claims, FS, data=data)
    assert res.contradictions == [] and res.counts["value_skipped_misassigned"] == 1
    fake = F.FakeOpenAI([none_found()])
    res = await co.judge_consistency(fake, PIECES, [num("C1", "47", "confusion_counts.1.fn")], FS, data=data)
    assert [c.type for c in res.contradictions] == ["value_mismatch"]  # 어느 데이터와도 안 맞는 숫자는 센다
