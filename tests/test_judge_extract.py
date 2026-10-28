"""tests/test_judge_extract.py — 공용 전처리 주장 추출(app/judge/extract.py, 심판 라운드 J2 ④, 계획 10-3). 가짜 OpenAI 로 배선만."""
from __future__ import annotations

import json

import pytest

from app.judge import extract
from app.judge.reading import Piece
from app.judge.sources import FieldEntry
from tests import judge_fakes as F

MARK = "ZQX고정표식"
PIECES = [
    Piece("slot:7.1", "7.1", "5.2", "ai", "Recall 은 0.912 로 목표 0.9 를 넘었고 FN 이 FP 보다 많다."),
    Piece("slot:8.2", "8.2", "5.1", "template", "종합 판정은 합격이다. 경계 근처 표본이 많음을 시사한다."),
    Piece("m:4.4-fixed17", "7.1", "4.4", "fixed", f"경계 오류는 12건이다. {MARK}"),
    Piece("m:4.4-counts", "7.1", "4.4", "table", f"클래스 | TP\n1 | 52 {MARK}"),
]
CATALOG = {n: FieldEntry(n, n, 0.5) for n in ("metric.M3.value", "metric.M3.threshold", "confusion_counts.1.fn",
                                              "confusion_counts.1.fp")}


def item(quote, piece="slot:7.1", kind="number", field="metric.M3.value", field_b="", number_text="0.912", polarity=""):
    return {"kind": kind, "quote": quote, "piece_id": piece, "field": field, "field_b": field_b,
            "number_text": number_text, "polarity": polarity}


def reply(*items):
    return F.json_response({"claims": list(items)}, model="gpt-6-luna")


async def run(*replies):
    fake = F.FakeOpenAI(list(replies))
    return fake, await extract.extract_claims(fake, PIECES, CATALOG)


async def test_one_self_check_when_nothing_new():
    fake, res = await run(reply(item("0.912")), reply())
    assert len(fake.calls) == 2 and res.rounds == [1, 1] and not res.stopped and res.failure is None
    c = res.claims[0]
    assert (c.id, c.slot, c.place, c.source, c.field, c.round) == ("C1", "7.1", "5.2", "ai", "metric.M3.value", 0)


async def test_at_most_two_self_checks_and_stop_flag():
    fake, res = await run(
        reply(item("0.912")),
        reply(item("목표 0.9", field="metric.M3.threshold", number_text="0.9")),
        reply(item("FN 이 FP 보다 많다", kind="comparison", field="confusion_counts.1.fn", field_b="confusion_counts.1.fp",
                   number_text="")),
    )
    assert len(fake.calls) == 3  # 3번째 자기 점검은 없다
    assert res.rounds == [1, 2, 3] and res.stopped
    assert [c.round for c in res.claims] == [0, 1, 2]
    assert res.claims[2].field_b == "confusion_counts.1.fp"


async def test_self_check_prompt_lists_found_claims_and_does_not_repeat():
    fake, res = await run(reply(item("0.912")), reply(item("0.912")))  # 같은 주장 되풀이 → 새것 아님
    assert len(res.claims) == 1 and len(fake.calls) == 2
    assert "이미 뽑은 주장" in fake.calls[1]["instructions"] and "[slot:7.1] number: 0.912" in fake.calls[1]["instructions"]
    assert "이미 뽑은 주장" not in fake.calls[0]["instructions"]


async def test_code_rejects_out_of_scope_piece_and_missing_quote():
    _, res = await run(reply(
        item("경계 오류는 12건이다", piece="m:4.4-fixed17"),  # 고정 문구 — 범위 밖
        item("0.913"),  # 조각에 없는 구절
        item("0.912", field="metric.M99.value"),  # 칸 목록에 없는 칸
    ), reply())
    assert [r["reason"] for r in res.rejected] == ["piece", "quote"]
    assert len(res.claims) == 1 and res.claims[0].field == "" and res.claims[0].unresolved_field


async def test_quote_match_ignores_whitespace_runs():
    _, res = await run(reply(item("Recall  은\n0.912")), reply())
    assert len(res.claims) == 1


async def test_only_ai_and_template_pieces_reach_the_model():
    fake, _ = await run(reply(item("0.912")), reply())
    sent = json.dumps(fake.calls, ensure_ascii=False)
    assert MARK not in sent
    assert "slot:7.1" in sent and "slot:8.2" in sent
    schema = fake.calls[0]["text"]["format"]["schema"]
    assert schema["properties"]["claims"]["items"]["properties"]["piece_id"]["enum"] == ["slot:7.1", "slot:8.2"]


async def test_instructions_come_from_claims_json_and_catalog_without_values():
    fake, _ = await run(reply(item("0.912")), reply())
    text = fake.calls[0]["instructions"]
    assert "시사한다" in text and "Precision 0.969, Recall 0.984" in text  # 주장 규칙(claims.json)
    assert "- metric.M3.value" in text and "0.5" not in text  # 칸 이름만, 값은 없다
    assert fake.calls[0]["model"] == "gpt-6-luna"


async def test_zero_claims_with_numbers_is_failure():
    _, res = await run(reply(), reply())
    assert res.failure == "no_claims"


async def test_zero_claims_without_numbers_is_fine():
    fake = F.FakeOpenAI([reply(), reply()])
    res = await extract.extract_claims(fake, [Piece("slot:7.3", "7.3", "5.6", "ai", "특이 사항이 없다.")], CATALOG)
    assert res.failure is None and res.claims == []


@pytest.mark.parametrize("bad, kind", [(F.response([F.message_item("x", refusal=True)]), "refusal"),
                                       (F.response([F.message_item("not json")]), "bad_output")])
async def test_call_failures_propagate(bad, kind):
    _, res = await run(bad)
    assert res.failure == kind and res.claims == []


async def test_self_check_failure_is_failure():
    _, res = await run(reply(item("0.912")), F.response([F.message_item("x", refusal=True)]))
    assert res.failure == "refusal" and len(res.calls) == 2
