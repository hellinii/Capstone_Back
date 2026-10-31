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


def verdicts(pieces, items):
    """문장마다 판정(1.17) — 문장마다 그 문장에 든 주장 수, 없으면 까닭."""
    out = []
    for s in extract.sentences_of(extract.for_module(pieces, "extract")):
        n = sum(sid == s.sid for sid in extract.place([s], [(i["piece_id"], i["quote"]) for i in items]))
        out.append({"sid": s.sid, "n_claims": n, "none_reason": "" if n else "주장 없음"})
    return out


def first(*items, pieces=PIECES, sentences=None):
    """첫 추출 답 — 문장 판정을 빠짐없이 붙인다(따로 주면 그대로)."""
    sents = verdicts(pieces, items) if sentences is None else sentences
    return F.json_response({"sentences": sents, "claims": list(items)}, model="gpt-6-luna")


async def run(*replies):
    fake = F.FakeOpenAI(list(replies))
    return fake, await extract.extract_claims(fake, PIECES, CATALOG)


async def test_one_self_check_when_nothing_new():
    fake, res = await run(first(item("0.912")), reply())
    assert len(fake.calls) == 2 and res.rounds == [1, 1] and not res.stopped and res.failure is None
    c = res.claims[0]
    assert (c.id, c.slot, c.place, c.source, c.field, c.round) == ("C1", "7.1", "5.2", "ai", "metric.M3.value", 0)


async def test_at_most_two_self_checks_and_stop_flag():
    fake, res = await run(
        first(item("0.912")),
        reply(item("목표 0.9", field="metric.M3.threshold", number_text="0.9")),
        reply(item("FN 이 FP 보다 많다", kind="comparison", field="confusion_counts.1.fn", field_b="confusion_counts.1.fp",
                   number_text="")),
    )
    assert len(fake.calls) == 3  # 3번째 자기 점검은 없다
    assert res.rounds == [1, 2, 3] and res.stopped
    assert [c.round for c in res.claims] == [0, 1, 2]
    assert res.claims[2].field_b == "confusion_counts.1.fp"


async def test_self_check_prompt_lists_found_claims_and_does_not_repeat():
    fake, res = await run(first(item("0.912")), reply(item("0.912")))  # 같은 주장 되풀이 → 새것 아님
    assert len(res.claims) == 1 and len(fake.calls) == 2
    second = fake.calls[1]["instructions"]
    assert "[s1] (slot:7.1) Recall 은 0.912" in second and "이미 뽑은 것: number: 0.912" in second
    assert "[s2] (slot:8.2) 종합 판정은 합격이다.\n    이미 뽑은 것: (아직 없음)" in second  # 1.17 표적 점검 — 문장마다
    assert "이미 뽑은 것" not in fake.calls[0]["instructions"]


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
    assert fake.calls[0]["model"] == "gpt-6-luna" and fake.calls[0]["reasoning"] == {"effort": "low"}


async def test_zero_claims_with_numbers_is_failure():
    _, res = await run(first(), reply())
    assert res.failure == "no_claims"


async def test_zero_claims_without_numbers_is_fine():
    piece = Piece("slot:7.3", "7.3", "5.6", "ai", "특이 사항이 없다.")
    fake = F.FakeOpenAI([first(pieces=[piece]), reply()])
    res = await extract.extract_claims(fake, [piece], CATALOG)
    assert res.failure is None and res.claims == []


@pytest.mark.parametrize("bad, kind", [(F.response([F.message_item("x", refusal=True)]), "refusal"),
                                       (F.response([F.message_item("not json")]), "bad_output")])
async def test_call_failures_propagate(bad, kind):
    _, res = await run(bad)
    assert res.failure == kind and res.claims == []


async def test_self_check_failure_is_failure():
    _, res = await run(first(item("0.912")), F.response([F.message_item("x", refusal=True)]))
    assert res.failure == "refusal" and len(res.calls) == 2


async def test_normative_inference_is_dropped_by_code():
    pieces = [Piece("slot:8.2", "8.2", "5.1", "ai", "별도 데이터에서 확인해야 한다. 경계 근처 표본이 많음을 시사한다.")]
    fake = F.FakeOpenAI([reply(item("별도 데이터에서 확인해야 한다", piece="slot:8.2", kind="inference", field="", number_text=""),
                               item("경계 근처 표본이 많음을 시사한다", piece="slot:8.2", kind="inference", field="", number_text="")),
                         reply()])
    res = await extract.extract_claims(fake, pieces, CATALOG)
    assert [c.quote for c in res.claims] == ["경계 근처 표본이 많음을 시사한다"]
    assert [r["reason"] for r in res.rejected] == ["normative"]


async def test_scope_sentence_inference_is_dropped_by_code():
    pieces = [Piece("slot:8.3", "8.3", "5.5", "ai", "응답 시간은 측정하지 않았으므로 지연 위험은 평가하지 않았다.")]
    fake = F.FakeOpenAI([reply(item("응답 시간은 측정하지 않았으므로", piece="slot:8.3", kind="inference", field="", number_text="")),
                         reply()])
    res = await extract.extract_claims(fake, pieces, CATALOG)
    assert res.claims == [] and [r["reason"] for r in res.rejected] == ["scope"]


# ── 1.17 문장마다 판정(빠진 문장만 한 번 다시 묻기) ─────────────────────────────


async def test_first_call_lists_sentences_and_asks_verdict_before_claims():
    fake, _ = await run(first(item("0.912")), reply())
    user = fake.calls[0]["input"][0]["content"]
    assert "## 문장 목록\n[s1] (slot:7.1) Recall 은 0.912" in user and "[s3] (slot:8.2) 경계 근처 표본이 많음을 시사한다." in user
    assert MARK not in user  # 고정 문구·표는 문장 목록에도 없다
    schema = fake.calls[0]["text"]["format"]["schema"]
    # 1.19: 주장이 먼저, 문장 판정은 점검표로 뒤에(1.18 R-0059 — 판정을 먼저 쓰게 하자 주장 목록을 비운 채 끝났다)
    assert list(schema["properties"]) == ["claims", "sentences"] and schema["required"] == ["claims", "sentences"]
    assert schema["properties"]["sentences"]["items"]["properties"]["sid"]["enum"] == ["s1", "s2", "s3"]
    assert "문장마다 판정" in fake.calls[0]["instructions"]
    # 자기 점검은 문장 판정을 받지 않는다
    assert "sentences" not in fake.calls[1]["text"]["format"]["schema"]["properties"]


async def test_fully_judged_first_answer_needs_no_fill():
    fake, res = await run(first(item("0.912")), reply())
    assert len(fake.calls) == 2 and res.sentence_checks == {"sentences": 3, "gaps": {}, "left": {}}


async def test_missing_sentences_are_asked_once_and_only_those():
    s1_only = [{"sid": "s1", "n_claims": 1, "none_reason": ""}]  # s2·s3 판정 없음, 주장도 없음
    fake, res = await run(
        first(item("0.912"), sentences=s1_only),
        reply(item("종합 판정은 합격이다", piece="slot:8.2", kind="name", field="", number_text="")),
        reply(),
    )
    assert [c["purpose"] for c in res.calls] == ["extract_0", "extract_0_fill", "extract_1"]
    fill = fake.calls[1]
    assert fill["text"]["format"]["schema"]["properties"]["sentences"]["items"]["properties"]["sid"]["enum"] == ["s2", "s3"]
    assert "[s2] (slot:8.2) 종합 판정은 합격이다." in fill["instructions"] and "[s1]" not in fill["instructions"].split("다시 읽을 문장")[1]
    assert res.sentence_checks["gaps"] == {"s2": "missing", "s3": "missing"}
    assert res.sentence_checks["left"] == {"s3": "missing"}  # 채우기 답이 s3 을 또 빠뜨렸다 — 기록만, 다시 묻지 않는다
    assert [c.quote for c in res.claims] == ["0.912", "종합 판정은 합격이다"]
    assert [c.round for c in res.claims] == [0, 0] and res.rounds == [2, 2]  # 채운 주장은 0번 시점
    assert "이미 뽑은 것: name: 종합 판정은 합격이다" in fake.calls[2]["instructions"]  # 자기 점검은 채운 뒤 목록으로


async def test_fill_is_asked_at_most_once():
    fake, res = await run(first(item("0.912"), sentences=[]), reply(), reply())
    assert len(fake.calls) == 3  # 첫 추출 + 채우기 1번 + 자기 점검 1번 — 채우기를 되풀이하지 않는다
    assert "[s2] (slot:8.2) 종합 판정은 합격이다.\n    이미 뽑은 것: (아직 없음)" in fake.calls[1]["instructions"]
    assert fake.calls[2]["instructions"].count("이미 뽑은 것") == 3 and "다시 읽을 문장" not in fake.calls[2]["instructions"]
    assert res.sentence_checks["gaps"] == {"s2": "missing", "s3": "missing"}  # s1 은 판정이 없어도 주장이 있어 본 것
    assert res.sentence_checks["left"] == {"s2": "missing", "s3": "missing"}


@pytest.mark.parametrize("said, items, want", [
    ({"s1": (2, "")}, [item("0.912")], {}),  # 2개라 적고 1개만 — 1.18 부터 다시 묻지 않는다(R-0058 채우기 압박)
    ({"s1": (1, "")}, [item("0.913")], {"s1": "lost"}),  # 인용이 틀려 하나도 안 들어감 — 본 것으로 치지 않는다
    ({"s1": (0, "")}, [], {"s1": "no_reason"}),  # 까닭 없는 '없음'
    ({"s1": (0, "권고 문장")}, [], {}),
    ({}, [item("0.912")], {}),  # 판정은 없어도 주장이 들어간 문장은 본 것
    ({"s1": (0, "")}, [item("0.912")], {}),  # '0개'라 했어도 주장이 들어갔으면 됐다
    ({"s1": (1, "")}, [item("FN 이 FP 보다 많다", kind="comparison")] * 2, {}),  # 되풀이로 버린 것도 본 것
    ({"s1": ("2", "")}, [], {"s1": "missing"}),  # 수가 정수가 아니면 판정이 없는 것과 같다
    ({"s1": (0, "빠뜨림")}, [], {"s1": "missed"}),  # 1.19 점검표에서 모델이 스스로 빠뜨렸다고 적음
    ({"s1": (0, "빠뜨림")}, [item("0.912")], {}),  # 그래도 주장이 들어갔으면 됐다
    ({"s1": (True, "")}, [], {"s1": "missing"}),
])
async def test_sentence_gaps(said, items, want):
    pieces = [PIECES[0]]
    sents = [{"sid": k, "n_claims": n, "none_reason": r} for k, (n, r) in said.items()]
    fake = F.FakeOpenAI([first(*items, pieces=pieces, sentences=sents), reply(), reply()])
    res = await extract.extract_claims(fake, pieces, CATALOG)
    assert res.sentence_checks["gaps"] == want
    assert len(fake.calls) == (3 if want else 2)


async def test_code_dropped_normative_counts_as_looked():
    pieces = [Piece("slot:8.2", "8.2", "5.1", "ai", "별도 데이터에서 확인해야 한다.")]
    norm = item("별도 데이터에서 확인해야 한다", piece="slot:8.2", kind="inference", field="", number_text="")
    fake = F.FakeOpenAI([first(norm, pieces=pieces), reply()])
    res = await extract.extract_claims(fake, pieces, CATALOG)
    assert res.claims == [] and res.sentence_checks["gaps"] == {} and len(fake.calls) == 2


async def test_fill_failure_keeps_first_extraction():
    # 검토 P1: 채우기는 덤 — 실패해도 첫 추출 주장은 살고 자기 점검으로 간다
    fake, res = await run(first(item("0.912"), sentences=[]), F.response([F.message_item("x", refusal=True)]), reply())
    assert res.failure is None and [c.quote for c in res.claims] == ["0.912"] and len(res.calls) == 3
    assert res.sentence_checks["fill_failure"] == "refusal"


async def test_fill_is_skipped_when_time_is_short():
    t = [0.0]
    fake = F.FakeOpenAI([first(item("0.912"), sentences=[]), reply()])
    res = await extract.extract_claims(fake, PIECES, CATALOG, deadline=extract.FILL_MIN_SECONDS - 1, clock=lambda: t[0])
    assert len(fake.calls) == 2 and res.sentence_checks["fill_skipped"] == "deadline"
    fake = F.FakeOpenAI([first(item("0.912"), sentences=[]), reply(), reply()])
    res = await extract.extract_claims(fake, PIECES, CATALOG, deadline=extract.FILL_MIN_SECONDS + 1, clock=lambda: t[0])
    assert len(fake.calls) == 3 and "fill_skipped" not in res.sentence_checks


def test_same_text_in_two_sentences_counts_once():
    # 검토 P1: '1건' 하나로 '171건' 문장까지, 되풀이 문장까지 본 것으로 쳤다 — 주장 하나는 한 문장에만
    piece = Piece("slot:7.1", "7.1", "5.2", "ai", "FN 은 1건이다. 전체는 171건이다. Recall 은 0.91 이다. Recall 은 0.91 이다.")
    sents = extract.sentences_of([piece])
    assert extract.place(sents, [("slot:7.1", "1건")]) == ["s1"]
    assert extract.place(sents, [("slot:7.1", "1건")] * 2) == ["s1", "s1"]  # '171건' 안은 자리가 아니다 — 남는 것은 마지막 자리로
    assert not extract._in_sentence("slot:7.1", "1건", sents[1])
    assert extract.place(sents, [("slot:7.1", "Recall 은 0.91"), ("slot:7.1", "Recall 은 0.91")]) == ["s3", "s4"]
    cl = [extract.Claim("", "number", q, "slot:7.1", "7.1", "5.2", "ai", "", "", "", "", 0) for q in ("1건", "Recall 은 0.91")]
    said = [{"sid": f"s{i}", "n_claims": 1, "none_reason": ""} for i in range(1, 5)]
    assert extract.sentence_gaps(sents, said, cl, []) == {"s2": "lost", "s4": "lost"}
    assert "전체는 171건이다.\n    이미 뽑은 것: (아직 없음)" in extract.by_sentence(sents, cl)


async def test_fill_keeps_only_claims_inside_asked_sentences():
    rest = [{"sid": "s2", "n_claims": 0, "none_reason": "x"}, {"sid": "s3", "n_claims": 0, "none_reason": "x"}]  # s1 판정 빠짐
    fake, res = await run(
        first(sentences=rest),
        reply(item("종합 판정은 합격이다", piece="slot:8.2", kind="name", field="", number_text=""),  # 다시 묻지 않은 문장
              item("FN 이 FP 보다 많다", kind="comparison", field="", number_text="")),  # 다시 물은 s1 의 새 주장
        reply(),
    )
    assert [c.quote for c in res.claims] == ["FN 이 FP 보다 많다"]
    assert [r["reason"] for r in res.rejected] == ["fill_outside"]
    assert res.sentence_checks["gaps"] == {"s1": "missing"} and res.sentence_checks["filled"] == 1
    assert res.sentence_checks["left"] == {}
    assert "[s2]" not in fake.calls[1]["instructions"].split("다시 읽을 문장")[1]


async def test_sentence_checks_reach_the_record():
    _, res = await run(first(item("0.912")), reply())
    assert res.as_record()["sentence_checks"]["sentences"] == 3


def test_scope_without_sentences_has_no_sentence_field():
    assert "sentences" not in extract.response_schema(["slot:7.1"])["properties"]
    assert extract.sentences_of([Piece("slot:7.1", "7.1", "5.2", "ai", "  ")]) == []


# ── 1.17 주장의 범위·표적 점검 지시문 ────────────────────────────────────


def test_extract_prompt_widens_and_narrows_claims():
    text = extract.instructions({})
    assert "n_claims" in text and "맨 지표 이름" in text and "억지로 뽑지 않는다" in text and "적은 수만큼" not in text
    assert "어떻게 계산되는지만 풀이하는 말은 주장이 아니다" in text
    for word in ("명사를 꾸미는 말", "'~중 ~개'", "확인 대상은 계산값만이 아니다", "의뢰자가 적은 값(용도, 운영 방식, 데이터 출처", "지시어 그대로",
                 "이것이 보류의 사유이다", "같은 구절을 비교로 또 뽑지 않는다", "분야 일반론", "먼저 claims 에 주장을 모두 적는다",
                 "none_reason 에 '빠뜨림'"):
        assert word in text, word


def test_self_check_prompt_targets_five_kinds():
    text = extract.criteria.load_text("prompts/extract_self_check.md")
    for word in ("꾸밈말", "비교·집계", "판정과 사유", "의뢰자 입력값", "정도·추론", "(아직 없음)", "{by_sentence}"):
        assert word in text, word
    assert "{found}" not in text and "{empty}" not in text



def test_fill_prompt_names_every_reask_reason_and_does_not_push():
    # 1.20: 채우기 지시문이 1.19 의 다시 묻는 까닭(lost·missed)을 모두 말하고, 억지로 뽑지 말라고 한다
    text = extract.criteria.load_text("prompts/extract_fill.md")
    for word in ("판정이 빠졌거나", "하나도 들어가지 않았거나", "'빠뜨림'", "까닭이 비어", "억지로 뽑지 않는다", "{by_sentence}"):
        assert word in text, word
