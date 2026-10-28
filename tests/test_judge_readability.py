"""tests/test_judge_readability.py — 표기(app/judge/readability.py, 심판 라운드 J4, 계획 11-5)."""
from __future__ import annotations

import json
from types import SimpleNamespace as M

from app.judge import readability as rd
from app.judge.reading import Piece, for_module
from tests import judge_fakes as F

VALUE_MARK = 0.123456789
FS = M(metrics=[M(display_name="F1 Score", value=VALUE_MARK), M(display_name="Accuracy", value=VALUE_MARK)],
       per_class=[M(label="cat")], confusion_counts=[M(label="dog")], distribution=M(class_distribution={"bird": 3}))
CLIENT = {"model": {"모델명": "ChurnAlert-RB"}, "environment": {"소프트웨어": "Python 3.11"}}
MARK = "ZQX고정표식"
PIECES = [
    Piece("slot:7.1", "7.1", "5.2", "ai", "F1 Score 는 0.497 이고 cat 클래스가 가장 약하다. FN 이 많다."),
    Piece("slot:8.2", "8.2", "5.1", "template", "M4 가 목표에 미달합니다. 재학습 필요함.\n- value_text 확인"),
    Piece("rec:6.2:P1:0:row", "9", "6.2", "fixed", "우선순위: 높음 / 대응 문제: P1 · 핵심 지표 미달"),
    Piece("rec:6.2:P1:0", "9", "6.2", "ai", "권고: ChurnAlert-RB 재학습 / 기대 효과: Recall 개선"),
    Piece("m:5.1-fixed11", "8.2", "5.1", "fixed", f"고정 문구 {MARK}. 합니다"),
]


def test_romanization_allow_list_comes_from_names_and_client_input():
    allowed = rd.allowed_latin(FS, CLIENT)
    assert {"F1", "Score", "Accuracy", "cat", "dog", "bird", "TP", "FN", "ChurnAlert-RB", "Python"} <= allowed
    found = rd.romanization(for_module(PIECES, "readability"), allowed)
    assert sorted(f.text for f in found) == ["M4", "Recall", "value_text"]  # 프론트 칸의 'P1' 은 읽지 않는다


def test_sentence_style_only_on_narrative_slots():
    found = rd.sentence_style(PIECES)
    texts = [f.text for f in found]
    assert "M4 가 목표에 미달합니다." in texts and "재학습 필요함." in texts and "- value_text 확인" in texts
    assert all(f.piece_id.startswith("slot:") for f in found)
    assert not any(f.piece_id == "slot:7.1" for f in found)


def ans(slot, present=False, quote="", piece=""):
    return {"slot": slot, "present": present, "quote": quote, "piece_id": piece}


async def _run(*replies):
    fake = F.FakeOpenAI(list(replies))
    return fake, await rd.judge_readability(fake, PIECES, FS, CLIENT)


async def test_counts_and_llm_flags():
    reply = F.json_response({"slots": [ans("7.1", True, "FN 이 많다.", "slot:7.1"), ans("8.2"), ans("9")]})
    fake, res = await _run(reply)
    assert res.counts["rules"] == 2 and res.counts["violations"] == len(res.findings) > 0
    assert [(f.slot, f.text) for f in res.llm_flags] == [("7.1", "FN 이 많다.")]
    sent = json.dumps(fake.calls, ensure_ascii=False)
    assert MARK not in sent and str(VALUE_MARK) not in sent  # 고정 문구·데이터는 가지 않는다


async def test_uncited_flag_asked_again_then_failure():
    bad = F.json_response({"slots": [ans("7.1", True, "없는 문장", "slot:7.1"), ans("8.2"), ans("9")]})
    again = F.json_response({"slots": [ans("7.1", True, "", "")]})
    fake, res = await _run(bad, again)
    assert len(fake.calls) == 2 and res.failure == "uncited"


async def test_missing_slot_answer_is_bad_output():
    _, res = await _run(F.json_response({"slots": [ans("7.1")]}))
    assert res.failure == "bad_output"


def test_pilot_false_positives_are_gone():
    fs = M(metrics=[M(display_name="Recall", value=0.5)], per_class=[], confusion_counts=[], distribution=None,
           problems=[M(id="P1"), M(id="P4")])
    allowed = rd.allowed_latin(fs, {}, ["내부 참조 기준치 v1 (공개 벤치마크 평균이 아닌 자체 설정 기준)"])
    pieces = [Piece("slot:7.2", "7.2", "5.3", "template", "비율 차가 10%p 이상이다. 권고(P4)와 같은 기준치 v1 이다.")]
    assert rd.romanization(pieces, allowed) == []
    tail = [Piece("slot:8.2", "8.2", "5.1", "template",
                  "종합 판정은 '최종 불합격'이다(미달한 핵심 지표: Recall(0.547, 목표 0.900)).")]
    assert rd.sentence_style(tail) == []
    assert [f.text[:6] for f in rd.sentence_style([Piece("slot:8.2", "8.2", "5.1", "ai", "미달함(Recall).")])] == ["미달함(Re"]
