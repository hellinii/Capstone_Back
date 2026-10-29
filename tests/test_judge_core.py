"""tests/test_judge_core.py — 합산·게이트(aggregate)·안전장치(safety)·채점 한 번(core.judge_once), 심판 라운드 J4(계획 12-1~12-3)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as M

import pytest

from app.judge import aggregate, core, criteria, safety
from app.judge.consistency import ConsistencyResult, Contradiction
from app.judge.coverage import CoverageResult, ItemResult
from app.judge.faithfulness import ClaimVerdict, FaithResult
from app.judge.reading import Piece
from app.narrative.baselines import build_benchmark_refs
from app.narrative.generator import generate_sections
from app.narrative.schemas import NarrativeRequest
from tests import judge_fakes as F

DEV = Path(__file__).parent / "data" / "judge_dev"


def faith(*verdicts):
    return FaithResult(list(verdicts), {})


def fv(verdict, slot="7.1", piece="slot:7.1", quote="q"):
    return ClaimVerdict("C", "number", verdict, slot, piece, quote)


def cons(*cs):
    return ConsistencyResult(list(cs), {})


def ct(slots, sources, pieces=("a", "b"), quotes=("q", "q"), t="value_mismatch"):
    return Contradiction(t, list(pieces), list(slots), list(sources), list(quotes))


# ── 게이트 진리표 ──


@pytest.mark.parametrize("verdicts, contradictions, want", [
    ([fv("match")], [], "pass"),
    ([fv("violation")], [], "blocked"),
    ([fv("inference_violation")], [], "blocked"),         # 단정형 추론
    ([fv("inference_ok"), fv("unverifiable"), fv("target_unclear")], [], "pass"),
    ([fv("match")], [ct(["7.1", "8.2"], ["ai", "ai"])], "blocked"),
])
def test_gate_truth_table(verdicts, contradictions, want):
    assert aggregate.gate(faith(*verdicts), cons(*contradictions)) == want


@pytest.mark.parametrize("contradiction, slots, unfixable", [
    (ct(["7.1", "8.2"], ["ai", "template"]), ["8.2"], 0),      # 둘 다 서술 칸 → 칸 이름 순서 뒤쪽
    (ct(["9", "7.2"], ["ai", "ai"]), ["9"], 0),
    (ct(["8.2", "8"], ["ai", "table"]), ["8.2"], 0),           # 한쪽이 표 → 서술 칸
    (ct(["7.1", "7.1"], ["fixed", "table"]), [], 1),           # 둘 다 표·고정 → 다시 만들 칸 없음
])
def test_flagged_slot_rules(contradiction, slots, unfixable):
    got, uf = aggregate.flagged_slots(faith(fv("violation", slot="7.3")), cons(contradiction))
    assert got == sorted({"7.3", *slots}, key=aggregate.SLOT_ORDER.index) and len(uf) == unfixable


def test_safety_rejects_quotes_not_in_text_and_table_evidence():
    pieces = [Piece("slot:7.1", "7.1", "5.2", "ai", "FN 은 43건이다."), Piece("t", "7.1", "4.4", "table", "FN | 43")]
    ok_f, bad_f = faith(fv("violation", quote="FN 은 43건")), faith(fv("violation", quote="FN 은 44건"))
    cov_bad = CoverageResult([ItemResult("분석-2", True, "t", "7.1", "4.4", "FN | 43")], [], {})
    assert safety.problems(pieces, ok_f, cons(), CoverageResult([], [], {})) == []
    assert safety.problems(pieces, bad_f, cons(), CoverageResult([], [], {})) == ["faithfulness:C"]
    assert safety.problems(pieces, ok_f, cons(), cov_bad) == ["coverage:분석-2"]
    c = ct(["7.1", "7.1"], ["ai", "table"], pieces=("slot:7.1", "t"), quotes=("43건", "99"))
    assert safety.problems(pieces, ok_f, cons(c), CoverageResult([], [], {})) == ["consistency:value_mismatch"]


# ── judge_once 배선(가짜 OpenAI — 출력 형식 이름으로 모듈을 가름) ──


def router(answers: dict):
    """text.format.name(claims·faithfulness·consistency·coverage·readability)로 답을 고르는 가짜 답."""
    def reply(kwargs):
        name = kwargs["text"]["format"]["name"]
        data = answers[name](kwargs) if callable(answers[name]) else answers[name]
        return F.json_response(data)
    return reply


async def _setup(name="dev_multiclass.request.json"):
    req = NarrativeRequest.model_validate(json.loads((DEV / name).read_text(encoding="utf-8")))
    gen = await generate_sections(None, req)
    refs = build_benchmark_refs(req.task_type.value, req.fact_sheet.metrics, edition=4)
    return req, gen, refs


def _empty_answers(kwargs_for_schema):
    """어느 모듈이든 '없음'으로 답한다(스키마 enum 을 읽어 항목·칸을 채움)."""
    def coverage(kw):
        ids = kw["text"]["format"]["schema"]["properties"]["items"]["items"]["properties"]["id"]["enum"]
        return {"items": [{"id": i, "present": False, "quote": "", "piece_id": ""} for i in ids]}

    def readability(kw):
        slots = kw["text"]["format"]["schema"]["properties"]["slots"]["items"]["properties"]["slot"]["enum"]
        return {"slots": [{"slot": s, "present": False, "quote": "", "piece_id": ""} for s in slots]}

    return {
        "claims": {"claims": []},
        "coverage": coverage,
        "readability": readability,
        "consistency": {"checks": [{"type": t, "present": False, "quote_a": "", "piece_a": "", "quote_b": "", "piece_b": ""}
                                   for t in ("verdict_vs_recommendation", "weakness_vs_recommendation", "analysis_vs_finding")]},
        "faithfulness": {"claims": []},
    }


async def test_judge_once_no_claims_safety_holds_with_numbers():
    req, gen, refs = await _setup()
    fake = F.FakeOpenAI([router(_empty_answers(None))] * 20)
    run = await core.judge_once(fake, req, gen.response, gen.section_sources, refs)
    assert (run.gate, run.failure) == (None, "no_claims")  # 숫자가 있는데 주장 0 → 보류


async def test_judge_once_passes_with_one_matching_claim_and_records(monkeypatch):
    # 기록의 기준 버전은 저장소 VERSION(잠그기 전 1.0-dev, 뒤 1.0)이 아니라 criteria.version() 이 돌려준 값이어야 한다 —
    # 어느 상태에서도 같은 것을 보게 눈에 띄는 가짜 버전을 꽂는다(코드에 박은 버전 글자면 빨개진다).
    monkeypatch.setattr(criteria, "version", lambda: "9.9-시험")
    req, gen, refs = await _setup()
    text = gen.response.conclusion.narrative
    answers = _empty_answers(None)
    quote = next(tok for tok in text.split() if any(ch.isdigit() for ch in tok))
    answers["claims"] = {"claims": [{"kind": "inference", "quote": quote, "piece_id": "slot:8.2", "field": "", "field_b": "",
                                     "number_text": "", "polarity": ""}]}
    # 추론 표시 없는 문장이면 위반이 되므로 판정이 결정적이게 — 추론 대신 칸 없는 숫자로 바꿔 에이전트가 no_data 로 답하게 한다
    answers["claims"]["claims"][0]["kind"] = "number"
    answers["faithfulness"] = lambda kw: {"claims": [{"claim_id": "C1", "resolution": "no_data", "expr": "", "expr_b": "",
                                                      "relation": "", "name_field": "", "named": "", "metric_id": "",
                                                      "inference_basis": ""}]}
    fake = F.FakeOpenAI([router(answers)] * 20)
    run = await core.judge_once(fake, req, gen.response, gen.section_sources, refs)
    assert (run.gate, run.failure, run.flagged_slots) == ("pass", None, [])
    card = run.scorecard
    assert list(card) == ["faithfulness", "consistency", "coverage", "readability"]
    assert card["faithfulness"]["unverifiable"] == 1 and card["consistency"]["contradictions"] == 0
    assert card["coverage"]["applicable"] + card["coverage"]["not_applicable"] == 18
    rec = run.record
    assert rec["criteria_version"] == "9.9-시험" and rec["criteria_fingerprint"] == criteria.fingerprint()
    assert len(rec["criteria_fingerprint"]) == 64
    assert {"extract", "coverage", "readability", "faithfulness", "consistency", "pieces", "seconds"} <= set(rec)
    assert {"extract", "coverage", "readability", "faithfulness", "consistency", "aggregate", "total"} <= set(rec["seconds"])
    assert run.cost_usd() is not None
    # 모듈이 서로의 결과를 보지 않는다 — 필수 내용·표기 호출에는 주장이 가지 않는다
    for call in fake.calls:
        if call["text"]["format"]["name"] in ("coverage", "readability", "consistency"):
            assert "C1" not in json.dumps(call["input"], ensure_ascii=False)


async def test_missing_material_is_no_material_without_any_call():
    req, gen, refs = await _setup()
    req = req.model_copy(update={"judge_material": None})
    fake = F.FakeOpenAI([])
    run = await core.judge_once(fake, req, gen.response, gen.section_sources, refs)
    assert (run.gate, run.failure, fake.calls) == (None, "no_material", [])


async def test_empty_lookup_data_is_no_material():
    req, gen, refs = await _setup()
    fs = req.fact_sheet.model_copy(update={"confusion_counts": [], "per_class": []})
    req = req.model_copy(update={"fact_sheet": fs})
    run = await core.judge_once(F.FakeOpenAI([]), req, gen.response, gen.section_sources, refs)
    assert run.failure == "no_material"


async def test_module_failure_means_no_gate():
    req, gen, refs = await _setup()
    answers = _empty_answers(None)
    quote = next(tok for tok in gen.response.conclusion.narrative.split() if any(ch.isdigit() for ch in tok))
    answers["claims"] = {"claims": [{"kind": "number", "quote": quote, "piece_id": "slot:8.2", "field": "", "field_b": "",
                                     "number_text": "", "polarity": ""}]}
    answers["coverage"] = lambda kw: {"items": []}  # 항목 답 없음 → bad_output
    fake = F.FakeOpenAI([router(answers)] * 20)
    run = await core.judge_once(fake, req, gen.response, gen.section_sources, refs)
    assert (run.gate, run.failure, run.scorecard) == (None, "bad_output", None)
    assert not any(c["text"]["format"]["name"] in ("faithfulness", "consistency") for c in fake.calls)  # 뒤 단계로 가지 않음


async def test_core_runs_safety_gate_on_module_output(monkeypatch):
    req, gen, refs = await _setup()
    quote = next(tok for tok in gen.response.conclusion.narrative.split() if any(ch.isdigit() for ch in tok))
    answers = _empty_answers(None)
    answers["claims"] = {"claims": [{"kind": "number", "quote": quote, "piece_id": "slot:8.2", "field": "", "field_b": "",
                                     "number_text": "", "polarity": ""}]}

    async def bad_faith(*_a, **_k):  # 모듈이 잘못돼 글에 없는 인용으로 위반을 냈다
        return FaithResult([ClaimVerdict("C1", "number", "violation", "8.2", "slot:8.2", "글에 없는 인용")], {"checked": 1})

    monkeypatch.setattr(core, "judge_faithfulness", bad_faith)
    run = await core.judge_once(F.FakeOpenAI([router(answers)] * 20), req, gen.response, gen.section_sources, refs)
    assert (run.gate, run.failure, run.record["safety"]) == (None, "safety", ["faithfulness:C1"])
