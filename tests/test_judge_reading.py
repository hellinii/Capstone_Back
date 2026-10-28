"""tests/test_judge_reading.py — 심판이 읽는 범위의 조각(app/judge/reading.py, 심판 라운드 J2 ③, 계획 10-2)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.judge import reading
from app.judge.material import parse_material
from app.narrative.baselines import build_benchmark_refs
from app.narrative.generator import generate_sections
from app.narrative.schemas import NarrativeRequest

TESTS = Path(__file__).parent
CONTRACT = TESTS / "contract"
DEV = TESTS / "data" / "judge_dev"
REF_CASES = "ref_table_ed4_cases.json"
FRONT_REF_CASES = TESTS.parents[1] / "Capstone_Front" / "src" / "test" / "contract" / REF_CASES
# 심판 라운드 J2 ③(2026-10-29)에 프론트 refTableContract.test.ts 가 뜬 바이트.
REF_CASES_SHA256 = "85aa0cf782f19cd95abc4a6aeb2b12f1af54f729f090a8434b6765348e4e66a8"
EXAMPLE_OF = {"bc_strong": "binary", "digits_strong": "multiclass", "yeast_strong": "multilabel"}


def _req(path: Path) -> NarrativeRequest:
    return NarrativeRequest.model_validate(json.loads(path.read_text(encoding="utf-8")))


# ── 4.6 참조 범위 표 = 화면 줄 ──


def test_ref_cases_pinned():
    assert hashlib.sha256((CONTRACT / REF_CASES).read_bytes()).hexdigest() == REF_CASES_SHA256


@pytest.mark.skipif(not FRONT_REF_CASES.exists(), reason="옆 저장소(Capstone_Front)가 없다 — CI 는 저장소 하나만 받는다")
def test_ref_cases_front_copy_identical():
    assert FRONT_REF_CASES.read_bytes() == (CONTRACT / REF_CASES).read_bytes()


@pytest.mark.parametrize("case", json.loads((CONTRACT / REF_CASES).read_text(encoding="utf-8"))["cases"], ids=lambda c: c["tag"])
def test_ref_table_rows_equal_screen_rows(case):
    req = _req(CONTRACT / f"fact_sheet_{EXAMPLE_OF[case['tag']]}_ed4.json")
    assert case["rows"], "대조할 줄이 없다"
    assert reading.ref_table_rows(case["refs"], req.fact_sheet.metrics) == case["rows"]
    piece = reading.ref_table_piece(case["refs"], req.fact_sheet.metrics)
    assert all(note in piece.text for note in case["notes"]) and case["notes"]


def test_ed4_places_match_slot_contract():
    slots = json.loads((CONTRACT / "narrative_slots_ed4.json").read_text(encoding="utf-8"))["slots"]
    contract = {s["slot"]: s["edition4_place"] for s in slots if s["slot"] in reading.ED4_PLACE}
    assert contract == reading.ED4_PLACE


# ── 조각 만들기(키 없는 규칙 문장 서술로) ──


async def _pieces(name: str):
    req = _req(DEV / name)
    gen = await generate_sections(None, req)
    refs = build_benchmark_refs(req.task_type.value, req.fact_sheet.metrics, edition=4)
    material, problem = parse_material(req.judge_material)
    assert problem is None
    return req, gen, reading.build_pieces(gen.response, gen.section_sources, material, refs, req.fact_sheet.metrics)


@pytest.mark.parametrize("name", ["dev_binary.request.json", "dev_multiclass.request.json", "dev_multilabel.request.json"])
async def test_pieces_from_rule_sentence_narrative(name):
    req, gen, pieces = await _pieces(name)
    assert pieces and len({p.id for p in pieces}) == len(pieces)
    sources = {p.source for p in pieces}
    assert {"template", "fixed", "table"} <= sources and "ai" not in sources  # 키 없음 → 규칙 문장
    assert [p.slot for p in pieces] == sorted((p.slot for p in pieces), key=reading.SLOT_ORDER.index)
    # 칸 글은 서술 결과 그대로
    by_id = {p.id: p for p in pieces}
    assert by_id["slot:8.2"].text == gen.response.conclusion.narrative and by_id["slot:8.2"].place == "5.1"
    # 권고 줄 = 서술 결과의 권고 수, 대응 문제 칸은 재료에서
    recs = [p for p in pieces if p.id.startswith("rec:") and not p.id.endswith(":row")]
    rows = [p for p in pieces if p.id.endswith(":row")]
    assert len(recs) == len(rows) == len(gen.response.recommendations) > 0
    assert all(p.source == "template" and p.slot == "9" and p.place in ("6.1", "6.2") for p in recs)
    assert all(p.source == "fixed" for p in rows)  # 우선순위·대응 문제 칸은 프론트 글
    for row, rec in zip(rows, recs):
        pid = rec.id.split(":")[2]
        assert row.id == rec.id + ":row" and f"{pid} · " in row.text and f"{pid} · " not in rec.text
        assert rec.text.startswith("권고: ")
    # 7.3 이 빈 글이면 고정 15 가 들어온다
    assert gen.response.interpretation.additional_observations == ""
    assert by_id["m:5.6-fixed15"].text == "해당 없음"


async def test_slot_empty_segments_follow_the_slot_text():
    req, gen, _ = await _pieces("dev_binary.request.json")
    material, _ = parse_material(req.judge_material)
    resp = gen.response.model_copy(deep=True)
    resp.interpretation.additional_observations = "추가 관찰 글"
    pieces = reading.build_pieces(resp, gen.section_sources, material, [], req.fact_sheet.metrics)
    ids = {p.id for p in pieces}
    assert "m:5.6-fixed15" not in ids and "slot:7.3" in ids
    # 8.1 B18 은 8.1 이 비고 출처가 fixed 일 때만
    sources = {**gen.section_sources, "8.1": "fallback"}
    resp.conclusion.benchmark = ""
    assert "m:5.4-b18" not in {p.id for p in reading.build_pieces(resp, sources, material, [], req.fact_sheet.metrics)}
    sources["8.1"] = "fixed"
    assert "m:5.4-b18" in {p.id for p in reading.build_pieces(resp, sources, material, [], req.fact_sheet.metrics)}


async def test_module_ranges():
    _, _, pieces = await _pieces("dev_multiclass.request.json")
    for module in ("extract", "faithfulness", "readability"):
        got = reading.for_module(pieces, module)
        assert got and {p.source for p in got} <= {"ai", "template"}
    for module in ("coverage", "consistency"):
        assert reading.for_module(pieces, module) == pieces
    with pytest.raises(KeyError):
        reading.for_module(pieces, "unknown")


async def test_without_material_only_narrative_and_ref_table():
    req, gen, _ = await _pieces("dev_binary.request.json")
    refs = build_benchmark_refs(req.task_type.value, req.fact_sheet.metrics, edition=4)
    pieces = reading.build_pieces(gen.response, gen.section_sources, None, refs, req.fact_sheet.metrics)
    assert {p.source for p in pieces} <= {"template", "table"}
    assert not any(p.id.startswith(("m:", "rec:")) for p in pieces)


def test_render_heads_each_piece():
    p = reading.Piece("slot:7.1", "7.1", "5.2", "ai", "글")
    assert reading.render([p, p]) == "[slot:7.1 | 칸 7.1 | 5.2 | ai]\n글\n\n[slot:7.1 | 칸 7.1 | 5.2 | ai]\n글"
