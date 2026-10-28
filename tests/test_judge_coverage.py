"""tests/test_judge_coverage.py — 필수 내용(app/judge/coverage.py·criteria/rules_conditions.py, 심판 라운드 J4, 계획 11-4)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.judge import coverage as cv
from app.judge import criteria
from app.judge.criteria.rules_conditions import applicable_items, conditions, eval_data_source_provided
from app.judge.reading import Piece
from app.narrative.baselines import build_benchmark_refs
from app.narrative.schemas import NarrativeRequest
from tests import judge_fakes as F

TESTS = Path(__file__).parent
DEV = TESTS / "data" / "judge_dev"
ITEMS = criteria.load_json("checklist.json")["items"]


def _req(name: str) -> NarrativeRequest:
    return NarrativeRequest.model_validate(json.loads((DEV / name).read_text(encoding="utf-8")))


def _ids(req) -> list[str]:
    refs = build_benchmark_refs(req.task_type.value, req.fact_sheet.metrics, edition=4)
    on, off = applicable_items(ITEMS, req.task_type.value, req.fact_sheet, req.client_inputs, refs)
    assert len(on) + len(off) == 18
    return [i["id"] for i in on]


def test_checklist_has_18_items_and_known_conditions():
    data = criteria.load_json("checklist.json")
    assert len(data["items"]) == 18 and len({i["id"] for i in data["items"]}) == 18
    assert {i["condition"] for i in data["items"]} == set(data["conditions"])
    assert data["evidence_rule"]["counts"] == ["ai", "template", "fixed"]


def test_applicable_items_for_dev_inputs():
    # 이진: 위험 방향 fn, 출처 두 줄, 분포 비교(학습 분포 제공), 확률 있음, 지연시간 없음, 목표값이 모두 참조 범위 안(소견-8 없음)
    assert _ids(_req("dev_binary.request.json")) == [
        "분석-1", "분석-2", "분석-3", "분석-4", "분석-6", "분석-7", "소견-1", "소견-2", "소견-3", "소견-5", "소견-6", "소견-7",
        "권고-1", "권고-2", "권고-3"]
    # 다중: 평균 행 둘 이상, 학습 분포 없음, 출처 없음
    ids = _ids(_req("dev_multiclass.request.json"))
    assert "분석-5" in ids and "분석-6" not in ids and "분석-7" not in ids and "소견-7" not in ids
    # 멀티: 출처 한 줄만 → 소견-7 없음
    assert "소견-7" not in _ids(_req("dev_multilabel.request.json"))


@pytest.mark.parametrize("change, item, on", [
    (lambda r: setattr(r.fact_sheet, "error_ratios", None), "분석-4", False),
    (lambda r: setattr(r.fact_sheet, "problems", []), "권고-1", False),
    (lambda r: setattr(r.fact_sheet.latency, "available", True), "소견-4", True),
    (lambda r: setattr(r.fact_sheet.facts.declarations, "risk_direction", "similar"), "분석-7", False),
])
def test_value_presence_switches_items(change, item, on):
    req = copy.deepcopy(_req("dev_binary.request.json"))
    change(req)
    assert (item in _ids(req)) is on


def test_eval_data_source_rule_matches_shared_table():
    table = json.loads((TESTS / "contract" / "reporting_item2_cases.json").read_text(encoding="utf-8"))["cases"]
    assert len(table) > 10
    for case in table:
        assert eval_data_source_provided({"test_data": case["test_data"]}) is case["provided"]["backend"], case["name"]


def test_no_reference_means_no_reference_items():
    req = _req("dev_binary.request.json")
    cond = conditions("binary", req.fact_sheet, req.client_inputs, [])
    assert not cond["reference_present"] and not cond["target_outside_reference"]


# ── LLM 판정(가짜) ──

PIECES = [
    Piece("slot:7.1", "7.1", "5.2", "ai", "가장 약한 클래스는 0 이고 F1 은 0.497 이다."),
    Piece("m:5.1-fixed11", "8.2", "5.1", "fixed", "평가 데이터는 학습 데이터와 다른 출처에서 수집되었다."),
    Piece("m:4.4-counts", "7.1", "4.4", "table", "클래스 | FN\n1 | 43"),
]


def ans(i, present=False, quote="", piece=""):
    return {"id": i, "present": present, "quote": quote, "piece_id": piece}


async def _run(*replies):
    req = _req("dev_binary.request.json")
    fake = F.FakeOpenAI(list(replies))
    res = await cv.judge_coverage(fake, PIECES, "binary", req.fact_sheet, req.client_inputs, [])
    return fake, res


def _all(**described):
    ids = _ids_no_ref()
    return F.json_response({"items": [described.get(i, ans(i)) for i in ids]})


def _ids_no_ref():
    req = _req("dev_binary.request.json")
    on, _ = applicable_items(ITEMS, "binary", req.fact_sheet, req.client_inputs, [])
    return [i["id"] for i in on]


async def test_described_items_record_real_print_place_and_tables_do_not_count():
    reply = _all(**{"분석-1": ans("분석-1", True, "가장 약한 클래스는 0 이고 F1 은 0.497 이다", "slot:7.1"),
                    "분석-2": ans("분석-2", True, "1 | 43", "m:4.4-counts"),
                    "소견-7": ans("소견-7", True, "다른 출처에서 수집되었다", "m:5.1-fixed11")})
    fake, res = await _run(reply)
    by = {i.id: i for i in res.items}
    assert (by["분석-1"].described, by["분석-1"].place) == (True, "5.2")
    assert by["분석-2"].described is False  # 표에만 있음 — 서술 아님(Q5)
    assert (by["소견-7"].described, by["소견-7"].place) == (True, "5.1")  # 고정 문구는 인정
    assert res.counts == {"applicable": len(_ids_no_ref()), "described": 2, "not_applicable": 18 - len(_ids_no_ref())}
    assert "## 확인할 항목" in fake.calls[0]["instructions"] and "분석-7" in fake.calls[0]["instructions"]


async def test_uncited_present_is_asked_again_then_failure():
    bad = _all(**{"분석-1": ans("분석-1", True, "없는 구절", "slot:7.1")})
    again = F.json_response({"items": [ans("분석-1", True, "", "")]})
    fake, res = await _run(bad, again)
    assert len(fake.calls) == 2 and res.failure == "uncited"
    assert "분석-1" in fake.calls[1]["instructions"] and "분석-2:" not in fake.calls[1]["instructions"]


async def test_missing_item_answer_is_bad_output():
    _, res = await _run(F.json_response({"items": [ans("분석-1")]}))
    assert res.failure == "bad_output"


async def test_reads_everything_including_fixed_and_tables():
    fake, _ = await _run(_all())
    sent = json.dumps(fake.calls[0]["input"], ensure_ascii=False)
    assert all(p.id in sent for p in PIECES)
