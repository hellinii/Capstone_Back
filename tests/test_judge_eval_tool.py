"""판정자 평가 실험 실행기(scripts/judge_eval.py, 사전 등록 v1) 시험.

실호출은 하지 않는다. dev 모드(개발 입력 3건의 규칙 문장 서술 + 가짜 OpenAI)와 임시 평가 폴더만 쓴다. 평가 폴더(JUDGE_ROUND/평가)와
검증 폴더는 열지 않는다(dev 경계 시험이 거부를 본다).
"""
from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 가 모듈을 찾는다
    spec.loader.exec_module(mod)
    return mod


jv = _load("judge_validate")
je = _load("judge_eval")

from app.judge.validation import evaluation as E  # noqa: E402
from app.judge.validation import variants as V  # noqa: E402

_BASES: dict = {}


def bases():
    if not _BASES:
        _BASES.update(asyncio.run(jv.dev_bases(jv.DEV_INPUTS)))
    return _BASES


@pytest.fixture
def vdir(tmp_path, monkeypatch):
    """가짜 검증 폴더 — dev 는 이 안을 건드리면 안 된다. 상한 환경변수는 지운다."""
    d = tmp_path / "검증"
    d.mkdir()
    monkeypatch.setenv("JUDGE_VALIDATION_DIR", str(d))
    monkeypatch.delenv("JUDGE_LIVE_BUDGET_USD", raising=False)
    monkeypatch.delenv("JUDGE_LIVE_MAX_RUNS", raising=False)
    return d


@pytest.fixture
def ctx(tmp_path, vdir):
    lines: list[str] = []
    c = je.make_ctx("dev", tmp_path / "평가실험", lines.append)
    c.lines = lines
    return c


def _sentence(tag: str, slot: str, k: int) -> str:
    return V.sentences(V.read_slot(bases()[tag].result, slot))[k]


def _row(rid: str, base: str, slot: str, old: str, new: str, **kw) -> dict:
    plan = E.PREREG_DEFECTS.get(rid)
    row = {"id": rid, "base": base, "axis": plan.axis if plan else "변형", "type": plan.type if plan else "연결 어미 교체",
           "slot": slot, "old": old, "new": new, "grounds": {"field": "", "value": "", "note": "합성"},
           "expected_axes": list(plan.expected_axes) if plan else [],
           "allowed_secondary_axes": list(plan.allowed_secondary_axes) if plan else [],
           "first_check_pass_by_design": True if plan else None, "author": "시험", "note": ""}
    row.update(kw)
    return row


def table_rows() -> list[dict]:
    """개발 입력 위의 합성 결함·변형 — 숫자 결함 F01, 판정↔권고 추가 C02, 단정형 추론 F08, 연결 어미 V01, 문장 순서 V02."""
    s71 = _sentence("dev_binary", "interpretation.confusion_analysis", 1)
    risks = V.read_slot(bases()["dev_binary"].result, "conclusion.risks")
    r = V.sentences(risks)
    swap_old = risks[risks.index(r[0]): risks.index(r[1]) + len(r[1])]
    s82 = _sentence("dev_multiclass", "conclusion.narrative", 0)
    s83 = _sentence("dev_multilabel", "conclusion.risks", 0)
    return [
        _row("F01", "dev_binary", "interpretation.confusion_analysis", s71, s71.replace("43건", "56건"),
             grounds={"field": "confusion_counts.1.fn", "value": 43, "note": "FN 자리에 FP 건수"}),
        _row("C02", "dev_multiclass", "conclusion.narrative", s82, s82 + " 현 상태로 운영해도 무방할 것으로 보인다."),
        _row("F08", "dev_multilabel", "conclusion.risks", s83, s83[:-1] + " 때문이다."),
        _row("V01", "dev_binary", "interpretation.distribution_analysis", "200건이며,", "200건이고,"),
        _row("V02", "dev_binary", "conclusion.risks", swap_old, f"{r[1]} {r[0]}", type="문장 순서 교체"),
    ]


def write_table(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "표.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


# ── plant ──


def test_plant_writes_folders_with_injected_sentences(ctx, tmp_path):
    rows = table_rows()
    assert je.cmd_plant(ctx, write_table(tmp_path, rows)) == ["F01", "C02", "F08", "V01", "V02"]
    folder = ctx.root / "02_결함"
    items = je.load_items(ctx)
    assert sorted(items) == ["C02", "F01", "F08", "V01", "V02"]
    meta, slots = items["F01"]
    assert meta["kind"] == "defect" and meta["expected_axes"] == ["faithfulness"] and meta["allowed_secondary_axes"] == ["consistency"]
    assert meta["injected"] == [[1, rows[0]["new"]]] and list(slots) == ["interpretation.confusion_analysis"]
    diff = (folder / "F01" / "diff.txt").read_text(encoding="utf-8").splitlines()
    assert [ln for ln in diff if ln.startswith("+") and not ln.startswith("+++")] == ["+" + rows[0]["new"]]
    assert items["C02"][0]["injected"] == [[1, "현 상태로 운영해도 무방할 것으로 보인다."]]  # 추가 문장
    assert [j for j, _ in items["V02"][0]["injected"]] == [0, 1]  # 순서 교체는 두 문장 모두
    # 주입 문장이 문서 문장 번호에 붙는다(원본 문장 번호 체계 그대로)
    docs = {d.id: d for d in je.documents(bases(), items)}
    base = bases()["dev_binary"]
    idx = je.doc_index(base, docs["F01"].slots)
    ids = je.injected_of(docs["F01"], base, idx)
    assert len(ids) == 1 and idx.get(ids[0]).text == rows[0]["new"]


def test_plant_refuses_existing_and_bad_rows_and_writes_nothing(ctx, tmp_path):
    rows = table_rows()
    je.cmd_plant(ctx, write_table(tmp_path, rows[:1]))
    s82 = _sentence("dev_multiclass", "conclusion.narrative", 0)
    bad = [
        rows[0],  # 이미 있다
        {**rows[1], "expected_axes": ["consistency", "faithfulness"]},  # 사전 등록 기대 축과 다름
        _row("F02", "dev_binary", "interpretation.confusion_analysis", "43건", "44건"),  # old 가 두 번 나온다
        _row("C04", "dev_multiclass", "conclusion.narrative", s82, s82.replace(s82[-2:], " 2건이다.")),  # 숫자를 바꾼 일관성 결함
        _row("F99", "dev_binary", "conclusion.risks", "a", "b"),  # 없는 ID
        _row("V03", "dev_binary", "interpretation.additional_observations", "a", "b"),  # 서술이 아닌 칸(7.3 고정 문구)
        _row("F03", "dev_binary", "conclusion.risks", "없는 글", "x"),
    ]
    with pytest.raises(je.RunnerError) as err:
        je.cmd_plant(ctx, write_table(tmp_path, bad))
    text = str(err.value)
    for part in ("F01: 02_결함/F01 가 이미 있다", "C02: 기대 축이 사전 등록과 다르다", "F02: old 가 칸 글에 정확히 한 번",
                 "C04: 일관성 결함(C01 제외)의 변경문은 숫자를 바꾸지 않는다", "F99: ID 는", "V03: 서술(AI·규칙 문장) 칸이 아니다",
                 "F03: old 가 칸 글에 정확히 한 번 나와야 한다(지금 0번)"):
        assert part in text, part
    assert sorted(p.name for p in (ctx.root / "02_결함").iterdir()) == ["F01"]  # 한 줄이라도 틀리면 아무것도 안 쓴다
    with pytest.raises(je.RunnerError, match="같은 ID 가 두 번"):
        je.cmd_plant(ctx, write_table(tmp_path, [rows[1], rows[1]]))


def test_plant_verify_requires_note_when_moved_report(tmp_path, monkeypatch):
    """verify 에서 사전 등록 결과서와 다른 결과서에 심으면 note(까닭)가 있어야 하고 moved_from 이 남는다(§3.5)."""
    row = table_rows()[0]
    bad, _ = je.check_plant_row({**row, "note": ""}, bases(), mode="verify", existing=set())
    assert any("§3.5" in p for p in bad)
    ok, plan = je.check_plant_row({**row, "note": "pima 에 넣을 문장이 없음"}, bases(), mode="verify", existing=set())
    assert ok == [] and plan["meta"]["moved_from"] == "pima_strong"


def test_dev_refuses_eval_and_validation_folders(tmp_path, vdir):
    with pytest.raises(je.RunnerError, match="평가 폴더"):
        je.make_ctx("dev", je.DEFAULT_EVAL / "x")
    with pytest.raises(je.RunnerError, match="검증 폴더"):
        je.make_ctx("dev", vdir / "x")
    assert je.eval_root(None) == vdir.resolve()  # verify 기본은 JUDGE_VALIDATION_DIR
    assert je.eval_root(tmp_path / "r") == (tmp_path / "r").resolve()


def test_verify_default_root_is_next_to_repo(monkeypatch):
    monkeypatch.delenv("JUDGE_VALIDATION_DIR", raising=False)
    assert je.eval_root(None) == (ROOT.parent / "JUDGE_ROUND" / "평가").resolve()


# ── precheck ──


def test_precheck_variants_same_defects_readability_only(ctx, tmp_path):
    je.cmd_plant(ctx, write_table(tmp_path, table_rows()))
    out = je.cmd_precheck(ctx)
    assert out["variants_differ"] == []
    assert out["items"]["V01"]["first_check_same"] is True and out["items"]["V02"]["readability_same"] is True
    assert "first_check_same" not in out["items"]["F01"] and out["items"]["F01"]["first_check"].startswith("돌리지 않음")
    saved = json.loads((ctx.root / "01_구성" / "precheck.json").read_text(encoding="utf-8"))
    assert saved["items"].keys() == out["items"].keys() and "02_결함/F01/meta.json" in saved["files"]


def test_precheck_flags_variant_that_changes_first_check(ctx, tmp_path):
    rows = [_row("V03", "dev_binary", "interpretation.distribution_analysis", "총 200건이며,", "총 201건이며,",
                 type="숫자 표기 변환")]
    je.cmd_plant(ctx, write_table(tmp_path, rows))
    out = je.cmd_precheck(ctx)
    assert out["variants_differ"] == ["V03"]
    diff = out["items"]["V03"]["first_check_diff"]
    assert list(diff) == ["7.2"] and diff["7.2"]["this"]["numbers"] == ["201"]
    assert je.main(["precheck", "--mode", "dev", "--root", str(ctx.root)]) == 1
