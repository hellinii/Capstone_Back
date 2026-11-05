"""판정자 평가 실험 실행기(scripts/judge_eval.py, 사전 등록 v1 + v3 2절) 시험.

실호출은 하지 않는다. dev 모드(개발 입력 3건의 규칙 문장 서술 + 가짜 OpenAI)와 임시 평가 폴더만 쓴다. 평가 폴더(JUDGE_ROUND/평가)와
검증 폴더는 열지 않는다(dev 경계 시험이 거부를 본다). .env 도 읽지 않는다(_no_dotenv).
뒤쪽 '검토 고침' 절은 두 검토(P0·P1, 변이 생존)를 고친 시험이다.
"""
from __future__ import annotations

import asyncio
import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
import textwrap
import time
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


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """verify 명령의 load_env 가 저장소 .env 를 읽어 환경을 바꾸지 않게 — 시험은 .env 없이."""
    monkeypatch.setattr(je, "load_env", lambda: None, raising=False)


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


# ── run: 장부·실행 번호·동시 실행·이어 하기·상한 ──


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def planted(ctx, tmp_path):
    je.cmd_plant(ctx, write_table(tmp_path, table_rows()))
    return ctx


class FakeJudge:
    """judge_validate.judge_one 자리 — 가짜 OpenAI 없이 JudgeRun 을 돌려준다. 동시에 몇 개가 돌았는지 센다.
    plan[문서 서술 해시] 로 문서마다 (게이트, 실패, 비용, 표시) 를 정한다."""

    def __init__(self, *, delay=0.0, gate="pass", failure=None, cost=0.01, raise_once: set | None = None, per_doc=None):
        self.delay, self.gate, self.failure, self.cost = delay, gate, failure, cost
        # 서술 해시 → 한 번 던질 예외(집합이면 끊김 ConnectionError)
        self.raise_once = dict(raise_once) if isinstance(raise_once, dict) else {k: ConnectionError("끊김(시험)")
                                                                                for k in raise_once or ()}
        self.per_doc = per_doc or {}
        self.now = self.peak = self.calls = 0
        self.seen: dict[str, int] = {}

    async def __call__(self, client, base, slots):
        from app.judge.core import JudgeRun

        self.calls += 1
        self.now += 1
        self.peak = max(self.peak, self.now)
        try:
            await asyncio.sleep(self.delay() if callable(self.delay) else self.delay)
            doc = next(iter(slots)) if slots else base.tag
            sha = je.narrative_sha(base, slots)
            if sha in self.raise_once:
                raise self.raise_once.pop(sha)
            spec = self.per_doc.get(sha, {})
            if isinstance(spec, list):  # 문서마다 부른 차례대로(1회차·2회차)
                seen = self.seen.get(sha, 0)
                self.seen[sha] = seen + 1
                spec = spec[min(seen, len(spec) - 1)]
            gate, failure = spec.get("gate", self.gate), spec.get("failure", self.failure)
            record = {"extract": {"claims": spec.get("claims", []), "calls": [{"cost_usd": spec.get("cost", self.cost),
                                                                                 "response_model": "gpt-6-luna"}]},
                      "faithfulness": {"verdicts": spec.get("verdicts", [])},
                      "consistency": {"contradictions": spec.get("contradictions", [])}, "seconds": {"total": 0.01}}
            card = None if failure else spec.get("card", {
                "faithfulness": {"checked": 10, "matched": 10, "inference_ok": 2, "inference_violations": 1},
                "coverage": {"applicable": 5, "described": 5},
                "consistency": {"contradictions": len(spec.get("contradictions", []))}, "readability": {"violations": 0}})
            return JudgeRun(None if failure else gate, failure, scorecard=card, record=record), sha
        finally:
            self.now -= 1


def _sha_of(doc_id: str) -> str:
    items = {}
    for row in table_rows():
        b = bases()[row["base"]]
        text = V.read_slot(b.result, row["slot"]).replace(row["old"], row["new"], 1)
        items[row["id"]] = je.narrative_sha(b, {row["slot"]: text})
    for tag, b in bases().items():
        items[tag] = je.narrative_sha(b, None)
    return items[doc_id]


def test_run_writes_one_row_and_file_per_doc_then_resume_skips(planted):
    ctx = planted
    judge = FakeJudge(delay=0.01)
    out = je.cmd_run(ctx, 1, concurrency=3, yes=True, judge=judge, client=object())
    rows = _rows(ctx.ledger_path)
    assert out["ran"] == 8 and len(rows) == 8 and judge.calls == 8
    assert sorted(r["run_id"] for r in rows) == [f"E-{i:04d}" for i in range(1, 9)]
    assert {r["stage"] for r in rows} == {"eval_r1"} and {r["repeat"] for r in rows} == {"1"} and {r["mode"] for r in rows} == {"dev"}
    assert sorted(E.doc_id(r["report"], r["variant"]) for r in rows) == sorted(["dev_binary", "dev_multiclass", "dev_multilabel",
                                                                               "F01", "C02", "F08", "V01", "V02"])
    assert tuple(rows[0]) == je.jl.LEDGER_COLUMNS  # 24열 그대로
    for r in rows:
        path = ctx.root / r["result_file"]
        assert r["result_file"].startswith("03_실행/") and je.sha256_file(path) == r["result_sha256"]
        body = json.loads(path.read_text(encoding="utf-8"))
        assert body["run_id"] == r["run_id"] and body["key"]["round"] == 1 and body["attempt"] == 1
        assert body["manifest_sha256"] == je.criteria.fingerprint() and body["gate"] == "pass"
        assert body["order_seed"] == 20261105 and body["narrative_sha256"] == _sha_of(body["key"]["doc"])
    plan = json.loads((ctx.root / "03_실행" / "plan-r1.json").read_text(encoding="utf-8"))
    import random as _r

    expect = sorted(je.documents(bases(), je.load_items(ctx)), key=lambda d: (d.report, d.variant))
    _r.Random(20261105).shuffle(expect)
    assert plan["seed"] == 20261105 and [o[2] for o in plan["order"]] == [d.id for d in expect]
    # 이어 하기 — 끝난 회차는 건너뛴다
    again = je.cmd_run(ctx, 1, concurrency=3, yes=True, judge=FakeJudge(), client=object())
    assert again["ran"] == 0 and len(_rows(ctx.ledger_path)) == 8


def test_concurrent_runs_get_unique_ids_and_respect_concurrency(planted):
    ctx = planted
    import random as _r

    rng = _r.Random(3)
    judge = FakeJudge(delay=lambda: rng.uniform(0.0, 0.03))
    je.cmd_run(ctx, 1, concurrency=3, yes=True, judge=judge, client=object())
    rows = _rows(ctx.ledger_path)
    ids = [r["run_id"] for r in rows]
    assert len(ids) == len(set(ids)) == 8 and 2 <= judge.peak <= 3  # 실제로 동시에 돌았고 3개를 넘지 않았다
    files = sorted(p.name for p in (ctx.root / "03_실행").glob("E-*.json"))
    assert files == sorted(f"{i}.json" for i in ids)


def test_ledger_writer_ids_unique_under_many_concurrent_commits(tmp_path):
    writer_path = tmp_path / "장부.csv"

    async def main():
        writer = je.LedgerWriter(writer_path)

        async def commit(k):
            await asyncio.sleep(0.001 * (k % 5))

            def build(run_id, rows):
                return {"stage": "eval_r1", "report": "x", "variant": f"v{k}", "repeat": 1}, [(tmp_path / f"{run_id}.json", "{}")]
            return await writer.commit(build)

        return await asyncio.gather(*(commit(k) for k in range(40)))

    rows = asyncio.run(main())
    assert sorted(r["run_id"] for r in rows) == [f"E-{i:04d}" for i in range(1, 41)]
    assert len(_rows(writer_path)) == 40 and len(list(tmp_path.glob("E-*.json"))) == 40


def test_round2_waits_for_round1_and_uses_its_seed(planted):
    ctx = planted
    je.cmd_run(ctx, 1, yes=True, judge=FakeJudge(), client=object(), only=["F01", "dev_binary"])
    with pytest.raises(je.RunnerError, match="1회차가 끝나지 않았다"):
        je.cmd_run(ctx, 2, yes=True, judge=FakeJudge(), client=object())
    je.cmd_run(ctx, 1, yes=True, judge=FakeJudge(), client=object())
    out = je.cmd_run(ctx, 2, yes=True, judge=FakeJudge(), client=object())
    rows = _rows(ctx.ledger_path)
    assert out["ran"] == 8 and len(rows) == 16 and sum(r["stage"] == "eval_r2" for r in rows) == 8
    assert json.loads((ctx.root / "03_실행" / "plan-r2.json").read_text(encoding="utf-8"))["seed"] == 20261106
    assert len({r["run_id"] for r in rows}) == 16


def test_api_failures_are_retried_but_judge_failures_are_final(planted):
    ctx = planted
    boom = FakeJudge(raise_once={_sha_of("F01")}, per_doc={_sha_of("C02"): {"failure": "uncited"},
                                                          _sha_of("V01"): {"failure": "timeout"}})
    out = je.cmd_run(ctx, 1, yes=True, judge=boom, client=object())
    assert out["gates"] == {"pass": 5, "재시도 대상": 2, "최종 실패": 1}
    rows = _rows(ctx.ledger_path)
    err = {E.doc_id(r["report"], r["variant"]): r["error"] for r in rows}
    assert err["F01"].startswith("api_exception:ConnectionError") and err["C02"] == "uncited" and err["V01"] == "timeout"
    # 다시 부르면 재시도 대상(예외·시간 초과)만 — 판정자 실패(uncited)는 결과를 이유로 다시 돌리지 않는다(§7.4)
    again = FakeJudge()
    out = je.cmd_run(ctx, 1, yes=True, judge=again, client=object())
    assert out["ran"] == 2 and again.calls == 2
    rows = _rows(ctx.ledger_path)
    f01 = [r for r in rows if r["variant"] == "F01"]
    assert len(f01) == 2 and f01[-1]["gate"] == "pass"
    assert json.loads((ctx.root / f01[-1]["result_file"]).read_text(encoding="utf-8"))["attempt"] == 2
    states = je.stage_states(rows, "eval_r1", fingerprint=je.criteria.fingerprint(), mode="dev")
    assert states[("dev_multiclass", "C02")]["status"] == "final" and states[("dev_binary", "F01")]["attempts"] == 2


def test_cap_refuses_start_and_stops_launching_mid_run(planted, monkeypatch):
    ctx = planted
    monkeypatch.setenv("JUDGE_LIVE_MAX_RUNS", "5")
    with pytest.raises(je.RunnerError, match="회차 상한을 넘는다: 장부 0회 \\+ 8회 > 5회"):
        je.cmd_run(ctx, 1, yes=True, judge=FakeJudge(), client=object())
    assert not ctx.ledger_path.exists() and not (ctx.root / "03_실행" / "plan-r1.json").exists()
    monkeypatch.setenv("JUDGE_LIVE_MAX_RUNS", "110")
    monkeypatch.setenv("JUDGE_LIVE_BUDGET_USD", "3")  # 예상 8 × $0.30 = $2.40 → 시작은 된다
    out = je.cmd_run(ctx, 1, concurrency=1, yes=True, judge=FakeJudge(cost=2.8), client=object())
    assert out["ran"] == 1 and len(out["stopped"]) == 7  # 첫 회차 실비 $2.80 + 다음 예상 $0.30 > $3 — 새 회차를 띄우지 않는다
    assert any("상한에 닿아 새 회차를 띄우지 않는다" in line for line in ctx.lines)
    assert je.main(["status", "--mode", "dev", "--root", str(ctx.root)]) == 0


def test_run_never_prompts_even_past_ten_dollar_marks(planted, monkeypatch):
    ctx = planted
    monkeypatch.setenv("JUDGE_LIVE_BUDGET_USD", "1000")
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError("확인 입력을 물었다")))
    out = je.cmd_run(ctx, 1, yes=True, judge=FakeJudge(cost=6.0), client=object())
    assert out["ran"] == 8 and je.spent(_rows(ctx.ledger_path))[0] == pytest.approx(48.0)


def test_run_without_yes_or_with_bad_args_touches_nothing(planted):
    ctx = planted
    with pytest.raises(je.RunnerError, match="--yes"):
        je.cmd_run(ctx, 1, judge=FakeJudge(), client=object())
    with pytest.raises(je.RunnerError, match="동시 실행은 1~3개"):
        je.cmd_run(ctx, 1, concurrency=4, yes=True, judge=FakeJudge(), client=object())
    with pytest.raises(je.RunnerError, match="없는 문서"):
        je.cmd_run(ctx, 1, yes=True, only=["F09"], judge=FakeJudge(), client=object())
    assert not ctx.ledger_path.exists() and not (ctx.root / "03_실행").exists()


def test_dev_run_with_fake_openai_scores_end_to_end(planted):
    """가짜 OpenAI(dev 서버의 '문제 없음' 답)로 진짜 judge_once 를 돈다 — 원본 하나만."""
    ctx = planted
    out = je.cmd_run(ctx, 1, yes=True, only=["dev_binary"])
    rows = _rows(ctx.ledger_path)
    assert out["ran"] == 1 and rows[0]["gate"] == "pass" and float(rows[0]["cost_usd"]) > 0


def test_verify_problems_add_prereg_fingerprint_and_models(tmp_path, monkeypatch):
    monkeypatch.setattr(je.criteria, "version", lambda: "1.25")
    monkeypatch.setattr(je.criteria, "version_problem", lambda *a, **k: None)
    manifest = je.criteria.recorded_manifest()

    def git(*args):
        return (0, "") if args[0] == "status" else (0, manifest)

    assert je.eval_verify_problems(live=True, ledger_path=tmp_path / "장부.csv", git=git) == []
    monkeypatch.setenv("JUDGE_MODEL", "gpt-other")
    monkeypatch.setenv("JUDGE_EXTRACT_EFFORT", "high")
    problems = je.eval_verify_problems(live=True, ledger_path=tmp_path / "장부.csv", git=git)
    assert len(problems) == 2 and "판정자 모델" in problems[0] and "추출 추론 강도" in problems[1]
    monkeypatch.delenv("JUDGE_MODEL")
    monkeypatch.delenv("JUDGE_EXTRACT_EFFORT")
    monkeypatch.setattr(je.criteria, "fingerprint", lambda: "0" * 64)
    problems = je.eval_verify_problems(live=False, ledger_path=tmp_path / "장부.csv", git=git)
    assert any("사전 등록 값과 다르다" in p for p in problems) and any("--live" in p for p in problems)


def test_verify_run_refuses_incomplete_set_and_unfrozen_defects(tmp_path, monkeypatch):
    monkeypatch.setattr(je, "eval_verify_problems", lambda **kw: [])
    monkeypatch.setattr(je, "load_bases", lambda ctx: {t: b for t, b in zip(E.EVAL_REPORTS, bases().values())})
    ctx = je.make_ctx("verify", tmp_path / "평가")
    with pytest.raises(je.RunnerError) as err:
        je.cmd_run(ctx, 1, live=True, yes=True, judge=FakeJudge(), client=object())
    assert "없는 문서: C01" in str(err.value) and "precheck.json 이 없다" in str(err.value)
    assert not (tmp_path / "평가" / "장부.csv").exists()


# ── 베이스라인 1·2 ──


def _f02_row() -> dict:
    """허용 목록 밖 숫자로 바꾼 결함 — 1차 점검이 그 문장에서 잡는다."""
    return _row("F02", "dev_binary", "interpretation.distribution_analysis", "불균형 비율(1.105)", "불균형 비율(1.205)",
                first_check_pass_by_design=False)


def test_baseline1_locates_findings_and_counts_defect_detection(ctx, tmp_path):
    je.cmd_plant(ctx, write_table(tmp_path, [*table_rows(), _f02_row()]))
    out = je.cmd_baseline1(ctx)
    docs = out["docs"]
    assert len(docs) == 9 and set(docs) >= {"dev_binary", "F01", "F02", "V01"}
    f02 = docs["F02"]
    assert f02["status"] == "탐지" and f02["first_check_pass_by_design"] is False
    hit = [f for f in f02["findings"] if f["check"] == "numbers"]
    assert hit and hit[0]["value"] == "1.205" and hit[0]["sentence_ids"] == f02["injected"]
    assert docs["F01"]["status"] == "미탐지"  # 허용 목록 안의 다른 값(FP 건수) — 1차 점검을 피한다
    assert "status" not in docs["V01"] and docs["dev_binary"]["source"] == "원본"
    saved = json.loads((ctx.root / "05_결과" / "baseline1.json").read_text(encoding="utf-8"))
    assert saved["docs"]["F02"]["status"] == "탐지"


def test_verify_baseline1_waits_for_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(je, "load_bases", lambda ctx: {t: b for t, b in zip(E.EVAL_REPORTS, bases().values())})
    with pytest.raises(je.RunnerError, match="동결 전"):
        je.cmd_baseline1(je.make_ctx("verify", tmp_path / "평가"))


B2_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["decision", "issues"],
    "properties": {"decision": {"type": "string", "enum": ["issue", "block"]},
                   "issues": {"type": "array", "items": {
                       "type": "object", "additionalProperties": False, "required": ["axis", "type", "quotes", "reason"],
                       "properties": {"axis": {"type": "string", "enum": ["faithfulness", "consistency"]}, "type": {"type": "string"},
                                      "quotes": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
                                      "reason": {"type": "string"}}}}}}


def write_b2_prompt(root: Path) -> Path:
    """사전 등록 v1 베이스라인 2 파일과 같은 꼴(절 제목·코드 블록·입력 틀)의 시험용 파일 + SHA256SUMS."""
    folder = root / "사전등록" / "v1"
    folder.mkdir(parents=True, exist_ok=True)
    body = ("# 베이스라인 2 — 시험용\n\n모델: `gpt-6.1-sol`, 추론 강도 `medium`(판정자와 같음).\n\n"
            "## 지시문 (instructions)\n\n```text\n시험용 지시문이다.\n둘째 줄.\n```\n\n"
            "## 입력 (input)\n\n```text\n[문서]\n<reading.render(pieces)>\n\n[데이터]\n<fact_sheet JSON>\n\n[고객 입력]\n"
            "<client_inputs JSON>\n```\n\n## 출력 스키마\n\n```json\n" + json.dumps(B2_SCHEMA, ensure_ascii=False, indent=2)
            + "\n```\n\n## 탐지로 세는 법\n\n- 시험.\n")
    path = folder / "베이스라인2_프롬프트.md"
    path.write_text(body, encoding="utf-8")
    (folder / "SHA256SUMS").write_text(f"{je.sha256_file(path)}  베이스라인2_프롬프트.md\n", encoding="utf-8")
    return path


def b2_client(planted_sentence: str, *, n: int, fail_first: bool = False):
    """judge_fakes.FakeOpenAI — 입력 글에 심은 문장이 있으면 그 문장을 인용한 지적(block), 없으면 지적 없음(issue)."""
    fakes = _load_fakes()

    def reply(kwargs):
        content = kwargs["input"][0]["content"]
        if planted_sentence in content:
            return fakes.json_response({"decision": "block", "issues": [
                {"axis": "faithfulness", "type": "숫자", "quotes": [planted_sentence], "reason": "데이터와 다른 건수"}]})
        return fakes.json_response({"decision": "issue", "issues": []})

    replies = ([fakes.timeout()] if fail_first else []) + [reply] * n
    return fakes.FakeOpenAI(replies)


def _load_fakes():
    sys.path.insert(0, str(ROOT / "tests"))
    import judge_fakes

    return judge_fakes


def test_baseline2_runs_each_doc_once_with_prereg_prompt_and_skips_done(planted):
    ctx = planted
    write_b2_prompt(ctx.root)
    f01_new = table_rows()[0]["new"]
    client = b2_client(f01_new, n=8)
    out = je.cmd_baseline2(ctx, yes=True, client=client)
    assert out["ran"] == 8 and out["gates"] == {"pass": 7, "blocked": 1}
    call = client.calls[0]
    assert call["model"] == "gpt-6.1-sol" and call["reasoning"] == {"effort": "medium"} and call["store"] is False
    assert call["instructions"] == "시험용 지시문이다.\n둘째 줄." and call["text"]["format"]["name"] == "baseline2"
    assert call["text"]["format"]["schema"] == B2_SCHEMA
    content = call["input"][0]["content"]
    assert content.startswith("[문서]\n[slot:") and "\n\n[데이터]\n{" in content and "\n\n[고객 입력]\n" in content
    rows = _rows(ctx.ledger_path)
    assert {r["stage"] for r in rows} == {"baseline2"} and sorted(r["run_id"] for r in rows) == [f"E-{i:04d}" for i in range(1, 9)]
    body = json.loads((ctx.root / "03_실행" / "B2-F01.json").read_text(encoding="utf-8"))
    assert body["result"]["decision"] == "block" and body["prompt_sha256"] == je.sha256_file(
        ctx.root / "사전등록" / "v1" / "베이스라인2_프롬프트.md")
    assert je.cmd_baseline2(ctx, yes=True, client=b2_client(f01_new, n=0))["ran"] == 0


def test_baseline2_refuses_changed_prompt_and_retries_api_failure(planted):
    ctx = planted
    path = write_b2_prompt(ctx.root)
    path.write_text(path.read_text(encoding="utf-8") + "\n덧붙임\n", encoding="utf-8")
    with pytest.raises(je.RunnerError, match="sha256 이 사전 등록 SHA256SUMS 와 다르다"):
        je.cmd_baseline2(ctx, yes=True, client=b2_client("※없는 문장※", n=8))
    write_b2_prompt(ctx.root)
    out = je.cmd_baseline2(ctx, yes=True, limit=2, concurrency=1, client=b2_client("※없는 문장※", n=2, fail_first=True))
    assert out["ran"] == 2 and out["gates"] == {"재시도 대상": 1, "pass": 1}
    failed = [r for r in _rows(ctx.ledger_path) if r["error"]]
    assert failed[0]["error"] == "timeout" and "-E-0001.json" in failed[0]["result_file"]
    out = je.cmd_baseline2(ctx, yes=True, limit=1, client=b2_client("※없는 문장※", n=1))
    assert out["ran"] == 1 and out["gates"] == {"pass": 1}  # 실패한 문서부터 다시
    states = je.stage_states(_rows(ctx.ledger_path), "baseline2", fingerprint=je.criteria.fingerprint(), mode="dev")
    assert sum(s["status"] == "done" for s in states.values()) == 2


# ── 집계·판정표·최종 보고 — 합성 채점 결과로 처음부터 끝까지 ──


def _v(cid, quote, piece, kind="number", verdict="violation", **detail):
    return {"claim_id": cid, "kind": kind, "verdict": verdict, "slot": piece.split(":")[1], "piece_id": piece, "quote": quote,
            "detail": detail}


def _c(cid, quote, piece, kind="number", field=""):
    return {"id": cid, "kind": kind, "quote": quote, "piece_id": piece, "slot": piece.split(":")[1], "field": field, "field_b": ""}


def synthetic_plan() -> dict:
    """문서 서술 해시 → 회차별 채점 결과. F01 두 회차 탐지(2회차엔 주입 문장 밖 표시도), C02 한 회차만(모순), F08 두 회차 미탐지
    (추론 주장이 추출됐지만 허용 — 규칙 허용), 원본 dev_binary 1회차만 주입과 무관한 위반(과잉 차단 후보·재현성 갈림)."""
    rows = {r["id"]: r for r in table_rows()}
    elsewhere = {"verdicts": [_v("k9", "불균형 비율(1.105)", "slot:7.2", expr="{facts.imbalance_ratio}", data=1.105)],
                 "claims": [_c("k9", "불균형 비율(1.105)", "slot:7.2", field="facts.imbalance_ratio")]}
    f01_hit = {"verdicts": [_v("k1", "거짓 음성(FN)은 56건", "slot:7.1", expr="{confusion_counts.1.fn}", data=43)],
               "claims": [_c("k1", "거짓 음성(FN)은 56건", "slot:7.1", field="confusion_counts.1.fn")]}
    added = "현 상태로 운영해도 무방할 것으로 보인다."
    mc = bases()["dev_multiclass"]
    rec = V.sentences(V.read_slot(mc.result, "recommendation_narrative.model_ops"))[0]
    c02_hit = {"contradictions": [{"type": "verdict_vs_recommendation", "pieces": ["slot:8.2", "slot:9:6.2"], "slots": ["8.2", "9"],
                                   "sources": ["template", "template"], "quotes": [added, rec], "detail": {}}]}
    f08 = rows["F08"]["new"]
    f08_spec = {"gate": "pass", "claims": [_c("k3", f08[:20], "slot:8.3", kind="inference")],
                "verdicts": [_v("k3", f08[:20], "slot:8.3", kind="inference", verdict="inference_ok", marked=False)]}
    return {
        _sha_of("F01"): [{"gate": "blocked", **f01_hit},
                         {"gate": "blocked", "verdicts": f01_hit["verdicts"] + elsewhere["verdicts"],
                          "claims": f01_hit["claims"] + elsewhere["claims"]}],
        _sha_of("C02"): [{"gate": "blocked", **c02_hit}, {"gate": "pass"}],
        _sha_of("F08"): [f08_spec, f08_spec],
        _sha_of("dev_binary"): [{"gate": "blocked", **elsewhere}, {"gate": "pass"}],
    }


@pytest.fixture
def scored(planted):
    ctx = planted
    judge = FakeJudge(per_doc=synthetic_plan())
    je.cmd_run(ctx, 1, yes=True, judge=judge, client=object())
    je.cmd_run(ctx, 2, yes=True, judge=judge, client=object())
    return ctx


def test_aggregate_refuses_incomplete_unless_partial(planted):
    ctx = planted
    je.cmd_run(ctx, 1, yes=True, judge=FakeJudge(), client=object(), only=["F01"])
    with pytest.raises(je.RunnerError, match="덜 끝났다"):
        je.cmd_aggregate(ctx)
    agg = je.cmd_aggregate(ctx, partial=True)
    assert agg["partial"] is True and "F01 2회차(미실행)" in agg["missing"]
    assert agg["defects"]["F01"]["gate"] == ["미탐지", "미실행"] and agg["defects"]["F01"]["gate_class"] == "한 회차 미탐지 · 한 회차 실패"
    assert (ctx.root / "05_결과" / "집계.md").read_text(encoding="utf-8").count("부분 집계") >= 1


def test_aggregate_counts_detection_causes_invariance_reproducibility(scored):
    ctx = scored
    je.cmd_baseline1(ctx)
    agg = je.cmd_aggregate(ctx)
    d = agg["defects"]
    assert agg["partial"] is False and agg["q1_gate"]["defects"] == 3
    assert d["F01"]["gate_class"] == "두 회차 모두 탐지" and d["F01"]["axes"]["faithfulness"]["class"] == "두 회차 모두 탐지"
    assert [o["count"] for o in d["F01"]["out_of_place"]] == [0, 1]
    assert d["C02"]["gate_class"] == "한 회차만 탐지" and d["C02"]["miss_causes"][0]["cause"] == "모순 미검출"
    assert d["F08"]["gate_class"] == "두 회차 모두 미탐지"
    assert [c["cause"] for c in d["F08"]["miss_causes"]] == ["규칙 허용", "규칙 허용"]
    assert agg["q1_gate"]["both_detected"]["k"] == 1 and agg["q1_gate"]["both_detected"]["n"] == 3
    assert agg["q2_axis"]["by_axis"]["consistency"]["any_run_missed"] == ["C02"]
    assert d["F01"]["independence"]["consistency"]["status"] in ("범위 안", "허용")
    assert agg["extraction_rate"]["related"] == {"k": 4, "n": 4, "rate": 1.0, "no_record": 0}  # F01·F08 × 2회
    assert agg["q3_invariance"]["variants"]["V01"]["status"] == "판단 불가(원본 갈림)"  # 원본 dev_binary 가 갈렸다
    rep = agg["q4_reproducibility"]
    assert rep["docs"]["dev_binary"]["group"] == "갈림" and rep["docs"]["dev_multiclass"]["group"] == "두 회차 모두 통과"
    assert rep["overall"]["n"] == 8 and rep["overall"]["decision_agreement"] == pytest.approx(6 / 8)
    assert agg["q3_blocked_before_judgement"]["count"] == 1
    assert agg["q5_baselines"]["defects"]["F01"]["baseline1"] == "미탐지"
    assert agg["costs"]["eval_r1"]["rows"] == 8 and agg["costs"]["total"]["cost_usd"] == pytest.approx(0.16)
    assert agg["numeric_recall"]["tokens"] > 0
    md = (ctx.root / "05_결과" / "집계.md").read_text(encoding="utf-8")
    assert "## Q1 게이트 탐지" in md and "두 회차 모두 탐지 1/3" in md and "규칙 허용" in md


def test_sheet_is_blind_grouped_and_report_tallies(scored):
    ctx = scored
    je.cmd_baseline1(ctx)
    rec = je.cmd_sheet(ctx, seed=11)
    sheet = ctx.root / "04_판정표" / "판정표.csv"
    key = ctx.root / "04_판정표" / "열쇠" / "열쇠.csv"
    text = sheet.read_text(encoding="utf-8")
    first, _, body = text.partition("\n")
    assert first == f"# 열쇠 sha256: {je.sha256_file(key)}" and rec["key_sha256"] == je.sha256_file(key)
    rows = list(csv.reader(body.splitlines()))
    assert tuple(rows[0]) == E.SHEET_COLUMNS
    items = rows[1:]
    assert len(items) == 1 and rec["flags"] == 2  # 원본 1회차와 F01 2회차(주입 밖)의 같은 문장·종류·필드 표시가 한 문항
    assert items[0][0] == "J001" and items[0][1] == "사실 일치 위반" and items[0][7] == "facts.imbalance_ratio"
    world = je.World(ctx)
    allowed = [x.text for idx in world.index.values() for x in idx.sentences] + ["1.105"]
    assert items and E.sheet_leaks([tuple(r) for r in items], doc_names=[d.id for d in world.docs], allowed_texts=allowed) == []
    for word in ("dev_binary", "F01", "E-00", "원본", "결함본", "회차"):
        assert word not in body, word  # 문서 이름·결함 ID·실행 번호·출처·회차가 표에 없다
    assert not os.access(key, os.W_OK) and not os.access(key.parent / "열쇠.sha256", os.W_OK)
    key_rows = list(csv.DictReader(key.open(encoding="utf-8")))
    assert {(k["문서"], k["출처"], k["회차"]) for k in key_rows} == {("dev_binary", "원본", "1"), ("F01", "결함본", "2")}
    with pytest.raises(je.RunnerError, match="이미 있다"):
        je.cmd_sheet(ctx, seed=11)
    # 판정 — 빈칸이면 열쇠를 열지 않는다
    blank = ctx.root.parent / "판정_빈칸.csv"
    blank.write_text(text, encoding="utf-8")
    with pytest.raises(je.RunnerError, match="판정을 마치지 않았다"):
        je.cmd_report(ctx, blank)
    filled = ctx.root.parent / "판정.csv"
    out_rows = [rows[0]] + [r[:9] + ["판정자 오판", "필드 값 1.105 와 문장의 1.105 가 같다"] for r in items]
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(out_rows)
    filled.write_text("\ufeff" + first + "\r\n" + buf.getvalue().replace("\n", "\r\n"), encoding="utf-8")  # 표 프로그램 저장 꼴(BOM·CRLF)
    report = je.cmd_report(ctx, filled)
    tally = report["overblocking"]
    assert tally["counts"]["과잉 차단"] == 1 and tally["counts"]["차단"] == 1
    assert tally["claim_level"]["wrong_faithfulness_violations"] == 1 and tally["claim_level"]["checked"] == 10 * 10
    assert tally["claim_level"]["inference"] == {"wrong": 0, "checked": 3 * 10, "rate": 0.0}  # 추론 판정 수는 따로(검토 P1-9)
    assert tally["claim_level"]["violation"]["wrong"] == 1
    assert tally["defect_out_of_place"]["판정자 오판"] == 1
    md = (ctx.root / "05_결과" / "최종보고.md").read_text(encoding="utf-8")
    assert "과잉 차단(판정자 오판만) | 1" in md and "# 판정자 평가 최종 보고" in md
    final = json.loads((ctx.root / "05_결과" / "최종보고.json").read_text(encoding="utf-8"))
    assert final["blind"] is False and final["q3_blocked_before_judgement"]["blocked_runs"]  # 판정 뒤에는 표시 세부까지
    assert "불균형 비율(1.105)" in json.dumps(final, ensure_ascii=False)
    # 열쇠 해시가 다른 표·도구 칸을 고친 표는 거부
    wrong = ctx.root.parent / "판정_다른열쇠.csv"
    wrong.write_text("# 열쇠 sha256: " + "0" * 64 + "\n" + buf.getvalue(), encoding="utf-8")
    with pytest.raises(je.RunnerError, match="짝과 맞지 않는다"):
        je.cmd_report(ctx, wrong)
    edited = ctx.root.parent / "판정_고침.csv"
    out_rows[1][2] = out_rows[1][2] + " 고침"
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(out_rows)
    edited.write_text(first + "\n" + buf.getvalue(), encoding="utf-8")
    with pytest.raises(je.RunnerError, match="판정 칸 밖을 고친"):
        je.cmd_report(ctx, edited)


def test_sheet_refuses_before_all_runs_finish_and_leaks(planted, monkeypatch):
    ctx = planted
    je.cmd_run(ctx, 1, yes=True, judge=FakeJudge(), client=object())
    with pytest.raises(je.RunnerError, match="덜 끝나"):
        je.cmd_sheet(ctx, seed=1)
    je.cmd_run(ctx, 2, yes=True, judge=FakeJudge(per_doc=synthetic_plan()), client=object())
    real = E.sheet_entries

    def leaky(*a, **k):
        return [{**e, "reason": e["reason"] + " (E-0001)"} for e in real(*a, **k)]

    monkeypatch.setattr(E, "sheet_entries", leaky)
    with pytest.raises(je.RunnerError, match="출처를 알려 주는 것"):
        je.cmd_sheet(ctx, seed=1)
    assert not (ctx.root / "04_판정표" / "판정표.csv").exists()


def test_cli_end_to_end_dev(planted, tmp_path):
    root = str(planted.root)
    assert je.main(["run", "--mode", "dev", "--root", root, "--round", "1", "--yes", "--only", "dev_multiclass"]) == 0
    assert je.main(["aggregate", "--mode", "dev", "--root", root, "--partial"]) == 0
    assert je.main(["baseline1", "--mode", "dev", "--root", root]) == 0
    assert je.main(["status", "--mode", "dev", "--root", root]) == 0
    assert je.main(["aggregate", "--mode", "dev", "--root", root]) == 2  # 덜 끝남 — --partial 없이 거부


def test_baseline2_detection_in_aggregate_and_mixed_into_sheet(scored):
    ctx = scored
    je.cmd_baseline1(ctx)
    write_b2_prompt(ctx.root)
    fakes = _load_fakes()
    f01_new = table_rows()[0]["new"]
    s82 = _sentence("dev_multiclass", "conclusion.narrative", 0)

    def reply(kwargs):
        content = kwargs["input"][0]["content"]
        issues = [{"axis": "faithfulness", "type": "숫자", "quotes": [q], "reason": "데이터와 다르다"} for q in (f01_new, s82) if q in content]
        return fakes.json_response({"decision": "block" if issues else "issue", "issues": issues})

    je.cmd_baseline2(ctx, yes=True, client=fakes.FakeOpenAI([reply] * 8))
    agg = je.cmd_aggregate(ctx)
    q5 = agg["q5_baselines"]
    assert q5["defects"]["F01"]["baseline2"] == "탐지" and q5["defects"]["F08"]["baseline2"] == "미탐지"
    assert q5["originals_variants"]["dev_multiclass"]["baseline2_issues"] == 1
    assert q5["originals_variants"]["dev_binary"]["baseline2_issues"] == 0
    rec = je.cmd_sheet(ctx, seed=4, include_b2=True)
    key_rec = json.loads((ctx.root / "04_판정표" / "열쇠" / "기록.json").read_text(encoding="utf-8"))
    assert key_rec["by_source"] == {"원본": 1, "결함본": 1, "베이스라인2": 1} and rec["items"] == 2
    body = (ctx.root / "04_판정표" / "판정표.csv").read_text(encoding="utf-8")
    assert ",지적," in body and "베이스라인" not in body


# ── 검토 고침 1 — P0·P1(고치기 전 코드에서 실패하던 시험) ──


def _run_both(ctx, plan):
    judge = FakeJudge(per_doc=plan)
    je.cmd_run(ctx, 1, yes=True, judge=judge, client=object())
    je.cmd_run(ctx, 2, yes=True, judge=judge, client=object())
    return judge


def test_review_p0_1_consistency_defect_may_name_a_metric_with_digits():
    """v3 2.8 — §3.4 숫자 검사는 §6.9 숫자 토큰으로(F1 의 1 은 숫자가 아니다). 진짜 숫자를 더하면 여전히 거부."""
    s82 = _sentence("dev_multiclass", "conclusion.narrative", 0)
    added = "F1 Score 미달은 판정에 영향을 주지 않으므로 별도의 조치 없이 현 상태로 운영해도 무방할 것으로 보인다."
    bad, plan = je.check_plant_row(_row("C03", "dev_multiclass", "conclusion.narrative", s82, f"{s82} {added}"), bases(),
                                   mode="dev", existing=set())
    assert bad == [] and plan["meta"]["injected"] == [[1, added]]
    bad, _ = je.check_plant_row(_row("C03", "dev_multiclass", "conclusion.narrative", s82, f"{s82} {added.replace('미달은', '0.5 미달은')}"),
                                bases(), mode="dev", existing=set())
    assert any("숫자를 바꾸지 않는다" in p for p in bad)


def test_review_p0_2_plant_takes_v3_allowed_secondary_axes():
    r0 = _sentence("dev_binary", "conclusion.risks", 0)
    row = _row("F03", "dev_binary", "conclusion.risks", r0, r0[:-1] + " 그렇다.", allowed_secondary_axes=["consistency"])
    assert je.check_plant_row(row, bases(), mode="dev", existing=set())[0] == []
    bad, _ = je.check_plant_row({**row, "allowed_secondary_axes": []}, bases(), mode="dev", existing=set())
    assert any("허용 부차 축이 사전 등록과 다르다" in p for p in bad)


def test_review_p0_3_aggregate_before_judgement_hides_flag_sources(scored):
    ctx = scored
    je.cmd_baseline1(ctx)
    agg = je.cmd_aggregate(ctx)
    quote = "불균형 비율(1.105)"  # 원본 1회차·F01 2회차의 위치 불일치 표시 인용 = 판정표 J001 의 내용
    for name in ("집계.json", "집계.md"):
        assert quote not in (ctx.root / "05_결과" / name).read_text(encoding="utf-8"), name
    assert agg["blind"] is True and agg["q3_blocked_before_judgement"] == {"count": 1}
    assert all("flags" not in r for d in agg["docs"].values() for r in d["runs"])
    assert all("flags" not in o for d in agg["defects"].values() for o in d["out_of_place"])
    assert all("sets" not in x for x in agg["q4_reproducibility"]["docs"].values())
    assert all("judge_flags" not in x for x in agg["q5_baselines"]["originals_variants"].values())
    rec = je.cmd_sheet(ctx, seed=11)
    public = json.loads((ctx.root / "04_판정표" / "기록.json").read_text(encoding="utf-8"))
    assert "by_source" not in public and "seed" not in public and "seed" not in rec and public["key_sha256"] == rec["key_sha256"]
    key_rec_path = ctx.root / "04_판정표" / "열쇠" / "기록.json"
    key_rec = json.loads(key_rec_path.read_text(encoding="utf-8"))
    assert key_rec["seed"] == 11 and key_rec["by_source"] == {"원본": 1, "결함본": 1} and not os.access(key_rec_path, os.W_OK)
    assert not any("시드 11" in line for line in ctx.lines)


def test_review_p1_2_sheet_neighbors_come_from_original(planted):
    ctx = planted
    dist = V.sentences(V.read_slot(bases()["dev_binary"].result, "interpretation.distribution_analysis"))
    assert "200건이며," in dist[0] and "불균형 비율(1.105)" in dist[1]
    flag = {"gate": "blocked", "verdicts": [_v("k9", "불균형 비율(1.105)", "slot:7.2", expr="{facts.imbalance_ratio}", data=1.105)],
            "claims": [_c("k9", "불균형 비율(1.105)", "slot:7.2", field="facts.imbalance_ratio")]}
    _run_both(ctx, {_sha_of("V01"): [flag, {"gate": "pass"}]})  # 변형 V01 만 — V01 이 바꾼 문장(200건이고,)의 뒤 문장을 표시
    je.cmd_sheet(ctx, seed=3)
    body = (ctx.root / "04_판정표" / "판정표.csv").read_text(encoding="utf-8")
    rows = list(csv.reader(body.splitlines()[1:]))
    assert len(rows) == 2 and rows[1][4] == dist[0]  # 앞 문장은 원본 글
    assert "200건이고," not in body


def test_review_p1_3_code_exceptions_are_final_not_retried(planted):
    ctx = planted
    boom = FakeJudge(raise_once={_sha_of("F01"): ValueError("매번 나는 코드 버그(시험)"),
                                 _sha_of("C02"): ConnectionError("끊김(시험)")})
    out = je.cmd_run(ctx, 1, yes=True, judge=boom, client=object())
    assert out["gates"] == {"pass": 6, "최종 실패": 1, "재시도 대상": 1}
    err = {E.doc_id(r["report"], r["variant"]): r["error"] for r in _rows(ctx.ledger_path)}
    assert err["F01"].startswith("exception:ValueError") and err["C02"].startswith("api_exception:ConnectionError")
    again = FakeJudge()
    out = je.cmd_run(ctx, 1, yes=True, judge=again, client=object())
    assert again.calls == 1 and out["ran"] == 1  # 네트워크 예외(C02)만 다시 — 코드 예외(F01)는 최종 실패(§7.4)
    states = je.stage_states(_rows(ctx.ledger_path), "eval_r1", fingerprint=je.criteria.fingerprint(), mode="dev")
    assert states[("dev_binary", "F01")]["status"] == "final" and states[("dev_multiclass", "C02")]["status"] == "done"
    import openai

    assert je.exception_error(openai.APITimeoutError(request=None)).startswith("api_exception:")  # type: ignore[arg-type]
    assert je.exception_error(asyncio.TimeoutError()).startswith("api_exception:")
    assert je.exception_error(KeyError("x")).startswith("exception:")


def test_review_p1_4_verify_reads_env_before_model_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(je.criteria, "version", lambda: "1.25")
    monkeypatch.setattr(je.criteria, "version_problem", lambda *a, **k: None)
    manifest = je.criteria.recorded_manifest()

    def git(*args):
        return (0, "") if args[0] == "status" else (0, manifest)

    monkeypatch.delenv("JUDGE_EFFORT", raising=False)
    order = []

    def dotenv_with_other_effort():  # .env 에 다른 추론 강도가 있는 경우
        order.append("env")
        monkeypatch.setenv("JUDGE_EFFORT", "high")

    monkeypatch.setattr(je, "load_env", dotenv_with_other_effort, raising=False)
    ctx = je.make_ctx("verify", tmp_path / "평가")
    with pytest.raises(je.RunnerError, match="판정자 추론 강도"):
        je.cmd_run(ctx, 1, live=True, yes=True, judge=FakeJudge(), client=object(), git=git)
    assert order == ["env"]
    monkeypatch.delenv("JUDGE_EFFORT")
    write_b2_prompt(ctx.root)
    with pytest.raises(je.RunnerError, match="판정자 추론 강도"):
        je.cmd_baseline2(ctx, live=True, yes=True, client=object(), git=git)


def test_review_p1_4_cap_cannot_be_raised_above_prereg_in_verify(monkeypatch):
    monkeypatch.delenv("JUDGE_LIVE_BUDGET_USD", raising=False)
    monkeypatch.delenv("JUDGE_LIVE_MAX_RUNS", raising=False)
    assert je.Cap.from_env("verify") == je.Cap(30.0, 110)
    monkeypatch.setenv("JUDGE_LIVE_BUDGET_USD", "40")
    with pytest.raises(je.RunnerError, match="§7.7"):
        je.Cap.from_env("verify")
    assert je.Cap.from_env("dev").usd == 40.0  # dev(시험)는 아무 값이나
    monkeypatch.setenv("JUDGE_LIVE_BUDGET_USD", "12")
    monkeypatch.setenv("JUDGE_LIVE_MAX_RUNS", "111")
    with pytest.raises(je.RunnerError, match="§7.7"):
        je.Cap.from_env("verify")
    monkeypatch.setenv("JUDGE_LIVE_MAX_RUNS", "80")
    assert je.Cap.from_env("verify") == je.Cap(12.0, 80)  # 낮추는 것은 된다


def test_review_p1_6_aggregate_reports_axis_level_misses(planted):
    ctx = planted
    added = "현 상태로 운영해도 무방할 것으로 보인다."
    faith_hit = {"gate": "blocked", "verdicts": [_v("k5", added, "slot:8.2", kind="degree")],
                 "claims": [_c("k5", added, "slot:8.2", kind="degree")]}
    _run_both(ctx, {_sha_of("C02"): [faith_hit, {"gate": "pass"}]})  # 1회차는 충실성으로만 잡힘 — 일관성 축은 놓침
    agg = je.aggregate(ctx, partial=True)
    c02 = agg["defects"]["C02"]
    assert c02["gate"] == ["탐지", "미탐지"] and c02["axes"]["consistency"]["runs"] == ["미탐지", "미탐지"]
    assert [(c["round"], c["axis"], c["cause"]) for c in c02["axis_miss_causes"]] == [
        (1, "consistency", "모순 미검출"), (2, "consistency", "모순 미검출")]
    assert [(c["round"], c["axis"], c["cause"]) for c in c02["miss_causes"]] == [(2, None, "모순 미검출")]
    assert agg["q2_axis"]["by_axis"]["consistency"]["any_run_missed"] == ["C02"]
    assert "| C02 | 1 | 축 consistency | 모순 미검출 |" in je.aggregate_markdown(agg)


def test_review_p1_7_hidden_files_do_not_break_freeze(planted):
    ctx = planted
    je.cmd_precheck(ctx)
    for path in (ctx.root / "02_결함" / ".DS_Store", ctx.root / "02_결함" / "F01" / ".DS_Store"):
        path.write_bytes(b"\0finder")
    assert je.frozen_problems(ctx) == []
    meta = ctx.root / "02_결함" / "F01" / "meta.json"
    meta.write_text(meta.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert any("바뀌었다" in p for p in je.frozen_problems(ctx))


def test_review_p1_8_baseline2_reason_may_use_source_words(scored):
    ctx = scored
    je.cmd_baseline1(ctx)
    write_b2_prompt(ctx.root)
    fakes = _load_fakes()
    s82 = _sentence("dev_multiclass", "conclusion.narrative", 0)
    reason = "원본 데이터와 다르다 — 변형된 값으로 보이며 베이스라인보다 낮다"

    def reply(kwargs):
        hit = s82 in kwargs["input"][0]["content"]
        issues = [{"axis": "faithfulness", "type": "숫자", "quotes": [s82], "reason": reason}] if hit else []
        return fakes.json_response({"decision": "block" if issues else "issue", "issues": issues})

    je.cmd_baseline2(ctx, yes=True, client=fakes.FakeOpenAI([reply] * 8))
    rec = je.cmd_sheet(ctx, seed=4, include_b2=True)
    body = (ctx.root / "04_판정표" / "판정표.csv").read_text(encoding="utf-8")
    assert rec["items"] == 2 and reason in body
    # 문서 이름은 사유 글 안이라도 잡는다
    row = ("J001", "지적", s82, "", "", "", "pima_strong 원본 데이터와 다르다", "", "", "", "")
    assert any("문서 이름" in p for p in E.sheet_leaks([row], allowed_texts=[s82, "pima_strong 원본 데이터와 다르다"]))


def test_review_v3_2_1_original_flagged_defects_leave_denominator(planted):
    ctx = planted
    plan = synthetic_plan()
    pre = {"gate": "blocked", "verdicts": [_v("k1", "거짓 음성(FN)은 43건", "slot:7.1", expr="{confusion_counts.1.fn}", data=43)],
           "claims": [_c("k1", "거짓 음성(FN)은 43건", "slot:7.1", field="confusion_counts.1.fn")]}
    plan[_sha_of("dev_binary")] = [pre, {"gate": "pass"}]  # 원본 1회차가 F01 의 원래 문장을 이미 표시
    _run_both(ctx, plan)
    agg = je.aggregate(ctx, partial=True)
    f01 = agg["defects"]["F01"]
    assert f01["gate_class"] == "두 회차 모두 탐지" and f01["original_flagged"]["gate"] is True
    assert f01["original_flagged"]["axes"] == {"faithfulness": True, "consistency": False}
    q1 = agg["q1_gate"]
    assert q1["original_flagged"] == ["F01"] and q1["counted"] == 2 and (q1["both_detected"]["k"], q1["both_detected"]["n"]) == (0, 2)
    assert {"defect": "F01", "class": "원본에서 이미 표시"} in q1["by_type"]["숫자"]
    fa = agg["q2_axis"]["by_axis"]["faithfulness"]
    assert fa["original_flagged"] == ["F01"] and fa["counted"] == 1
    assert agg["defects"]["C02"]["original_flagged"]["applies"] is False  # 덧붙이기 결함은 해당 없음
    # v3 2.4 — V02(dev 표의 문장 순서 교체)는 결정과 차단 축만, V01 은 네 축
    inv = agg["q3_invariance"]["variants"]
    assert inv["V02"]["judged_axes"] == ["faithfulness", "consistency"] and len(inv["V01"]["judged_axes"]) == 4


def test_review_v3_defect_sums_are_checked(planted):
    ctx = planted
    out = je.cmd_precheck(ctx)
    v3 = ctx.root / "사전등록" / "v3"
    v3.mkdir(parents=True)
    assert out["files"]
    (v3 / "결함_SHA256SUMS").write_text("".join(f"{d}  {n}\n" for n, d in out["files"].items()), encoding="utf-8")
    pre = ctx.root / "01_구성" / "precheck.json"
    (v3 / "SHA256SUMS").write_text(f"{je.sha256_file(pre)}  precheck.json\n", encoding="utf-8")
    assert je.frozen_problems(ctx) == [] and je.integrity(ctx)["defects_v3"] == {"ok": True, "mismatch": []}
    short = (v3 / "결함_SHA256SUMS").read_text(encoding="utf-8").replace("  02_결함/", "  ")
    (v3 / "결함_SHA256SUMS").write_text(short, encoding="utf-8")  # 'F01/meta.json' 꼴 이름도 받는다
    assert je.frozen_problems(ctx) == []
    (ctx.root / "02_결함" / "F01" / "메모.txt").write_text("나중에 더함", encoding="utf-8")
    pre.write_text(pre.read_text(encoding="utf-8") + " ", encoding="utf-8")
    problems = je.frozen_problems(ctx)
    assert any("결함_SHA256SUMS 에 없는 02_결함 파일: 02_결함/F01/메모.txt" in p for p in problems)
    assert any("SHA256SUMS 와 다르다: precheck.json" in p for p in problems)


# ── 검토 고침 2 — 살아남은 변이를 죽이는 시험 ──


def test_teeth_tampered_meta_injected_is_refused(planted):
    ctx = planted
    path = ctx.root / "02_결함" / "F01" / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["injected"] = [[0, "손으로 고친 문장."]]
    path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(je.RunnerError, match="meta.injected"):
        je.World(ctx)  # 변이 I6


def test_teeth_midrun_cap_counts_runs_in_flight(planted, monkeypatch):
    """동시 3개 — 상한 검사는 아직 장부에 없는 도는 회차도 회당 예상으로 센다(변이 R6)."""
    ctx = planted
    monkeypatch.setenv("JUDGE_LIVE_BUDGET_USD", "2.4")  # 시작: 8 × $0.30 = $2.40 — 딱 맞아 시작은 된다
    delays = iter([0.01, 0.3, 0.3] + [0.01] * 10)
    judge = FakeJudge(cost=1.0, delay=lambda: next(delays))
    out = je.cmd_run(ctx, 1, concurrency=3, yes=True, judge=judge, client=object())
    # 1번째가 끝난 뒤: $1.00 + (도는 2 + 새 1) × $0.30 = $1.90 → 띄움. 4번째가 끝난 뒤: $2.00 + 3 × $0.30 = $2.90 → 멈춤
    assert out["ran"] == 4 and len(out["stopped"]) == 4 and judge.peak == 3


def _slow_b2(delays):
    """동시에 몇 개가 불렸는지 세는 가짜 OpenAI(베이스라인 2) — 부른 차례마다 delays 만큼 쉰다."""
    fakes = _load_fakes()

    class Slow(fakes.FakeOpenAI):
        now = peak = 0

        async def create(self, **kwargs):
            Slow.now += 1
            Slow.peak = max(Slow.peak, Slow.now)
            try:
                await asyncio.sleep(next(delays))
                return await super().create(**kwargs)
            finally:
                Slow.now -= 1

    return Slow, fakes


def test_teeth_baseline2_respects_concurrency(planted):
    ctx = planted
    write_b2_prompt(ctx.root)
    Slow, fakes = _slow_b2(iter([0.02] * 8))
    out = je.cmd_baseline2(ctx, yes=True, concurrency=2, client=Slow([fakes.json_response({"decision": "issue", "issues": []})] * 8))
    assert out["ran"] == 8 and Slow.peak == 2


def test_teeth_baseline2_midrun_cap_counts_calls_in_flight(planted, monkeypatch):
    """베이스라인 2 도 run 과 같은 상한 — 호출마다 실비 $1.00(입력 50만 토큰 × $2/백만), 동시 3개."""
    ctx = planted
    write_b2_prompt(ctx.root)
    monkeypatch.setenv("JUDGE_LIVE_BUDGET_USD", "2.4")
    Slow, fakes = _slow_b2(iter([0.01, 0.3, 0.3] + [0.01] * 10))
    costly = fakes.json_response({"decision": "issue", "issues": []}, use=fakes.usage(input_tokens=500_000, output_tokens=0,
                                                                                         reasoning=0))
    out = je.cmd_baseline2(ctx, yes=True, concurrency=3, client=Slow([costly] * 8))
    assert out["ran"] == 4 and len(out["stopped"]) == 4 and Slow.peak == 3
    assert [float(r["cost_usd"]) for r in _rows(ctx.ledger_path)] == [pytest.approx(1.0)] * 4


def test_teeth_aggregate_passes_v3_expected_changes(planted, monkeypatch):
    """v3 2.6 '예상된 변화'를 집계가 결함마다 넘긴다(dev 표에는 C03~C06 이 없어 C02 로 시험)."""
    ctx = planted
    monkeypatch.setitem(E.EXPECTED_CHANGES, "C02", ("coverage",))
    low = {"gate": "blocked", "card": {"faithfulness": {"checked": 10, "matched": 10}, "coverage": {"applicable": 5, "described": 1},
                                       "consistency": {"contradictions": 0}, "readability": {"violations": 0}}}
    _run_both(ctx, {_sha_of("C02"): [low, low]})
    agg = je.aggregate(ctx, partial=True)
    assert agg["defects"]["C02"]["independence"]["coverage"]["status"] == "예상된 변화"
    assert agg["defects"]["F08"]["independence"]["coverage"]["status"] == "범위 안"


def test_teeth_spent_cap_boundaries_and_no_overwrite(tmp_path):
    assert je.spent([{"cost_usd": ""}, {"cost_usd": "0.5"}, {"cost_usd": " "}]) == (pytest.approx(1.1), 3)  # 변이 R8
    assert je.cap_problem([], je.Cap(30.0, 2), 2) is None and je.cap_problem([], je.Cap(30.0, 2), 3)  # 변이 R7
    assert je.cap_problem([], je.Cap(0.6, 110), 2) is None and je.cap_problem([], je.Cap(0.59, 110), 2)
    (tmp_path / "E-0001.json").write_text("먼저", encoding="utf-8")

    async def main():
        writer = je.LedgerWriter(tmp_path / "장부.csv")
        await writer.commit(lambda run_id, rows: ({"stage": "eval_r1"}, [(tmp_path / f"{run_id}.json", "{}")]))

    with pytest.raises(je.RunnerError, match="덮어쓰지 않는다"):
        asyncio.run(main())  # 변이 R9
    assert (tmp_path / "E-0001.json").read_text(encoding="utf-8") == "먼저" and je.jl.Ledger(tmp_path / "장부.csv").rows() == []


def test_teeth_ledger_ids_unique_across_processes(tmp_path):
    """같은 평가 폴더에 run 을 두 프로세스로 띄워도 실행 번호가 겹치지 않는다 — 장부 flock(변이 R4)."""
    ledger = tmp_path / "장부.csv"
    je.jl.Ledger(ledger)
    script = textwrap.dedent(f"""
        import asyncio, sys, time
        from pathlib import Path
        sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / "scripts")!r}]
        import judge_eval as je
        tag, folder = sys.argv[1], Path(sys.argv[2])
        (folder / f"ready-{{tag}}").touch()
        while not (folder / "go").exists():
            time.sleep(0.005)

        def build(run_id, rows):
            time.sleep(0.002)  # 장부를 읽은 뒤 쓰기 전 — 잠금이 없으면 다른 프로세스가 같은 번호를 읽는다
            return {{"stage": "eval_r1", "report": tag, "variant": run_id}}, []

        async def main():
            writer = je.LedgerWriter(folder / "장부.csv")
            for _ in range(40):
                await writer.commit(build)

        asyncio.run(main())
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script, tag, str(tmp_path)], cwd=str(ROOT)) for tag in ("a", "b")]
    try:
        deadline = time.monotonic() + 60
        while not all((tmp_path / f"ready-{t}").exists() for t in ("a", "b")):
            assert time.monotonic() < deadline and all(p.poll() is None for p in procs)
            time.sleep(0.01)
        (tmp_path / "go").touch()
        assert all(p.wait(timeout=60) == 0 for p in procs)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    ids = [r["run_id"] for r in je.jl.Ledger(ledger).rows()]
    assert len(ids) == 80 and len(set(ids)) == 80


@pytest.mark.parametrize("error, again", [
    ("timeout", True), ("transient", True), ("api_error", True), ("error", True), ("deadline", True), ("no_client", True),
    ("api_exception:ConnectionError: x", True), ("uncited", False), ("bad_output", False), ("refusal", False),
    ("exception:ValueError: x", False), ("", False)])
def test_teeth_retryable_is_api_failure_and_timeout_only(error, again):
    assert je.retryable(error) is again  # 변이 F8


def test_teeth_finished_row_wins_and_plan_order_is_checked(planted):
    fp = je.criteria.fingerprint()
    done = {"stage": "eval_r1", "report": "r", "variant": "base", "mode": "dev", "manifest_sha256": fp,
            "result_file": "03_실행/E-0001.json", "gate": "pass", "error": "", "run_id": "E-0001"}
    later = {**done, "run_id": "E-0002", "gate": "", "error": "timeout", "result_file": "03_실행/E-0002.json"}
    st = je.stage_states([done, later], "eval_r1", fingerprint=fp, mode="dev")[("r", "base")]
    assert st["status"] == "done" and st["row"]["run_id"] == "E-0001" and st["attempts"] == 2  # 변이 F3
    ctx = planted
    je.cmd_run(ctx, 1, yes=True, judge=FakeJudge(), client=object(), only=["F01"])
    plan_path = ctx.root / "03_실행" / "plan-r1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["order"] = list(reversed(plan["order"]))
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(je.RunnerError, match="순서가 지금 문서 목록과 다르다"):
        je.cmd_run(ctx, 1, yes=True, judge=FakeJudge(), client=object())  # 변이 F11


def test_teeth_final_failed_defect_run_is_failure_without_cause(planted):
    ctx = planted
    plan = synthetic_plan()
    plan[_sha_of("F08")] = [{"failure": "uncited"}, plan[_sha_of("F08")][1]]
    _run_both(ctx, plan)
    agg = je.aggregate(ctx, partial=True)
    f08 = agg["defects"]["F08"]
    assert f08["gate"] == ["실패", "미탐지"] and f08["gate_class"] == "한 회차 미탐지 · 한 회차 실패"  # 변이 F13
    assert [c["round"] for c in f08["miss_causes"]] == [2] and [c["round"] for c in f08["axis_miss_causes"]] == [2]  # 변이 C4
    first = agg["docs"]["F08"]["runs"][0]
    assert (first["status"], first["failure"]) == ("실패", "uncited")


def _filled_sheet(ctx, rows, first, path_name):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    path = ctx.root.parent / path_name
    path.write_text(first + "\n" + buf.getvalue(), encoding="utf-8")
    return path


def test_teeth_report_refuses_edited_tool_columns_and_blank_reasons(scored):
    ctx = scored
    je.cmd_baseline1(ctx)
    je.cmd_sheet(ctx, seed=11)
    first, _, body = (ctx.root / "04_판정표" / "판정표.csv").read_text(encoding="utf-8").partition("\n")
    rows = list(csv.reader(body.splitlines()))
    filled = [rows[0]] + [r[:9] + ["판정자 오판", "필드 값 1.105 와 같다"] for r in rows[1:]]
    assert len(filled) == 2
    for col in (6, 7, 8):  # 판정자 사유·대상 필드·필드 값(변이 S19)
        edited = [r[:] for r in filled]
        edited[1][col] += " 고침"
        with pytest.raises(je.RunnerError, match="판정 칸 밖을 고친"):
            je.cmd_report(ctx, _filled_sheet(ctx, edited, first, f"고침{col}.csv"))
    no_reason = [r[:] for r in filled]
    no_reason[1][10] = ""
    with pytest.raises(je.RunnerError, match="판정 사유 빈칸"):
        je.cmd_report(ctx, _filled_sheet(ctx, no_reason, first, "사유없음.csv"))  # 변이 S20
    assert je.cmd_report(ctx, _filled_sheet(ctx, filled, first, "판정.csv"))["overblocking"]["counts"]["과잉 차단"] == 1


def test_teeth_sheet_allows_source_words_inside_data_values(planted):
    ctx = planted
    s82 = _sentence("dev_multiclass", "conclusion.narrative", 0)
    flag = {"gate": "blocked", "verdicts": [_v("k1", s82[:12], "slot:8.2", kind="name", data="변형 클래스")],
            "claims": [_c("k1", s82[:12], "slot:8.2", kind="name")]}
    _run_both(ctx, {_sha_of("dev_multiclass"): [flag, {"gate": "pass"}]})
    rec = je.cmd_sheet(ctx, seed=2)
    body = (ctx.root / "04_판정표" / "판정표.csv").read_text(encoding="utf-8")
    assert rec["items"] == 1 and "변형 클래스" in body  # 데이터 값의 낱말은 출처가 아니다(변이 S16)


# ── 2026-11-05 실호출 E-0003: 판정은 났지만 호출 하나가 응답 없이 API 오류(크레딧 소진 429)로 끝남 → §7.4 API 실패로 다시 돈다 ──


def _call(error="", response_model="gpt-6.1-sol"):
    return {"purpose": "faithfulness", "requested_model": "gpt-6.1-sol", "response_model": response_model, "attempts": 2,
            "error": error, "status": "completed" if response_model else None, "cost_usd": 0.01 if response_model else None}


def test_failed_calls_finds_only_calls_without_a_response():
    err = "RateLimitError: Error code: 429 - insufficient_quota"
    record = {"faithfulness": {"calls": [_call(), _call(err), _call(err, response_model=None)]},
              "extract": {"calls": [_call()]}}
    got = je.failed_calls(record)
    assert len(got) == 1 and got[0]["response_model"] is None  # 두 번째 시도에서 받은 호출(오류 글 있음)은 실패가 아니다
    assert je.failed_calls({"faithfulness": {"calls": [_call(err)]}}) == []
    assert je.failed_calls({}) == []


def test_judged_run_with_a_failed_call_is_retried_not_done(tmp_path):
    fp = je.criteria.fingerprint()
    (tmp_path / "03_실행").mkdir()
    bad = {"record": {"faithfulness": {"calls": [_call(), _call("RateLimitError: 429", response_model=None)]}}}
    good = {"record": {"faithfulness": {"calls": [_call(), _call("RateLimitError: 429")]}}}
    (tmp_path / "03_실행" / "E-0003.json").write_text(json.dumps(bad), encoding="utf-8")
    (tmp_path / "03_실행" / "E-0004.json").write_text(json.dumps(good), encoding="utf-8")
    row = {"stage": "eval_r1", "report": "r", "variant": "C03", "mode": "live", "manifest_sha256": fp,
           "result_file": "03_실행/E-0003.json", "gate": "blocked", "error": "", "run_id": "E-0003"}
    assert je.row_status(row, fingerprint=fp, mode="live") == "done"  # 결과 원본을 안 보면 끝난 줄로 보인다
    assert je.row_status(row, fingerprint=fp, mode="live", root=tmp_path) == "retry"
    ok = {**row, "result_file": "03_실행/E-0004.json", "run_id": "E-0004"}
    assert je.row_status(ok, fingerprint=fp, mode="live", root=tmp_path) == "done"
    st = je.stage_states([row, ok], "eval_r1", fingerprint=fp, mode="live", root=tmp_path)[("r", "C03")]
    assert st["status"] == "done" and st["row"]["run_id"] == "E-0004" and st["attempts"] == 2
    marked = {**ok, "error": je.API_CALL_FAILED}
    assert je.retryable(je.API_CALL_FAILED) and je.row_status(marked, fingerprint=fp, mode="live") == "retry"
