"""판정자 평가 실험 실행기(scripts/judge_eval.py, 사전 등록 v1) 시험.

실호출은 하지 않는다. dev 모드(개발 입력 3건의 규칙 문장 서술 + 가짜 OpenAI)와 임시 평가 폴더만 쓴다. 평가 폴더(JUDGE_ROUND/평가)와
검증 폴더는 열지 않는다(dev 경계 시험이 거부를 본다).
"""
from __future__ import annotations

import asyncio
import csv
import importlib.util
import io
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
        self.raise_once = set(raise_once or ())
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
                self.raise_once.discard(sha)
                raise ConnectionError("끊김(시험)")
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
                "faithfulness": {"checked": 10, "matched": 10}, "coverage": {"applicable": 5, "described": 5},
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
    assert err["F01"].startswith("exception:ConnectionError") and err["C02"] == "uncited" and err["V01"] == "timeout"
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
    assert tally["defect_out_of_place"]["판정자 오판"] == 1
    md = (ctx.root / "05_결과" / "최종보고.md").read_text(encoding="utf-8")
    assert "과잉 차단(판정자 오판만) | 1" in md and "# 판정자 평가 최종 보고" in md
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
    assert rec["by_source"] == {"원본": 1, "결함본": 1, "베이스라인2": 1} and rec["items"] == 2
    body = (ctx.root / "04_판정표" / "판정표.csv").read_text(encoding="utf-8")
    assert ",지적," in body and "베이스라인" not in body
