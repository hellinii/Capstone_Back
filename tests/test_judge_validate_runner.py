"""심판 검증 실행기(scripts/judge_validate.py, 계획 17-3) 시험 — 심판 라운드 J8.

실호출은 하지 않는다. dev 모드는 가짜 OpenAI(scripts/dev_fake_llm_server, '문제 없음' 답)와 개발 입력(규칙 문장 서술)만 쓴다.
- verify 시작 조건 다섯: 하나씩만 어긋나게 해서 그 사유 문장만 나오는지.
- dev 읽기 경계: 검증 폴더 안 경로(장부·입력·결함·출력)는 열기 전에 거부.
- 이어 하기·순서 섞기(시드)·시작 확인·묶음 회차 상한·라운드 상한·묶음 예산 멈춤.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("judge_validate", ROOT / "scripts" / "judge_validate.py")
jv = importlib.util.module_from_spec(_SPEC)
sys.modules["judge_validate"] = jv  # dataclass 가 모듈을 찾는다
_SPEC.loader.exec_module(jv)

from app.judge import criteria  # noqa: E402
from app.judge.validation import runs as R  # noqa: E402


@pytest.fixture
def vdir(tmp_path, monkeypatch):
    """가짜 검증 폴더(미끼 파일 하나) — dev 는 이 안을 건드리면 안 된다."""
    d = tmp_path / "검증"
    (d / "00_봉인").mkdir(parents=True)
    (d / "00_봉인" / "미끼.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("JUDGE_VALIDATION_DIR", str(d))
    monkeypatch.delenv("JUDGE_LIVE_BUDGET_USD", raising=False)
    monkeypatch.delenv("JUDGE_LIVE_MAX_RUNS", raising=False)
    return d


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _dev(tmp_path, batch="reproducibility", **kw) -> "jv.RunArgs":
    base = dict(mode="dev", batch=batch, ledger=tmp_path / "work" / "장부.csv", out=tmp_path / "work" / "runs", yes=True,
                seed=7, reports=["dev_binary"], repeats=2)
    base.update(kw)
    return jv.RunArgs(**base)


def _quiet():
    lines: list[str] = []
    return lines, lines.append


# ── verify 시작 조건 다섯 ──


def _git_ok(manifest: str):
    def git(*args):
        if args[:1] == ("status",):
            return 0, ""
        if args[:1] == ("show",):
            return 0, manifest
        raise AssertionError(args)
    return git


@pytest.fixture
def frozen(monkeypatch):
    """기준이 1.0 으로 고정된 것처럼(지금 사본은 1.0-dev)."""
    monkeypatch.setattr(criteria, "version", lambda: "1.0")
    monkeypatch.setattr(criteria, "version_problem", lambda *a, **k: None)


def test_verify_all_conditions_met_has_no_problem(tmp_path, frozen):
    ledger = tmp_path / "장부.csv"
    assert jv.verify_problems(live=True, ledger_path=ledger, git=_git_ok(criteria.recorded_manifest())) == []


@pytest.mark.parametrize("case, expected", [
    ("dirty", "작업 트리가 깨끗하지 않다"),
    ("unfrozen", "기준 버전이 고정되지 않았다"),
    ("no_tag", "태그 judge-criteria-v1.0 가 없다"),
    ("manifest", "MANIFEST 가 태그 judge-criteria-v1.0 의 MANIFEST 와 다르다"),
    ("no_live", "--live 가 없다"),
    ("ledger", "장부에 쓸 수 없다"),
])
def test_verify_each_condition_gives_its_own_reason(tmp_path, monkeypatch, frozen, case, expected):
    manifest = criteria.recorded_manifest()
    git = _git_ok(manifest)
    live, ledger = True, tmp_path / "장부.csv"
    if case == "dirty":
        git = lambda *a: (0, " M app/x.py\n") if a[0] == "status" else _git_ok(manifest)(*a)  # noqa: E731
    elif case == "unfrozen":
        monkeypatch.setattr(criteria, "version", lambda: "1.0-dev")
    elif case == "no_tag":
        git = lambda *a: (128, "") if a[0] == "show" else _git_ok(manifest)(*a)  # noqa: E731
    elif case == "manifest":
        git = _git_ok(manifest.replace(manifest[:8], "00000000", 1))
    elif case == "no_live":
        live = False
    else:
        ledger = tmp_path / "없는폴더" / "장부.csv"
    problems = jv.verify_problems(live=live, ledger_path=ledger, git=git)
    assert len(problems) == 1 and expected in problems[0], problems


def test_verify_manifest_must_also_match_files(tmp_path, monkeypatch, frozen):
    # 태그와 기록된 MANIFEST 는 같아도 파일이 MANIFEST 와 다르면 거부(몰래 바꾼 파일).
    monkeypatch.setattr(criteria, "manifest_text", lambda: "다름\n")
    problems = jv.verify_problems(live=True, ledger_path=tmp_path / "l.csv", git=_git_ok(criteria.recorded_manifest()))
    assert len(problems) == 1 and "MANIFEST" in problems[0]


def test_verify_execute_refuses_with_every_reason_and_touches_nothing(tmp_path, vdir):
    ledger = tmp_path / "장부.csv"
    git = lambda *a: (0, " M x\n") if a[0] == "status" else (128, "")  # noqa: E731
    with pytest.raises(jv.RunnerError) as err:
        jv.execute(jv.RunArgs(mode="verify", batch="reproducibility", ledger=ledger, reports=["bc_strong"]), git=git)
    text = str(err.value)
    for part in ("작업 트리", "기준 버전", "태그", "--live"):
        assert part in text
    assert not ledger.exists() and not (vdir / "03_실행").exists()


def test_verify_cli_refuses_dev_only_paths(capsys):
    assert jv.main(["run", "reproducibility", "--mode", "verify", "--inputs", "x"]) == 2
    assert "dev 전용" in capsys.readouterr().err


# ── dev 읽기 경계 ──


@pytest.mark.parametrize("which", ["ledger", "inputs", "defects", "out"])
def test_dev_refuses_any_path_inside_validation_folder(tmp_path, vdir, which):
    args = _dev(tmp_path, batch="defects")
    setattr(args, which, vdir / "00_봉인")
    with pytest.raises(jv.RunnerError, match="dev 모드는 검증 폴더를"):
        jv.execute(args, client=object())


def test_dev_report_refuses_validation_folder(tmp_path, vdir):
    with pytest.raises(jv.RunnerError, match="dev 모드는 검증 폴더를"):
        jv.cmd_report("dev", ledger=vdir / "장부.csv", out=tmp_path / "o", api={})


def test_validation_root_defaults_next_to_repo(monkeypatch):
    monkeypatch.delenv("JUDGE_VALIDATION_DIR", raising=False)
    assert jv.validation_root() == (ROOT.parent / "JUDGE_ROUND" / "검증").resolve()


# ── 회차 계획(순수) ──


def test_plan_runs_defaults_are_the_reduced_v2():
    d = R.DEFAULTS
    assert (d["reproducibility"].items * d["reproducibility"].repeats, d["defects"].items * d["defects"].repeats,
            d["sham"].items * d["sham"].repeats) == (9, 6, 3)
    assert R.estimate_usd(18) == 2.7 and R.COST_PER_SCORING_USD == 0.15


def test_plan_runs_and_pending_skip_only_same_mode_and_fingerprint():
    keys = R.plan_runs("reproducibility", [("a", "base"), ("b", "base")], 2)
    assert len(keys) == 4 and keys[0] == R.RunKey("reproducibility", "a", "base", 1)
    row = {"stage": "validate_reproducibility", "report": "a", "variant": "base", "repeat": "1", "mode": "live",
           "manifest_sha256": "f" * 64, "result_file": "03_실행/R-0001.json"}
    assert R.pending(keys, [row], fingerprint="f" * 64, mode="live") == keys[1:]
    assert R.pending(keys, [row], fingerprint="e" * 64, mode="live") == keys  # 지문이 다르면(버전 올림) 다시
    assert R.pending(keys, [row], fingerprint="f" * 64, mode="dev") == keys
    assert R.pending(keys, [{**row, "result_file": ""}], fingerprint="f" * 64, mode="live") == keys  # 결과 없는 줄
    with pytest.raises(ValueError):
        R.plan_runs("reproducibility", [("a", "base"), ("a", "base")], 1)
    assert R.plan_runs("extraction", [("a", "base")], 3) == []


def test_shuffle_is_seeded():
    keys = R.plan_runs("reproducibility", [(t, "base") for t in "abcdef"], 3)
    assert R.shuffled(keys, 1) == R.shuffled(list(reversed(keys)), 1)
    assert R.shuffled(keys, 1) != R.shuffled(keys, 2)
    assert sorted(R.shuffled(keys, 1)) == keys


# ── dev 실행: 장부·결과·이어 하기 ──


def test_dev_reproducibility_writes_one_row_and_file_per_run_then_resume_skips(tmp_path, vdir):
    lines, echo = _quiet()
    summary = jv.execute(_dev(tmp_path), echo=echo)
    ledger = tmp_path / "work" / "장부.csv"
    rows = _rows(ledger)
    assert summary["ran"] == 2 and len(rows) == 2
    assert {(r["stage"], r["report"], r["variant"], r["mode"]) for r in rows} == {
        ("validate_reproducibility", "dev_binary", "base", "dev")}
    assert sorted(r["repeat"] for r in rows) == ["1", "2"]
    assert all(r["manifest_sha256"] == criteria.fingerprint() for r in rows)
    for r in rows:
        body = json.loads(Path(r["result_file"]).read_text(encoding="utf-8"))
        assert body["order_seed"] == 7 and body["gate"] in ("pass", "blocked") and body["run_id"] == r["run_id"]
    plan = json.loads((tmp_path / "work" / "runs" / "plan-reproducibility-7.json").read_text(encoding="utf-8"))
    assert plan["seed"] == 7 and len(plan["order"]) == 2
    assert lines[0].startswith("[dev] 검증 묶음 reproducibility: 계획 2회 · 끝남 0회(건너뜀) · 남음 2회 · 순서 시드 7")
    # 이어 하기: 같은 명령 → 끝난 회차는 건너뛰고 장부에 새 줄이 없다. 회차를 늘리면 늘린 만큼만.
    lines2, echo2 = _quiet()
    assert jv.execute(_dev(tmp_path), echo=echo2)["ran"] == 0
    assert "끝남 2회(건너뜀) · 남음 0회" in lines2[0] and len(_rows(ledger)) == 2
    assert jv.execute(_dev(tmp_path, repeats=3), echo=_quiet()[1])["ran"] == 1
    assert sorted(r["repeat"] for r in _rows(ledger)) == ["1", "2", "3"]


def test_start_without_yes_stops_before_any_run(tmp_path, vdir):
    lines, echo = _quiet()
    with pytest.raises(jv.RunnerError, match="--yes"):
        jv.execute(_dev(tmp_path, yes=False), echo=echo)
    assert "예상 2회" in lines[1] and "누계 $0.00·0회" in lines[1]
    assert _rows(tmp_path / "work" / "장부.csv") == []


def test_batch_run_cap_refuses_start(tmp_path, vdir):
    with pytest.raises(jv.RunnerError, match="묶음 회차 상한"):
        jv.execute(_dev(tmp_path, repeats=4, max_runs=3), echo=_quiet()[1])
    assert _rows(tmp_path / "work" / "장부.csv") == []


def test_round_cap_refuses_whole_batch(tmp_path, vdir, monkeypatch):
    monkeypatch.setenv("JUDGE_LIVE_MAX_RUNS", "1")
    with pytest.raises(jv.RunnerError, match="회차 상한을 넘는다"):
        jv.execute(_dev(tmp_path), echo=_quiet()[1])
    assert _rows(tmp_path / "work" / "장부.csv") == []


def test_batch_budget_stops_mid_batch_and_resume_finishes(tmp_path, vdir):
    with pytest.raises(jv.RunnerError, match="묶음 예산"):
        jv.execute(_dev(tmp_path, repeats=3, batch_budget_usd=1e-9), echo=_quiet()[1])
    ledger = tmp_path / "work" / "장부.csv"
    first = _rows(ledger)
    assert len(first) == 1 and float(first[0]["cost_usd"]) > 0  # 가짜 응답도 토큰 → 비용이 찍힌다
    assert jv.execute(_dev(tmp_path, repeats=3), echo=_quiet()[1])["ran"] == 2
    assert len(_rows(ledger)) == 3


def test_defects_batch_refuses_malformed_variant_before_any_run(tmp_path, vdir):
    import asyncio

    from app.judge.validation import variants as V

    base = asyncio.run(jv.dev_bases(jv.DEV_INPUTS))["dev_binary"]
    slot = "conclusion.narrative"
    text = V.read_slot(base.result, slot)
    first = V.sentences(text)[0]
    meta = {"id": "D01", "base": "dev_binary", "axis": "faithfulness", "kind": "숫자", "planted_by": "인용",  # 역할 틀림
            "slot": slot, "intent": "합성", "before": first, "after": first + " 덧"}
    V.write_variant(tmp_path / "결함", meta, base.result, V.replace_sentence(text, 0, meta["after"]))
    with pytest.raises(jv.RunnerError, match="결함본·sham 꼴이 틀려 시작하지 않는다"):
        jv.execute(_dev(tmp_path, batch="defects", defects=tmp_path / "결함", repeats=1), echo=_quiet()[1])
    assert not (tmp_path / "work" / "장부.csv").exists() and not (tmp_path / "work" / "runs").exists()


def test_extraction_batch_calls_nothing(tmp_path, vdir):
    lines, echo = _quiet()

    class Boom:
        def __getattr__(self, name):
            raise AssertionError("추출 묶음이 채점을 불렀다")

    assert jv.execute(_dev(tmp_path, batch="extraction"), client=Boom(), echo=echo)["ran"] == 0
    assert "채점을 따로 부르지 않는다" in lines[0]


# ── ⑥ 가짜 LLM 으로 처음부터 끝까지 한 바퀴(dev) — 개발 입력 3건 → 재현성 → 합성 결함 1·sham 1 → report(가짜 통계) ──

_AUDIT = {"on": False, "paths": []}


def _audit(event, args):
    if _AUDIT["on"] and event == "open" and args and isinstance(args[0], (str, bytes, Path)):
        _AUDIT["paths"].append(str(args[0]))


sys.addaudithook(_audit)


class FakeStats:
    """통계 함수 자리 — 받은 인자를 적고 그럴듯한 값을 돌려준다(배선만 본다)."""

    RETURNS = {
        "axis_scores": {"faithfulness": 1.0, "coverage": 0.5, "consistency": 0.0, "readability": 0.0},
        "reproducibility": {"gates_passed": True, "axes_passed": True}, "defect_detected": True,
        "detection_passed": {"passed": True}, "independence_ok": {"ok": True, "axes": {}}, "sham_ok": {"ok": True, "axes": {}},
        "agreement_rate": 0.9, "human_agreement_ok": True, "gold_from_csv": [{"id": "G1"}], "pred_from_extract": [{"id": "P1"}],
        "match_claims": {"pairs": [1]}, "recall_precision": {"recall": 1.0, "precision": 1.0, "n_pairs": 1, "n_gold": 1, "n_pred": 1},
        "runs_distribution": {"recall": {}}, "cohen_kappa": 0.5, "gwet_ac1": 0.6,
    }

    def __init__(self):
        self.calls: list[tuple] = []

    def install(self, monkeypatch):
        import types

        for mod in sorted(set(jv.STATS_API.values())):
            m = types.ModuleType(f"fake_{mod}")
            for name, owner in jv.STATS_API.items():
                if owner == mod:
                    setattr(m, name, self._fn(name))
            monkeypatch.setitem(sys.modules, f"app.judge.validation.{mod}", m)

    def _fn(self, name):
        def fn(*args, **kw):
            self.calls.append((name, args, kw))
            return self.RETURNS[name]
        return fn

    def named(self, name):
        return [(a, k) for n, a, k in self.calls if n == name]


def _plant_synthetic(defects: Path, bases):
    from app.judge.validation import variants as V

    b = bases["dev_binary"]
    slot = "conclusion.narrative"
    text = V.read_slot(b.result, slot)
    first = V.sentences(text)[0]
    after = first[:-1] + " (worse)."
    V.write_variant(defects, {"id": "D01", "base": "dev_binary", "axis": "readability", "kind": "영어 코드값 노출",
                              "planted_by": "인용", "slot": slot, "intent": "코드값 노출(합성)", "before": first, "after": after},
                    b.result, V.replace_sentence(text, 0, after))
    slot = "interpretation.confusion_analysis"
    text = V.read_slot(b.result, slot)
    gone = V.sentences(text)[-1]
    V.write_variant(defects, {"id": "D02", "base": "dev_binary", "axis": "coverage", "kind": "항목 지우기", "planted_by": "인용",
                              "slot": slot, "intent": "마지막 문장 지움(합성)", "before": gone, "after": "", "item_id": "분석-1"},
                    b.result, V.replace_sentence(text, len(V.sentences(text)) - 1, None))
    m = bases["dev_multiclass"]
    slot = "conclusion.risks"
    text = V.read_slot(m.result, slot)
    first = V.sentences(text)[0]
    V.write_variant(defects, {"id": "S01", "base": "dev_multiclass", "axis": "sham", "kind": "어순", "planted_by": "민서",
                              "slot": slot, "intent": "뜻 그대로(합성)", "before": first, "after": "즉, " + first},
                    m.result, V.replace_sentence(text, 0, "즉, " + first))


def _answers(folder: Path, base) -> None:
    """합성 정답지 — 개발 서술의 표시지(annotation 꼴)와 숫자가 든 첫 문장의 첫 숫자 낱말 하나."""
    from app.judge.validation import annotation as A

    rows = A.sentence_rows(A.pieces_from_sealed({"request": base.request, "view": {"result": base.result}}))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{base.tag}.sentences.csv").write_text(A.to_csv(A.SENTENCE_COLUMNS, rows), encoding="utf-8")
    no, text = next((r[0], r[5]) for r in rows if any(ch.isdigit() for ch in r[5]))
    tok = next(t for t in text.split() if any(ch.isdigit() for ch in t))
    (folder / f"{base.tag}.claims.csv").write_text(A.to_csv(A.CLAIM_COLUMNS, [(no, "number", tok, "")]), encoding="utf-8")


def _fill_sheets(folder: Path) -> dict:
    """합성 사람 채점 — 인용 님은 열쇠의 시스템 판정 그대로, 민서는 겹침 문항 하나를 뒤집고 인용 님 판정 하나를 비운다."""
    import csv as _csv

    key = {r["문항"]: r["시스템_문제"] for r in _rows(folder / "열쇠" / "열쇠.csv")}
    sheets = {who: _rows(folder / f"채점지_{who}.csv") for who in ("인용", "민서")}
    overlap = sorted({r["문항"] for r in sheets["인용"]} & {r["문항"] for r in sheets["민서"]})
    flipped, blanked = overlap[0], next(r["문항"] for r in sheets["인용"] if r["문항"] not in overlap)
    for who, rows in sheets.items():
        for r in rows:
            v = key[r["문항"]]
            if who == "민서" and r["문항"] == flipped:
                v = "아니오" if v == "예" else "예"
            if who == "인용" and r["문항"] == blanked:
                v = ""
            r["판정(예/아니오)"] = v
        with (folder / f"채점지_{who}.csv").open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return {"overlap": len(overlap), "total": sum(len(r) for r in sheets.values())}


def _full_loop(tmp_path) -> dict:
    """개발 입력 3건 × 2회 재현성 → 결함 2(표기·필수 내용)·sham 1 → 추출 → 채점지 → 합성 채점. 경로들을 돌려준다."""
    import asyncio

    work = tmp_path / "work"
    bases = asyncio.run(jv.dev_bases(jv.DEV_INPUTS))
    _plant_synthetic(work / "결함", bases)
    _answers(work / "정답지", bases["dev_binary"])
    common = dict(mode="dev", ledger=work / "장부.csv", out=work / "runs", yes=True, seed=11, defects=work / "결함")
    ran = [jv.execute(jv.RunArgs(batch=b, reports=[], repeats=2 if b == "reproducibility" else None, **common),
                      echo=_quiet()[1])["ran"] for b in ("reproducibility", "defects", "sham", "extraction")]
    jv.cmd_sheets("dev", ledger=work / "장부.csv", out=work / "사람", n=16, overlap=2, seed=5, defects=work / "결함",
                  echo=_quiet()[1])
    filled = _fill_sheets(work / "사람")
    return {"work": work, "ran": ran, **filled}


def test_one_full_dev_loop_with_fake_llm_never_touches_validation_folder(tmp_path, vdir, monkeypatch):
    """배선 — 통계 함수를 가짜로 꽂아 report 가 무엇을 어떤 꼴로 넘기는지 본다. 한 바퀴 동안 검증 폴더는 한 번도 열지 않는다."""
    fake = FakeStats()
    fake.install(monkeypatch)
    _AUDIT.update(on=True, paths=[])
    try:
        loop = _full_loop(tmp_path)
        work = loop["work"]
        result = jv.cmd_report("dev", ledger=work / "장부.csv", out=work / "결과", answers=work / "정답지", human=work / "사람",
                               echo=_quiet()[1])
    finally:
        _AUDIT["on"] = False
    opened = [Path(p).resolve() for p in _AUDIT["paths"]]
    assert any(p == (work / "장부.csv").resolve() for p in opened)  # 감시가 실제로 잡는다(빈 감시는 늘 통과)
    assert [p for p in opened if p == vdir or vdir in p.parents] == []

    assert loop["ran"] == [6, 2, 1, 0]
    rows = _rows(work / "장부.csv")
    assert len(rows) == 9 and all(r["mode"] == "dev" and r["result_sha256"] for r in rows)

    ((per, kw),) = fake.named("reproducibility")
    assert sorted(per[0]) == ["dev_binary", "dev_multiclass", "dev_multilabel"] and kw == {"expected_runs": 2}
    assert all([r["key"].repeat for r in v] == [1, 2] for v in per[0].values())
    det = {a[0]["key"].variant: (a[1], k) for a, k in fake.named("defect_detected")}
    assert det["D01"] == ("readability", {"sentence": V_after(work, "D01"), "item_id": None})
    assert det["D02"][0] == "coverage" and det["D02"][1]["item_id"] == "분석-1"
    ((by_defect,), _), = fake.named("detection_passed")
    assert by_defect == {"D01": {"axis": "readability", "hits": [True]}, "D02": {"axis": "coverage", "hits": [True]}}
    indep = fake.named("independence_ok")
    assert [(len(a[0]), len(a[1]), a[2]) for a, _ in indep] == [(2, 1, "readability"), (2, 1, "coverage")]
    ((sham_args, _),) = fake.named("sham_ok")
    assert len(sham_args[0]) == 2 and sham_args[1][0]["key"].variant == "S01"
    assert len(fake.named("gold_from_csv")) == 1 and len(fake.named("pred_from_extract")) == 2  # dev_binary 회차 2개
    assert len(fake.named("match_claims")) == 2 and len(fake.named("runs_distribution")) == 1
    assert fake.named("human_agreement_ok") and fake.named("cohen_kappa") and fake.named("gwet_ac1")
    assert result["detection"] == {"passed": True} and result["sham_passed"] and result["independence_passed"]
    assert result["extraction"]["pooled"] == {"recall": 1.0, "precision": 1.0, "n_pairs": 2, "n_gold": 2, "n_pred": 2}
    saved = json.loads((work / "결과" / "검증결과.json").read_text(encoding="utf-8"))
    assert saved["fingerprint"] == criteria.fingerprint() and saved["runs"] == 9


def V_after(work: Path, vid: str) -> str:
    return json.loads((work / "결함" / vid / "meta.json").read_text(encoding="utf-8"))["after"]


def test_one_full_dev_loop_with_real_stats_gives_real_numbers(tmp_path, vdir):
    """실제 통계 모듈로 한 바퀴 — report 가 7.6 의 숫자를 낸다(가짜 LLM 이라 값의 좋고 나쁨이 아니라 꼴과 셈을 본다)."""
    loop = _full_loop(tmp_path)
    work = loop["work"]
    r = jv.cmd_report("dev", ledger=work / "장부.csv", out=work / "결과", answers=work / "정답지", human=work / "사람",
                      echo=_quiet()[1])
    repro = r["reproducibility"]
    assert set(repro["axes"]) == {"faithfulness", "coverage", "consistency", "readability"} and repro["n_runs"] == 2
    for axis, a in repro["axes"].items():  # 가짜 심판은 회차마다 같은 답 — ICC 대신 분산 0 표시(Q10)이거나 점수 없음
        assert a["zero_variance"] or a["dropped"], (axis, a)
        assert a["passed"] == (a["zero_variance"] and not a["dropped"])
    assert repro["gates_passed"] is True and r["gates_passed"] is True
    assert all(g["gates"] == ["pass", "pass"] for g in repro["gates"].values())
    assert r["base_gate_pass"] == {"dev_binary": True, "dev_multiclass": True, "dev_multilabel": True}
    assert r["defects"]["D01"]["hits"] == [True]  # 'worse' 를 표기 코드 규칙이 그 문장에서 잡는다
    assert isinstance(r["defects"]["D02"]["hits"][0], bool)
    assert r["detection"]["defects"]["D01"] == {"axis": "readability", "rate": 1.0, "ok": True}
    assert set(r["detection"]) >= {"gate_axes_ok", "others_ok", "passed"}
    ind = r["defects"]["D01"]["independence"]
    assert set(ind["axes"]) == {"faithfulness", "coverage", "consistency"} and isinstance(ind["ok"], bool)
    sham = r["sham"]["S01"]["sham"]
    assert set(sham["axes"]) == {"faithfulness", "coverage", "consistency", "readability"} and isinstance(r["sham_passed"], bool)
    ext = r["extraction"]
    assert ext["pooled"]["n_gold"] == 2 and ext["pooled"]["n_pred"] >= 2  # 정답 1개 × 회차 2
    assert 0.0 <= ext["pooled"]["recall"] <= 1.0 and 0.0 <= ext["pooled"]["precision"] <= 1.0
    assert ext["reports"]["dev_binary"]["distribution"]["recall"]["n"] == 2
    agr = r["agreement"]
    # 짝 셈: 판정마다 시스템 짝 하나(빈칸 하나 뺌), 사람끼리는 겹침 문항마다 하나. 민서가 겹침 하나를 뒤집었다.
    n, k = loop["total"] - 1, loop["overlap"]
    assert (agr["blank"], agr["system_human"]["n"], agr["human_human"]["n"]) == (1, n, k)
    assert agr["system_human"]["rate"] == pytest.approx((n - 1) / n)
    assert agr["human_human"]["rate"] == pytest.approx((k - 1) / k)
    assert agr["passed"] is ((n - 1) / n >= (k - 1) / k - 0.05)
    assert agr["system_human"]["kappa"] is not None or agr["system_human"]["ac1"] is not None


def test_report_refuses_tampered_result_file(tmp_path, vdir):
    jv.execute(_dev(tmp_path, repeats=1), echo=_quiet()[1])
    row = _rows(tmp_path / "work" / "장부.csv")[0]
    Path(row["result_file"]).write_text("{}", encoding="utf-8")
    with pytest.raises(jv.RunnerError, match="결과 원본 해시가 장부와 다르다"):
        jv.cmd_report("dev", ledger=tmp_path / "work" / "장부.csv", out=tmp_path / "o", api={})


def test_stats_api_names_exist_in_real_modules():
    api = jv.load_stats_api()
    assert set(api) == set(jv.STATS_API) and all(callable(f) for f in api.values())
    assert set(jv.STATS_API.values()) == {"stats", "extraction", "agreement"}
