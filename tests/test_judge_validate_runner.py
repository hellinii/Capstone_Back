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


def test_extraction_batch_calls_nothing(tmp_path, vdir):
    lines, echo = _quiet()

    class Boom:
        def __getattr__(self, name):
            raise AssertionError("추출 묶음이 채점을 불렀다")

    assert jv.execute(_dev(tmp_path, batch="extraction"), client=Boom(), echo=echo)["ran"] == 0
    assert "채점을 따로 부르지 않는다" in lines[0]
