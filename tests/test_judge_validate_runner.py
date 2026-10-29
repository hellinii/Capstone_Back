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


def test_plan_runs_defaults_are_the_v1_decision():
    """V1 결정 1·2(2026-10-29): 재현성 3건 × 2회, 결함 3개 × 1회, sham 1 → 채점 10회 ≈ $1.5. 성적서 기본 3건."""
    d = R.DEFAULTS
    assert [(d[b].items, d[b].repeats) for b in ("reproducibility", "defects", "sham")] == [(3, 2), (3, 1), (1, 1)]
    assert R.plan_total_runs() == 10 and R.estimate_usd(10) == 1.5 and R.COST_PER_SCORING_USD == 0.15
    assert R.DEFAULT_REPORTS == ("bc_strong", "digits_weak", "yeast_strong")
    assert R.plan_summary() == ("V2 전체 계획: 재현성 3건 × 2회 + 결함 3개 × 1회 + sham 1 = 채점 10회 · 약 $1.50(1회 $0.15)")


def test_verify_reproducibility_defaults_to_the_three_decided_reports(tmp_path, monkeypatch):
    """verify 재현성에서 --reports 를 안 주면 V1 결정 2 의 3건을 봉인에서 읽는다(봉인은 합성 — 읽으려는 태그만 본다)."""
    asked = []

    def fake_sealed(folder, tags):
        asked.append(list(tags))
        raise jv.RunnerError("봉인 읽기 멈춤(시험)")

    monkeypatch.setattr(jv, "sealed_bases", fake_sealed)
    monkeypatch.setattr(jv, "verify_problems", lambda **kw: [])
    with pytest.raises(jv.RunnerError, match="봉인 읽기 멈춤"):
        jv.execute(jv.RunArgs(mode="verify", batch="reproducibility", ledger=tmp_path / "장부.csv", live=True, yes=True))
    assert asked == [["bc_strong", "digits_weak", "yeast_strong"]]
    with pytest.raises(jv.RunnerError, match="봉인 읽기 멈춤"):
        jv.execute(jv.RunArgs(mode="verify", batch="reproducibility", ledger=tmp_path / "장부.csv", live=True, yes=True,
                              reports=["bc_strong"]))
    assert asked[-1] == ["bc_strong"]


def test_plan_runs_and_pending_skip_only_same_mode_and_fingerprint():
    keys = R.plan_runs("reproducibility", [("a", "base"), ("b", "base")], 2)
    assert len(keys) == 4 and keys[0] == R.RunKey("reproducibility", "a", "base", 1)
    row = {"stage": "validate_reproducibility", "report": "a", "variant": "base", "repeat": "1", "mode": "live",
           "manifest_sha256": "f" * 64, "result_file": "03_실행/R-0001.json", "gate": "blocked"}
    assert R.pending(keys, [row], fingerprint="f" * 64, mode="live") == keys[1:]
    assert R.pending(keys, [{**row, "gate": ""}], fingerprint="f" * 64, mode="live") == keys  # 채점 실패 줄(P0-1)
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
    assert lines[0] == "[dev] " + R.plan_summary()  # V2 전체 계획 안내가 먼저
    assert lines[1].startswith("[dev] 검증 묶음 reproducibility: 계획 2회 · 끝남 0회(건너뜀) · 남음 2회 · 순서 시드 7")
    # 이어 하기: 같은 명령 → 끝난 회차는 건너뛰고 장부에 새 줄이 없다. 회차를 늘리면 늘린 만큼만.
    lines2, echo2 = _quiet()
    assert jv.execute(_dev(tmp_path), echo=echo2)["ran"] == 0
    assert "끝남 2회(건너뜀) · 남음 0회" in lines2[1] and len(_rows(ledger)) == 2
    assert jv.execute(_dev(tmp_path, repeats=3), echo=_quiet()[1])["ran"] == 1
    assert sorted(r["repeat"] for r in _rows(ledger)) == ["1", "2", "3"]


def test_start_without_yes_stops_before_any_run(tmp_path, vdir):
    lines, echo = _quiet()
    with pytest.raises(jv.RunnerError, match="--yes"):
        jv.execute(_dev(tmp_path, yes=False), echo=echo)
    assert "예상 2회" in lines[2] and "누계 $0.00·0회" in lines[2]
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
        "missed_rows": [(1, 3, "7.1", "degree", "몰려 있다", "", "")],
        "confirmed_additions": [{"문장번호": "3", "종류": "degree", "인용 구절": "몰려 있다", "메모": "보탬", "added": True}],
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
    ml = bases["dev_multilabel"]  # 사실 일치 결함(민서가 심음 — 7.2 역할) — V1 결정 3 의 결함 셋을 맞춘다
    slot = "conclusion.narrative"
    text = V.read_slot(ml.result, slot)
    first = V.sentences(text)[0]
    after = first[:-1] + " 양호하다."
    V.write_variant(defects, {"id": "D03", "base": "dev_multilabel", "axis": "faithfulness", "kind": "정도 표현 극성 뒤집기",
                              "planted_by": "민서", "slot": slot, "intent": "미달인데 양호(합성)", "before": first, "after": after},
                    ml.result, V.replace_sentence(text, 0, after))
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


def _read_keyed(path: Path) -> tuple[str, list[dict]]:
    """머리 줄(열쇠 해시) + 표."""
    head, _, body = path.read_text(encoding="utf-8").partition("\n")
    assert head.startswith("# 열쇠 sha256: ")
    import io

    return head, list(csv.DictReader(io.StringIO(body)))


def _write_keyed(path: Path, head: str, rows: list[dict]) -> None:
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    path.write_text(head + "\n" + buf.getvalue(), encoding="utf-8")


def _fill_sheets(folder: Path) -> dict:
    """합성 사람 채점 — 인용 님은 열쇠의 시스템 판정 그대로, 민서는 겹침 문항 하나를 뒤집고 인용 님 판정 하나를 비운다."""
    key = {r["문항"]: r["시스템_문제"] for r in _rows(folder / "열쇠" / "열쇠.csv")}
    read = {who: _read_keyed(folder / f"채점지_{who}.csv") for who in ("인용", "민서")}
    sheets = {who: rows for who, (_, rows) in read.items()}
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
        _write_keyed(folder / f"채점지_{who}.csv", read[who][0], rows)
    return {"overlap": len(overlap), "total": sum(len(r) for r in sheets.values())}


PLAN2 = {"reproducibility": R.BatchDefault(items=3, repeats=2)}  # 한 바퀴의 계획: 개발 입력 3건 × 2회(= V1 결정 기본값)


def _full_loop(tmp_path) -> dict:
    """새 기본 회차(V1 결정 1 — 회차·바탕 확인 인자를 주지 않는다): 개발 입력 3건 × 2회 재현성 → 결함 3(표기·필수 내용·사실
    일치) × 1회·sham 1 → 추출 → 채점지 → 합성 채점. 경로들을 돌려준다."""
    import asyncio

    work = tmp_path / "work"
    bases = asyncio.run(jv.dev_bases(jv.DEV_INPUTS))
    _plant_synthetic(work / "결함", bases)
    _answers(work / "정답지", bases["dev_binary"])
    common = dict(mode="dev", ledger=work / "장부.csv", out=work / "runs", yes=True, seed=11, defects=work / "결함")
    ran = [jv.execute(jv.RunArgs(batch=b, reports=[], **common), echo=_quiet()[1])["ran"]
           for b in ("reproducibility", "defects", "sham", "extraction")]
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
        judged = "번호,문장번호,칸,종류,인용 구절,진짜 주장인가(예/아니오),메모\n1,3,7.1,degree,몰려 있다,예,\n"
        (work / "정답지" / "dev_binary.missed.csv").write_text(judged, encoding="utf-8")
        result = jv.cmd_report("dev", ledger=work / "장부.csv", out=work / "결과", answers=work / "정답지", human=work / "사람",
                               echo=_quiet()[1], plan=PLAN2)
    finally:
        _AUDIT["on"] = False
    opened = [Path(p).resolve() for p in _AUDIT["paths"]]
    assert any(p == (work / "장부.csv").resolve() for p in opened)  # 감시가 실제로 잡는다(빈 감시는 늘 통과)
    assert [p for p in opened if p == vdir or vdir in p.parents] == []

    assert loop["ran"] == [6, 3, 1, 0]  # V1 결정 1 의 기본 회차 그대로(재현성 3×2, 결함 3×1, sham 1 — 채점 10회)
    rows = _rows(work / "장부.csv")
    assert len(rows) == 10 and all(r["mode"] == "dev" and r["result_sha256"] for r in rows)

    ((per, kw),) = fake.named("reproducibility")
    assert sorted(per[0]) == ["dev_binary", "dev_multiclass", "dev_multilabel"] and kw == {"expected_runs": 2}
    assert all([r["key"].repeat for r in v] == [1, 2] for v in per[0].values())
    det = {a[0]["key"].variant: (a[1], k) for a, k in fake.named("defect_detected")}
    assert det["D01"] == ("readability", {"sentence": V_after(work, "D01"), "item_id": None})
    assert det["D02"][0] == "coverage" and det["D02"][1]["item_id"] == "분석-1"
    assert det["D03"] == ("faithfulness", {"sentence": V_after(work, "D03"), "item_id": None})
    ((by_defect,), _), = fake.named("detection_passed")
    assert by_defect == {"D01": {"axis": "readability", "hits": [True]}, "D02": {"axis": "coverage", "hits": [True]},
                         "D03": {"axis": "faithfulness", "hits": [True]}}
    indep = fake.named("independence_ok")
    assert [(len(a[0]), len(a[1]), a[2]) for a, _ in indep] == [(2, 1, "readability"), (2, 1, "coverage"),
                                                                  (2, 1, "faithfulness")]
    ((sham_args, _),) = fake.named("sham_ok")
    assert len(sham_args[0]) == 2 and sham_args[1][0]["key"].variant == "S01"
    # dev_binary 회차 2개. 놓친 주장 "예" 줄이 있어(가짜 confirmed_additions) 보탠 정답지로 한 번 더 — V1 결정 10④
    golds = fake.named("gold_from_csv")
    assert len(golds) == 2 and len(fake.named("pred_from_extract")) == 2
    assert golds[1][0][0] == [*golds[0][0][0], *FakeStats.RETURNS["confirmed_additions"]]  # 원 정답지 줄 + 보탠 줄
    assert len(fake.named("match_claims")) == 4 and len(fake.named("runs_distribution")) == 2
    ((hargs, hkw),) = fake.named("human_agreement_ok")
    assert hkw == {"overlap_n": loop["overlap"]}  # 여유 = 한 문항(겹침 문항 수로 — V1 결정 7)
    assert fake.named("cohen_kappa") and fake.named("gwet_ac1")
    assert result["detection"] == {"passed": True} and result["sham_passed"] and result["independence_passed"]
    assert result["extraction"]["pooled"] == {"recall": 1.0, "precision": 1.0, "n_pairs": 2, "n_gold": 2, "n_pred": 2}
    assert result["extraction"]["pooled_with_additions"] == {"recall": 1.0, "precision": 1.0, "n_pairs": 2, "n_gold": 2,
                                                             "n_pred": 2, "added": 1}
    assert result["extraction"]["reports"]["dev_binary"]["with_additions"]["added"] == 1
    # 놓친 주장(Q12②) — 회차마다 match 결과를 missed_rows 에, 정답지 폴더의 판정표를 confirmed_additions 에 넘긴다
    assert [a for a, _ in fake.named("missed_rows")] == [({"pairs": [1]},), ({"pairs": [1]},)]
    ((judged_rows,), _), = fake.named("confirmed_additions")
    assert [r["인용 구절"] for r in judged_rows] == ["몰려 있다"]
    missed = result["missed_claims"]["dev_binary"]
    assert missed["judged"] and missed["rows"] == [(1, 3, "7.1", "degree", "몰려 있다", "", "")]  # 두 회차 같은 줄은 하나
    assert missed["additions"] == FakeStats.RETURNS["confirmed_additions"]
    assert _rows(work / "결과" / "놓친주장_dev_binary.csv")[0]["인용 구절"] == "몰려 있다"
    assert _rows(work / "결과" / "보탤주장_dev_binary.csv") == [{"문장번호": "3", "종류": "degree", "인용 구절": "몰려 있다",
                                                              "메모": "보탬"}]
    saved = json.loads((work / "결과" / "검증결과.json").read_text(encoding="utf-8"))
    assert saved["fingerprint"] == criteria.fingerprint() and saved["runs"] == 10
    assert "pooled" in saved["extraction"] and "pooled_with_additions" in saved["extraction"]  # 보태기 전·후 둘 다


def V_after(work: Path, vid: str) -> str:
    return json.loads((work / "결함" / vid / "meta.json").read_text(encoding="utf-8"))["after"]


def test_one_full_dev_loop_with_real_stats_gives_real_numbers(tmp_path, vdir):
    """실제 통계 모듈로 한 바퀴 — report 가 7.6 의 숫자를 낸다(가짜 LLM 이라 값의 좋고 나쁨이 아니라 꼴과 셈을 본다)."""
    loop = _full_loop(tmp_path)
    work = loop["work"]
    r = jv.cmd_report("dev", ledger=work / "장부.csv", out=work / "결과", answers=work / "정답지", human=work / "사람",
                      echo=_quiet()[1], plan=PLAN2)
    repro = r["reproducibility"]
    assert set(repro["axes"]) == {"faithfulness", "coverage", "consistency", "readability"} and repro["n_runs"] == 2
    for axis, a in repro["axes"].items():  # 가짜 심판은 회차마다 같은 답 — 분산 0 표시(Q10)이거나 해당 없음(분모 0)
        assert a["zero_variance"] or a["dropped"] or a["not_applicable"], (axis, a)
        if a["not_applicable"] and a["n_targets"] < 2:
            assert a["passed"] is None  # 해당 없음을 빼고 남은 대상이 2개 미만 — 계산 불가(V1 결정 5)
        else:
            assert a["passed"] == (a["zero_variance"] and not a["dropped"])
    assert repro["gates_passed"] is True and r["gates_passed"] is True
    assert all(g["gates"] == ["pass", "pass"] for g in repro["gates"].values())
    assert r["base_gate_pass"] == {"dev_binary": True, "dev_multiclass": True, "dev_multilabel": True}
    assert r["defects"]["D01"]["hits"] == [True]  # 'worse' 를 표기 코드 규칙이 그 문장에서 잡는다
    assert isinstance(r["defects"]["D02"]["hits"][0], bool)
    assert r["detection"]["defects"]["D01"] == {"axis": "readability", "rate": 1.0, "ok": True}
    assert r["detection"]["rule"] == "all" and isinstance(r["detection"]["passed"], bool)  # V1 결정 4 — 결함 모두
    assert r["detection"]["passed"] == all(all(d["hits"]) for d in r["defects"].values())
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
    assert agr["passed"] is ((n - 1) / n >= (k - 1) / k - 1 / k)  # 여유 = 한 문항(V1 결정 7)
    assert agr["system_human"]["kappa"] is not None or agr["system_human"]["ac1"] is not None


def test_blind_reading_bundle_and_effectiveness(tmp_path, vdir):
    """8절(P1) — 결함본은 심지 않은 사람에게만, 한 사람에게 한 성적서의 한 판본만. 사람이 찾은 수·헛짚음을 효용성 표로 센다."""
    loop = _full_loop(tmp_path)
    work = loop["work"]
    lines: list[str] = []
    got = jv.cmd_reading("dev", out=work / "사람" / "읽기", seed=4, defects=work / "결함", echo=lines.append)
    # 바탕 성적서 3건(dev_binary·dev_multiclass·dev_multilabel) — 사람마다 3문서. D01·D02(인용 님이 심음, dev_binary)는 민서만
    # 읽을 수 있고 민서는 dev_binary 를 한 판본만 받으므로 둘 중 하나는 자리가 없다. S01(민서가 심음, dev_multiclass)과
    # D03(민서가 심음, dev_multilabel)은 인용 님에게.
    assert got["docs"] == {"인용": 3, "민서": 3} and "시드 4" in lines[0]
    assert len(got["unplaced"]) == 1 and got["unplaced"][0] in ("D01", "D02") and "자리가 없는 결함본" in lines[1]
    key = _rows(work / "사람" / "읽기" / "열쇠" / "읽기_열쇠.csv")
    placed = ({"D01", "D02"} - set(got["unplaced"])).pop()
    assert {(k["읽는_사람"], k["성적서"], k["변형"]) for k in key} == {
        ("민서", "dev_binary", placed), ("민서", "dev_multiclass", "base"), ("민서", "dev_multilabel", "base"),
        ("인용", "dev_binary", "base"), ("인용", "dev_multiclass", "S01"), ("인용", "dev_multilabel", "D03")}
    docs = sorted(p.name for p in (work / "사람" / "읽기" / "민서" / "문서").iterdir())
    assert len(docs) == 3 and not any("D0" in d or "base" in d for d in docs)
    sheet = work / "사람" / "읽기" / "민서" / "읽기표.csv"
    head, rows = _read_keyed(sheet)
    doc_of = {k["변형"]: k["문서"] for k in key if k["읽는_사람"] == "민서"}
    for r in rows:
        if r["문서"] == doc_of[placed]:  # 결함 문서지만 엉뚱한 곳을 짚음 → 찾은 것 아님
            r["문제 문장(글자 그대로)"] = "엉뚱한 곳을 짚은 글"
        if r["문서"] == doc_of["base"]:
            r["문제 문장(글자 그대로)"] = "괜찮은 문장을 잘못 짚음"
    _write_keyed(sheet, head, rows)
    r = jv.cmd_report("dev", ledger=work / "장부.csv", out=work / "결과", human=work / "사람", echo=_quiet()[1], plan=PLAN2)
    eff = r["effectiveness"]
    assert (eff["defects"], eff["human_read"], eff["human_found"], eff["human_false_alarms"]) == (3, 2, 0, 1)
    assert eff["system_found"] == sum(any(d["hits"]) for d in r["defects"].values()) >= 1
    for row in rows:  # 이번엔 심은 문장을 그대로 옮김 → 찾음
        if row["문서"] == doc_of[placed]:
            row["문제 문장(글자 그대로)"] = V_after(work, placed) or "분석-1"
            row["메모"] = "분석-1"
    _write_keyed(sheet, head, rows)
    r = jv.cmd_report("dev", ledger=work / "장부.csv", out=work / "결과2", human=work / "사람", echo=_quiet()[1], plan=PLAN2)
    assert r["effectiveness"]["human_found"] == 1


def test_report_uses_the_api_it_is_given(tmp_path, vdir):
    # 넘긴 api 를 쓴다(모듈을 다시 읽지 않는다) — 시험·합칠 때 가짜를 꽂는 길.
    jv.execute(_dev(tmp_path, repeats=2), echo=_quiet()[1])
    fake = FakeStats()
    api = {name: fake._fn(name) for name in jv.STATS_API}
    r = jv.cmd_report("dev", ledger=tmp_path / "work" / "장부.csv", out=tmp_path / "o", api=api, echo=_quiet()[1],
                      plan={"reproducibility": R.BatchDefault(items=1, repeats=2)})
    assert fake.named("reproducibility") and r["reproducibility"] == FakeStats.RETURNS["reproducibility"]


def test_report_refuses_tampered_result_file(tmp_path, vdir):
    jv.execute(_dev(tmp_path, repeats=1), echo=_quiet()[1])
    row = _rows(tmp_path / "work" / "장부.csv")[0]
    Path(row["result_file"]).write_text("{}", encoding="utf-8")
    with pytest.raises(jv.RunnerError, match="결과 원본 해시가 장부와 다르다"):
        jv.cmd_report("dev", ledger=tmp_path / "work" / "장부.csv", out=tmp_path / "o", api={})


def _missed_runs(claims_per_run: list[list[dict]]) -> list[dict]:
    return [{"key": R.RunKey("reproducibility", "T1", "base", i), "gate": "pass", "record": {"extract": {"claims": cs}}}
            for i, cs in enumerate(claims_per_run, 1)]


_SENT = [dict(zip(("문장번호", "조각", "칸", "판4자리", "출처", "문장"), r)) for r in (
    ("1", "p71", "7.1", "5.2", "ai", "정확도는 0.912이다."),
    ("2", "p71", "7.1", "5.2", "ai", "오류는 클래스 3에 몰려 있다."),
)]


def _claim(kind, quote, number_text=""):
    return {"kind": kind, "quote": quote, "piece_id": "p71", "slot": "7.1", "number_text": number_text, "round": 0}


def test_report_missed_claims_with_real_extraction(tmp_path):
    """합성 — 실제 extraction 함수로: 시스템만 찾은 주장을 회차를 합쳐 판정표로, 판정한 "예" 줄만 보탤 목록으로."""
    api = jv.load_stats_api()
    api["reproducibility"] = lambda base, expected_runs=None: {"gates_passed": True}
    gold_rows = [{"문장번호": "1", "종류": "number", "인용 구절": "0.912", "메모": ""}]
    runs = _missed_runs([
        [_claim("number", "0.912", "0.912"), _claim("degree", "몰려 있다")],
        [_claim("number", "0.912", "0.912"), _claim("degree", "몰려 있다"), _claim("name", "여기 없는 구절")],
    ])
    r = jv.report(runs, api, human_claims={"T1": (gold_rows, _SENT)})
    m = r["missed_claims"]["T1"]
    assert m["rows"] == [(1, 2, "7.1", "degree", "몰려 있다", "", ""), (2, "", "7.1", "name", "여기 없는 구절", "", "")]
    assert m["judged"] is False and m["additions"] == []
    assert r["extraction"]["pooled"]["n_pairs"] == 2  # 짝짓기는 그대로

    judged = [dict(zip(jv.importlib.import_module("app.judge.validation.extraction").MISSED_COLUMNS, row)) for row in m["rows"]]
    judged[0]["진짜 주장인가(예/아니오)"] = "예"
    judged[1]["진짜 주장인가(예/아니오)"] = "아니오"
    r = jv.report(runs, api, human_claims={"T1": (gold_rows, _SENT)}, human_missed={"T1": judged})
    added = r["missed_claims"]["T1"]["additions"]
    assert [(a["문장번호"], a["인용 구절"], a["added"]) for a in added] == [(2, "몰려 있다", True)]
    # V1 결정 10④ — 보탠 정답지로 다시 잰 값(뒤)과 원 정답지 값(앞)이 둘 다. 정밀도가 오른다(놓친 주장이 진짜였다).
    ext = r["extraction"]
    assert ext["pooled"] == {"recall": 1.0, "precision": 2 / 5, "n_pairs": 2, "n_gold": 2, "n_pred": 5}
    assert ext["pooled_with_additions"] == {"recall": 1.0, "precision": 4 / 5, "n_pairs": 4, "n_gold": 4, "n_pred": 5, "added": 1}
    after = ext["reports"]["T1"]["with_additions"]
    assert after["added"] == 1 and [m["n_gold"] for m in after["runs"]] == [2, 2]
    assert [m["n_gold"] for m in ext["reports"]["T1"]["runs"]] == [1, 1]  # 앞 값은 원 정답지 그대로
    assert gold_rows == [{"문장번호": "1", "종류": "number", "인용 구절": "0.912", "메모": ""}]  # 원 정답지 줄은 고치지 않는다
    assert after["distribution"]["precision"]["max"] == 1.0
    paths = jv.write_missed(tmp_path, r["missed_claims"])
    assert [p.name for p in paths] == ["놓친주장_T1.csv", "보탤주장_T1.csv"]
    assert [row["인용 구절"] for row in _rows(tmp_path / "보탤주장_T1.csv")] == ["몰려 있다"]

    judged[1]["진짜 주장인가(예/아니오)"] = "예"  # 문장번호 없는 줄을 "예" — 실행기는 멈춤 사유로
    with pytest.raises(jv.RunnerError, match="T1 놓친 주장 판정표"):
        jv.report(runs, api, human_claims={"T1": (gold_rows, _SENT)}, human_missed={"T1": judged})


def test_report_detection_needs_every_planned_defect():
    """V1 결정 4 는 '결함 3개 모두' — 계획보다 적게 돈 결함 묶음은 통과로 세지 않는다(계산 불가). 계획만큼이면 통계 판정 그대로."""
    api = jv.load_stats_api()
    base = [_run("reproducibility", t, "base", i) for t in ("a", "b", "c") for i in (1, 2)]
    for r in base:
        r["scorecard"] = {"faithfulness": {"checked": 1, "matched": 1}, "coverage": {"applicable": 1, "described": 1},
                          "consistency": {"contradictions": 0}, "readability": {"violations": 0}}
    hit = {"faithfulness": {"verdicts": [{"verdict": "violation", "quote": "심은 문장"}]}}

    def defect(vid, tag):
        r = _run("defects", tag, vid, 1, meta={"axis": "faithfulness", "after": "심은 문장"})
        r["record"], r["scorecard"] = hit, base[0]["scorecard"]
        return r

    two = jv.report([*base, defect("D01", "a"), defect("D02", "b")], api)
    assert two["detection"]["passed"] is None and "덜 돎" in two["detection"]["note"]
    assert two["verdicts"]["detection"] == "계산 불가"
    three = jv.report([*base, defect("D01", "a"), defect("D02", "b"), defect("D03", "c")], api)
    assert three["detection"]["passed"] is True and three["verdicts"]["detection"] == "통과"
    miss = defect("D03", "c")
    miss["record"] = {"faithfulness": {"verdicts": []}}
    assert jv.report([*base, defect("D01", "a"), defect("D02", "b"), miss], api)["verdicts"]["detection"] == "미달"


def test_stats_api_names_exist_in_real_modules():
    api = jv.load_stats_api()
    assert set(api) == set(jv.STATS_API) and all(callable(f) for f in api.values())
    assert set(jv.STATS_API.values()) == {"stats", "extraction", "agreement"}


# ── J10 검토 반영 ① 이어 하기는 판정이 난 회차만 끝난 것으로(P0-1) ──


def test_resume_reruns_failed_scoring_and_keeps_failed_cost(tmp_path, vdir):
    """채점이 실패한 회차(gate None·failure, 결과 파일·장부 줄은 써짐)는 끝난 것이 아니다 — 같은 명령으로 다시 돈다.
    실패 줄은 장부에 그대로 남고(비용 누계에 든다), report 는 다시 돈 성공 줄만 본다."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from dev_fake_llm_server import FakeOpenAI

    args = _dev(tmp_path, reports=[], repeats=None)  # 기본 회차(V1 결정 1 — 3건 × 2회)
    first = jv.execute(args, client=FakeOpenAI("error"), echo=_quiet()[1])
    assert first["ran"] == 6 and set(first["gates"]) == {None}
    ledger = tmp_path / "work" / "장부.csv"
    failed = _rows(ledger)
    assert len(failed) == 6 and all(r["gate"] == "" and r["error"] and r["result_file"] for r in failed)
    second = jv.execute(args, client=FakeOpenAI("pass"), echo=_quiet()[1])
    assert second["ran"] == 6 and set(second["gates"]) <= {"pass", "blocked"}
    rows = _rows(ledger)
    assert len(rows) == 12 and rows[:6] == failed  # 실패 줄은 지우지 않는다(비용 누계)
    assert jv.execute(args, client=FakeOpenAI("error"), echo=_quiet()[1])["ran"] == 0  # 이제는 끝났다
    runs = jv.collect_runs(rows, fingerprint=criteria.fingerprint(), mode="dev", root=None)
    assert len(runs) == 6 and all(r["gate"] in ("pass", "blocked") for r in runs)


def test_collect_runs_keeps_last_success_over_later_failure(tmp_path, vdir):
    sys.path.insert(0, str(ROOT / "scripts"))
    from dev_fake_llm_server import FakeOpenAI

    jv.execute(_dev(tmp_path, repeats=1), client=FakeOpenAI("pass"), echo=_quiet()[1])
    ok = _rows(tmp_path / "work" / "장부.csv")[0]
    late_fail = {**ok, "run_id": "X", "gate": "", "error": "transport", "result_file": "없는파일.json"}
    runs = jv.collect_runs([ok, late_fail], fingerprint=criteria.fingerprint(), mode="dev", root=None)
    assert [r["run_id"] for r in runs] == [ok["run_id"]]
    assert R.pending(R.plan_runs("reproducibility", [("dev_binary", "base")], 1), [late_fail],
                     fingerprint=criteria.fingerprint(), mode="dev") != []


# ── J10 ② 계획 회차로 맞추기·덜 돎·계산 불가(P1-1), 결함·sham 전 바탕 확인(P1-2) ──


def _run(batch, report, variant, repeat, gate="pass", meta=None):
    return {"key": R.RunKey(batch, report, variant, repeat), "gate": gate, "run_id": f"{report}-{variant}-{repeat}",
            "variant_meta": meta or {}, "scorecard": {}, "record": {}}


def test_report_expected_runs_comes_from_plan_not_observed():
    fake = FakeStats()
    api = {name: fake._fn(name) for name in jv.STATS_API}
    runs = [_run("reproducibility", t, "base", i) for t in ("a", "b", "c") for i in (1, 2)]
    full = jv.report(runs, api, plan={"reproducibility": R.BatchDefault(3, 2)})
    assert fake.named("reproducibility")[0][1] == {"expected_runs": 2} and full["gates_passed"] is True
    fake.calls.clear()
    short = jv.report([r for r in runs if r["key"].repeat == 1], api)  # 기본 계획 2회(V1 결정 1) — 1회만 돌았다
    assert fake.named("reproducibility") == [] and short["reproducibility"]["verdict"] == "계산 불가"
    assert short["gates_passed"] is None and short["verdicts"]["gates_passed"] == "계산 불가"
    assert "덜 돎" in short["reproducibility"]["note"]


def test_report_partial_mixed_batch_is_uncomputable_with_real_stats():
    runs = [_run("reproducibility", "a", "base", i) for i in (1, 2, 3)] + [_run("reproducibility", "b", "base", 1)]
    r = jv.report(runs, jv.load_stats_api(), plan={"reproducibility": R.BatchDefault(2, 3)})  # ValueError 로 죽지 않는다
    assert r["reproducibility"]["verdict"] == "계산 불가" and r["verdicts"]["axes_passed"] == "계산 불가"


def test_report_defect_on_unready_base_is_uncomputable_not_failed():
    fake = FakeStats()
    api = {name: fake._fn(name) for name in jv.STATS_API}
    plan = {"reproducibility": R.BatchDefault(2, 2)}
    runs = ([_run("reproducibility", "a", "base", i) for i in (1, 2)]
            + [_run("reproducibility", "b", "base", 1), _run("reproducibility", "b", "base", 2, gate="blocked")]
            + [_run("defects", "b", "D01", 1, meta={"axis": "readability", "after": "x"}),
               _run("sham", "b", "S01", 1, meta={"axis": "sham"})])
    r = jv.report(runs, api, plan=plan)
    assert r["defects"]["D01"]["independence"] is None and "계산 불가" in r["defects"]["D01"]["note"]
    assert r["independence_passed"] is None and r["sham_passed"] is None
    assert r["verdicts"]["independence_passed"] == r["verdicts"]["sham_passed"] == "계산 불가"
    assert fake.named("independence_ok") == [] and fake.named("sham_ok") == []
    ok = [*runs[:2], _run("defects", "a", "D02", 1, meta={"axis": "readability", "after": "x"})]
    assert jv.report(ok, api, plan={"reproducibility": R.BatchDefault(1, 2)})["independence_passed"] is True
    assert fake.named("independence_ok")
    assert jv.report(ok, api, plan=plan)["independence_passed"] is None  # 계획 2건인데 1건만 — 덜 돎이면 흔들림 폭도 없다


def test_base_problems_needs_all_planned_runs_passed():
    fp = "f" * 64
    row = lambda rep, gate, rid="1": {"stage": "validate_reproducibility", "report": "a", "variant": "base", "repeat": str(rep),  # noqa: E731
                                      "mode": "dev", "manifest_sha256": fp, "result_file": "x.json", "gate": gate, "run_id": rid}
    assert "1/2회만" in jv.base_problems([row(1, "pass")], ["a"], fingerprint=fp, mode="dev", repeats=2)[0]
    assert "모두 pass 가 아니다" in jv.base_problems([row(1, "pass"), row(2, "blocked")], ["a"], fingerprint=fp, mode="dev",
                                                   repeats=2)[0]
    assert "0/2회만" in jv.base_problems([row(1, "")], ["a"], fingerprint=fp, mode="dev", repeats=2)[0]  # 실패 줄은 안 셈
    assert jv.base_problems([row(1, "pass"), row(2, "pass")], ["a"], fingerprint=fp, mode="dev", repeats=2) == []
    assert jv.base_problems([row(1, "pass"), row(2, "pass")], ["a"], fingerprint="e" * 64, mode="dev", repeats=2)


def test_defects_batch_refuses_before_reproducibility_has_run(tmp_path, vdir):
    import asyncio

    bases = asyncio.run(jv.dev_bases(jv.DEV_INPUTS))
    _plant_synthetic(tmp_path / "결함", bases)
    with pytest.raises(jv.RunnerError, match="바탕 성적서가 준비되지 않아"):
        jv.execute(_dev(tmp_path, batch="defects", defects=tmp_path / "결함", repeats=1), echo=_quiet()[1])
    assert not (tmp_path / "work" / "runs").exists()
    jv.execute(_dev(tmp_path, reports=["dev_binary", "dev_multilabel"], repeats=1), echo=_quiet()[1])  # 계획(기본 2회)보다 적다
    with pytest.raises(jv.RunnerError, match="1/2회만"):
        jv.execute(_dev(tmp_path, batch="defects", defects=tmp_path / "결함", repeats=1), echo=_quiet()[1])
    assert jv.execute(_dev(tmp_path, batch="defects", defects=tmp_path / "결함", repeats=1, base_repeats=1),
                      echo=_quiet()[1])["ran"] == 3


# ── J10 ③ dev 허용 목록·철자 우회(P1-3) ──


def _spellings(path: Path) -> list[Path]:
    """같은 폴더의 다른 철자 — 자모 분해(NFD), 대소문자, '..' 우회."""
    import unicodedata

    text = str(path)
    return [Path(unicodedata.normalize("NFD", text)), Path(text.upper()), path.parent / "x" / ".." / path.name]


def test_dev_guard_refuses_validation_folder_in_every_spelling(tmp_path, vdir):
    for p in _spellings(vdir / "00_봉인" / "미끼.json"):
        with pytest.raises(jv.RunnerError, match="검증 폴더"):
            jv.guard(p, "dev")


def test_dev_guard_refuses_default_validation_place_even_with_env(tmp_path, vdir):
    # 환경변수가 다른 곳을 가리켜도 저장소 옆 기본 자리(진짜 검증 폴더)는 거부 — 내용은 열지 않는다(경로만)
    with pytest.raises(jv.RunnerError, match="검증 폴더"):
        jv.guard(jv.DEFAULT_VALIDATION / "00_봉인" / "x.json", "dev")


def test_dev_guard_refuses_outside_allow_list(tmp_path, vdir, monkeypatch):
    outside = Path.home() / "j10_없는_폴더" / "장부.csv"
    with pytest.raises(jv.RunnerError, match="허용 목록 밖"):
        jv.guard(outside, "dev")
    monkeypatch.setenv("JUDGE_DEV_DIRS", str(outside.parent))
    assert jv.guard(outside, "dev") == outside.resolve()
    monkeypatch.setenv("JUDGE_DEV_DIRS", str(vdir))  # 허용 목록에 적어도 검증 폴더는 거부
    with pytest.raises(jv.RunnerError, match="검증 폴더"):
        jv.guard(vdir / "00_봉인", "dev")
    assert jv.guard(jv.DEV_INPUTS / "dev_binary.request.json", "dev")


def test_dev_guard_follows_symlink_into_validation_folder(tmp_path, vdir):
    link = tmp_path / "링크"
    link.symlink_to(vdir, target_is_directory=True)
    with pytest.raises(jv.RunnerError, match="검증 폴더"):
        jv.guard(link / "00_봉인", "dev")


def test_dev_sheets_reading_report_guard_every_path(tmp_path, vdir):
    jv.execute(_dev(tmp_path, repeats=1), echo=_quiet()[1])
    ledger = tmp_path / "work" / "장부.csv"
    bad = vdir / "00_봉인"
    with pytest.raises(jv.RunnerError, match="검증 폴더"):  # sheets 장부
        jv.cmd_sheets("dev", ledger=bad / "장부.csv", out=tmp_path / "s1", echo=_quiet()[1])
    with pytest.raises(jv.RunnerError, match="검증 폴더"):  # sheets 출력
        jv.cmd_sheets("dev", ledger=ledger, out=bad / "s", echo=_quiet()[1])
    with pytest.raises(jv.RunnerError, match="검증 폴더"):  # reading 출력
        jv.cmd_reading("dev", out=bad / "r", defects=tmp_path / "없는결함", echo=_quiet()[1])
    for kw in ({"answers": bad}, {"human": bad}):  # report 정답지·사람 채점
        with pytest.raises(jv.RunnerError, match="검증 폴더"):
            jv.cmd_report("dev", ledger=ledger, out=tmp_path / "o", api={}, **kw)
    assert not (bad / "s").exists() and not (bad / "r").exists()


def test_dev_collect_runs_refuses_result_file_pointing_into_validation_folder(tmp_path, vdir):
    jv.execute(_dev(tmp_path, repeats=1), echo=_quiet()[1])
    row = {**_rows(tmp_path / "work" / "장부.csv")[0], "result_file": str(vdir / "00_봉인" / "미끼.json")}
    with pytest.raises(jv.RunnerError, match="검증 폴더"):
        jv.collect_runs([row], fingerprint=criteria.fingerprint(), mode="dev", root=None)


# ── J10 ④ 덮어쓰기 금지·열쇠 짝 맞대기(P1-4), 블라인드(P1-5) ──


def test_sheets_and_reading_refuse_nonempty_output(tmp_path, vdir):
    loop = _full_loop(tmp_path)
    work = loop["work"]
    before = (work / "사람" / "채점지_인용.csv").read_bytes()
    with pytest.raises(jv.RunnerError, match="비어 있지 않다"):
        jv.cmd_sheets("dev", ledger=work / "장부.csv", out=work / "사람", n=16, overlap=2, seed=6, defects=work / "결함",
                      echo=_quiet()[1])
    assert (work / "사람" / "채점지_인용.csv").read_bytes() == before
    jv.cmd_reading("dev", out=work / "읽기", seed=1, defects=work / "결함", echo=_quiet()[1])
    with pytest.raises(jv.RunnerError, match="비어 있지 않다"):
        jv.cmd_reading("dev", out=work / "읽기", seed=2, defects=work / "결함", echo=_quiet()[1])
    (work / "새사람" / "읽기").mkdir(parents=True)  # 읽기 묶음을 먼저 만든 자리에는 채점지를 만들 수 있다
    (work / "새사람" / "읽기" / "x").write_text("", encoding="utf-8")
    jv.cmd_sheets("dev", ledger=work / "장부.csv", out=work / "새사람", n=16, overlap=2, seed=6, defects=work / "결함",
                  echo=_quiet()[1])


@pytest.mark.parametrize("where", ["sheet_head", "draw", "sha_file", "reading_head"])
def test_report_refuses_key_that_does_not_match_its_sheets(tmp_path, vdir, where):
    loop = _full_loop(tmp_path)
    human = loop["work"] / "사람"
    jv.cmd_reading("dev", out=human / "읽기", seed=4, defects=loop["work"] / "결함", echo=_quiet()[1])
    other = "0" * 64
    if where == "sheet_head":
        head, rows = _read_keyed(human / "채점지_민서.csv")
        _write_keyed(human / "채점지_민서.csv", "# 열쇠 sha256: " + other, rows)
    elif where == "draw":
        draw = json.loads((human / "뽑기.json").read_text(encoding="utf-8"))
        (human / "뽑기.json").write_text(json.dumps({**draw, "key_sha256": other}), encoding="utf-8")
    elif where == "sha_file":
        f = human / "열쇠" / "열쇠.sha256"
        f.chmod(0o644)
        f.write_text(other + "  열쇠.csv\n", encoding="utf-8")
    else:
        sheet = human / "읽기" / "민서" / "읽기표.csv"
        head, rows = _read_keyed(sheet)
        rows[0]["메모"] = "무언가"
        _write_keyed(sheet, "# 열쇠 sha256: " + other, rows)
    with pytest.raises(jv.RunnerError, match="짝과 맞지 않는다"):
        jv.cmd_report("dev", ledger=loop["work"] / "장부.csv", out=tmp_path / "o", human=human, echo=_quiet()[1], plan=PLAN2)


def test_build_reading_gives_one_edition_per_report_to_each_reader():
    from app.judge.validation import sheets as SH

    variants = {}
    for i, (base, axis, by) in enumerate([("a", "faithfulness", "민서"), ("a", "coverage", "인용"), ("b", "consistency", "민서"),
                                          ("b", "readability", "인용"), ("c", "faithfulness", "민서"), ("c", "coverage", "인용")]):
        variants[f"D0{i + 1}"] = {"base": base, "axis": axis, "planted_by": by, "after": f"문장 {i}"}
    for i, (base, by) in enumerate([("a", "민서"), ("b", "인용"), ("c", "민서")]):
        variants[f"S0{i + 1}"] = {"base": base, "axis": "sham", "planted_by": by}
    for seed in range(20):
        lists, key, unplaced = SH.build_reading(["a", "b", "c"], variants, seed)
        for reader, docs in lists.items():
            tags = [t for _, t, _ in docs]
            assert sorted(tags) == ["a", "b", "c"], (seed, reader, docs)  # 성적서마다 한 판본
            assert all(variants.get(v, {}).get("planted_by") != reader for _, _, v in docs)  # 심은 사람은 안 읽음
        read = {v for docs in lists.values() for _, _, v in docs}
        assert read >= {v for v in variants if v.startswith("D")} - set(unplaced)
        assert unplaced == []  # 성적서마다 두 사람이 하나씩 읽을 결함이 있다 — 6개 모두 자리


# ── J10 ⑧ 생존 변이 보강 — 순서 섞기 배선(M2)·봉인 해시(M4)·열쇠 읽기 전용(M6) ──


def test_run_order_is_the_seeded_shuffle_of_pending(tmp_path, vdir):
    args = _dev(tmp_path, reports=[], repeats=2, seed=3)
    jv.execute(args, echo=_quiet()[1])
    plan = json.loads((tmp_path / "work" / "runs" / "plan-reproducibility-3.json").read_text(encoding="utf-8"))
    keys = R.plan_runs("reproducibility", [(t, "base") for t in ("dev_binary", "dev_multiclass", "dev_multilabel")], 2)
    want = [k.__dict__ for k in R.shuffled(keys, 3)]
    assert want != [k.__dict__ for k in keys]  # 이 시드는 정렬 순서와 다르다(섞지 않으면 걸린다)
    assert plan["order"] == want
    rows = _rows(tmp_path / "work" / "장부.csv")
    assert [(r["report"], int(r["repeat"])) for r in rows] == [(k["report"], k["repeat"]) for k in want]  # 실제로 그 순서로 돌았다


def _sealed_vdir(vdir: Path, tags=("t1",)) -> None:
    """합성 봉인 폴더 — 개발 입력의 규칙 문장 서술을 봉인 꼴({request, view.result})로 + SHA256SUMS."""
    import asyncio
    import hashlib

    bases = asyncio.run(jv.dev_bases(jv.DEV_INPUTS))
    sealed = vdir / "00_봉인"
    lines = []
    for tag, src in zip(tags, sorted(bases)):
        body = json.dumps({"request": bases[src].request, "view": {"result": bases[src].result}}, ensure_ascii=False).encode()
        (sealed / f"{tag}.json").write_bytes(body)
        lines.append(f"{hashlib.sha256(body).hexdigest()}  {tag}.json")
    (sealed / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_verify_reads_sealed_only_when_hash_matches(tmp_path, vdir, frozen):
    sys.path.insert(0, str(ROOT / "scripts"))
    from dev_fake_llm_server import FakeOpenAI

    _sealed_vdir(vdir)
    args = jv.RunArgs(mode="verify", batch="reproducibility", live=True, yes=True, seed=1, reports=["t1"], repeats=1,
                      max_runs=1)
    git = _git_ok(criteria.recorded_manifest())
    got = jv.execute(args, client=FakeOpenAI("pass"), echo=_quiet()[1], git=git)
    assert got["ran"] == 1
    row = _rows(vdir / "장부.csv")[0]
    assert row["mode"] == "live" and row["result_file"].startswith("03_실행/")  # 검증 폴더 기준 상대 경로
    sealed = vdir / "00_봉인" / "t1.json"
    sealed.write_bytes(sealed.read_bytes().replace(b"}", b" }", 1))  # 한 바이트만 바꿈
    with pytest.raises(jv.RunnerError, match="봉인 파일 해시가 SHA256SUMS 와 다르다: t1.json"):
        jv.execute(jv.RunArgs(**{**args.__dict__, "repeats": 2, "max_runs": 2}), client=FakeOpenAI("pass"),
                   echo=_quiet()[1], git=git)
    assert len(_rows(vdir / "장부.csv")) == 1


def test_keys_are_read_only(tmp_path, vdir):
    import stat

    loop = _full_loop(tmp_path)
    human = loop["work"] / "사람"
    jv.cmd_reading("dev", out=human / "읽기", seed=4, defects=loop["work"] / "결함", echo=_quiet()[1])
    keys = [human / "열쇠" / "열쇠.csv", human / "열쇠" / "열쇠.sha256",
            human / "읽기" / "열쇠" / "읽기_열쇠.csv", human / "읽기" / "열쇠" / "읽기_열쇠.sha256"]
    for f in keys:
        mode = f.stat().st_mode
        assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH), (f, oct(mode))
