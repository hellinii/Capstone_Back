"""
judge_validate.py — 심판 검증 실행기(계획 17-3, 설계 7절). pytest 시험이 아니다.

두 모드
- dev    : 가짜 LLM(scripts/dev_fake_llm_server 의 가짜 OpenAI, 심판 '문제 없음' 답)으로 흐름만 돈다. 입력은 개발 판 4 입력
           (tests/data/judge_dev — 규칙 문장 서술을 키 없이 만든다)과 합성 결함·sham 뿐. 검증 폴더(JUDGE_VALIDATION_DIR, 없으면
           저장소 옆 JUDGE_ROUND/검증) 안의 경로는 입력·결함·장부·출력 어느 것으로도 받지 않는다(오염 방지 규칙 2). 결과는 임시 폴더.
- verify : 진짜 검증(V2). 아래 다섯이 모두 맞아야 시작한다 — 사유는 하나씩 따로 찍는다.
           ① 작업 트리가 깨끗하다(git status 빈칸)  ② 기준 버전이 고정 버전이고 기록된 지문과 같다(1.0-dev 면 거부)
           ③ 지금 MANIFEST == 태그 judge-criteria-v1.0 의 MANIFEST(파일 ↔ MANIFEST 도)  ④ --live  ⑤ 장부에 쓸 수 있다
           입력은 검증 폴더의 봉인 서술(00_봉인/<태그>.json — SHA256SUMS 와 맞아야), 결함·sham 은 02_결함/, 결과 원본은
           03_실행/<run_id>.json, 장부는 검증/장부.csv. 서술 글은 화면에 찍지 않는다(개수·해시·게이트만).

하위 명령(레포 루트에서)
- run <묶음> : reproducibility | defects | sham | extraction. 회차마다 장부 한 줄 + 결과 원본 파일 하나.
    - 이어 하기: 같은 모드·같은 기준 지문으로 결과 파일까지 적힌 회차는 건너뛴다(멈춘 뒤 같은 명령을 다시 치면 이어 간다).
    - 순서 섞기: 남은 회차를 시드로 섞는다(--seed, 없으면 새로 뽑아 찍음). 시드·순서는 plan-<묶음>-<시드>.json 과 결과 원본에.
    - 시작 때 예상 회차·비용(채점 1회 $0.15 실측)·장부 누계를 찍고 --yes 가 없으면 멈춘다. 묶음 회차 상한(기본 = 줄인 V2 안)을
      넘거나 라운드 남은 상한(JUDGE_LIVE_BUDGET_USD·JUDGE_LIVE_MAX_RUNS)보다 크면 시작하지 않는다. 돌다가 묶음 예산
      (예상 × 1.5)을 넘으면 멈춘다. 누계 $10 단위·비용 모름 20회마다 멈춰 확인(scripts/judge_live.py 의 Runner 그대로).
    - extraction 은 채점을 부르지 않는다(재현성 회차의 추출 기록을 report 가 정답지와 짝짓는다).
- report     : 장부·결과 원본으로 7.6 판정값을 계산한다. 계산은 app.judge.validation 의 stats·extraction·agreement 가 하고,
               이 명령은 그 함수들을 부르는 얇은 층이다(STATS_API). 결과는 verify 면 05_결과/검증결과.json, dev 면 출력 폴더.
- rescope    : 재검증 범위(정10) — 바뀐 파일(--changed 또는 --since <커밋>) → 다시 돌 모듈·묶음·사람 일치도 축.
- status     : 장부 누계와 상한.

상호작용
- 의존(import): scripts/judge_live.py(장부·상한·확인·클라이언트 — 다시 만들지 않는다), app.judge.core.judge_once(채점만),
  app.judge.validation.runs·variants, app.judge.criteria. dev 는 scripts/dev_fake_llm_server.FakeOpenAI
- 사용처: 사람이 직접 실행. 시험은 tests/test_judge_validate_runner.py
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import judge_live as jl  # noqa: E402

from app.judge import criteria  # noqa: E402
from app.judge.validation import runs as R  # noqa: E402
from app.judge.validation import variants as V  # noqa: E402

RunnerError = jl.RunnerError
MODES = ("dev", "verify")
TAG = "judge-criteria-v1.0"
MANIFEST_IN_REPO = "app/judge/criteria/MANIFEST.sha256"
DEV_INPUTS = ROOT / "tests" / "data" / "judge_dev"
SEALED = "00_봉인"
DEFECTS = "02_결함"
RUNS_OUT = "03_실행"
RESULTS_OUT = "05_결과"
ANSWERS = "01_정답지"
HUMAN = "04_사람평가"


# ── 읽기 경계 ─────────────────────────────────────────────────────────


def validation_root() -> Path:
    """검증 폴더 — 환경변수 JUDGE_VALIDATION_DIR, 없으면 저장소 옆 JUDGE_ROUND/검증(없어도 경로로만 쓴다)."""
    return jl.validation_dir() or (ROOT.parent / "JUDGE_ROUND" / "검증").resolve()


def guard(path: Path | str, mode: str) -> Path:
    """dev 모드는 검증 폴더 안 경로를 받지 않는다 — 열기 전에 거부한다."""
    resolved = Path(path).resolve()
    if mode == "dev":
        vdir = validation_root()
        if resolved == vdir or vdir in resolved.parents:
            raise RunnerError(f"dev 모드는 검증 폴더를 읽거나 쓰지 않는다: {resolved}")
    return resolved


# ── verify 시작 조건 ──────────────────────────────────────────────────


def _git(*args: str) -> tuple[int, str]:
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError as exc:
        return 1, str(exc)
    return p.returncode, p.stdout


def ledger_writable(path: Path) -> bool:
    path = Path(path)
    if path.exists():
        return path.is_file() and os.access(path, os.W_OK)
    return path.parent.is_dir() and os.access(path.parent, os.W_OK)


def verify_problems(*, live: bool, ledger_path: Path, git: Callable[..., tuple[int, str]] = _git) -> list[str]:
    """verify 모드를 막는 사유 전부(빈 목록이면 시작해도 된다). 사유마다 문장 하나."""
    problems = []
    code, out = git("status", "--porcelain")
    if code != 0 or out.strip():
        problems.append("작업 트리가 깨끗하지 않다 — 커밋하지 않은 변경이 있다(검증은 커밋된 기준으로만 돈다)")
    ver = criteria.version()
    if ver.endswith("-dev") or criteria.version_problem() is not None:
        problems.append(f"기준 버전이 고정되지 않았다(지금 {ver}) — V1 에서 scripts/judge_criteria_manifest.py freeze 1.0 으로 고정한다")
    code, tagged = git("show", f"{TAG}:{MANIFEST_IN_REPO}")
    if code != 0:
        problems.append(f"태그 {TAG} 가 없다 — 기준 고정 뒤 태그를 붙인다(정8)")
    elif tagged != criteria.recorded_manifest() or criteria.recorded_manifest() != criteria.manifest_text():
        problems.append(f"지금 기준 묶음 MANIFEST 가 태그 {TAG} 의 MANIFEST 와 다르다 — 고정한 기준으로만 검증한다(7.4)")
    if not live:
        problems.append("--live 가 없다 — verify 는 실호출이다")
    if not ledger_writable(ledger_path):
        problems.append(f"장부에 쓸 수 없다: {ledger_path}")
    return problems


# ── 바탕 서술 ─────────────────────────────────────────────────────────


@dataclass
class Base:
    tag: str
    request: dict
    result: dict  # 작업 결과의 result(서술 본문 + 칸별 출처)


async def dev_bases(inputs: Path) -> dict[str, Base]:
    """개발 입력마다 규칙 문장 서술(키 없음, 무료)을 만든다 — 작업 결과와 같은 모양(jobs.customer_result)."""
    from app.narrative.generator import generate_sections
    from app.narrative.jobs import customer_result
    from app.narrative.schemas import NarrativeRequest

    bases = {}
    for path in sorted(Path(inputs).glob("*.request.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        gen = await generate_sections(None, NarrativeRequest.model_validate(raw))
        tag = path.name[: -len(".request.json")]
        bases[tag] = Base(tag, raw, customer_result(gen))
    return bases


def sealed_bases(sealed: Path, tags: list[str]) -> dict[str, Base]:
    """봉인 서술(verify 전용). 파일 해시가 SHA256SUMS 와 다르면 거부. 내용은 찍지 않는다."""
    sums = {}
    for line in (sealed / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[name.strip()] = digest
    bases = {}
    for tag in tags:
        path = sealed / f"{tag}.json"
        body = path.read_bytes()
        if sums.get(path.name) != hashlib.sha256(body).hexdigest():
            raise RunnerError(f"봉인 파일 해시가 SHA256SUMS 와 다르다: {path.name}")
        record = json.loads(body)
        bases[tag] = Base(tag, record["request"], record["view"]["result"])
    return bases


# ── 회차 실행 ─────────────────────────────────────────────────────────


@dataclass
class Ctx:
    mode: str
    ledger: Path
    out: Path  # 결과 원본 폴더
    rel_root: Path | None = None  # 장부의 result_file 을 이 폴더 기준 상대 경로로(verify = 검증 폴더)
    confirm: Callable[[str], str] = input
    echo: Callable[[str], None] = print
    meta: dict = field(default_factory=dict)


async def judge_one(client, base: Base, slots: dict[str, str] | None):
    from app.judge.core import judge_once
    from app.narrative.baselines import build_benchmark_refs
    from app.narrative.schemas import NarrativeJobResult, NarrativeRequest

    req = NarrativeRequest.model_validate(base.request)
    body = V.apply_slots(base.result, slots) if slots else base.result
    result = NarrativeJobResult.model_validate(body)
    refs = build_benchmark_refs(req.task_type.value, req.fact_sheet.metrics, edition=4)  # 작업 경로와 같은 참조 범위
    run = await judge_once(client, req, result, result.section_sources, refs)
    return run, hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def batch_items(batch: str, *, bases: dict[str, Base], reports: list[str], variants: dict[str, tuple[dict, dict]]):
    """묶음의 (성적서, 변형) 목록."""
    if batch == "reproducibility":
        missing = [r for r in reports if r not in bases]
        if missing:
            raise RunnerError(f"없는 성적서: {missing}")
        return [(r, R.BASE_VARIANT) for r in reports]
    prefix = "D" if batch == "defects" else "S"
    items = []
    for vid, (meta, _) in sorted(variants.items()):
        if vid.startswith(prefix):
            if meta.get("base") not in bases:
                raise RunnerError(f"{vid}: 바탕 성적서 {meta.get('base')!r} 가 입력에 없다")
            items.append((meta["base"], vid))
    return items


async def run_batch(ctx: Ctx, batch: str, *, client, bases: dict[str, Base], reports: list[str],
                    variants: dict[str, tuple[dict, dict]], repeats: int, max_runs: int, budget: jl.Budget,
                    batch_budget_usd: float | None, seed: int | None, yes: bool) -> dict:
    ledger = jl.Ledger(ctx.ledger)
    if batch == "extraction":
        ctx.echo("[추출] 채점을 따로 부르지 않는다 — report 가 재현성 회차의 추출 기록을 정답지와 짝짓는다")
        return {"batch": batch, "ran": 0}
    keys = R.plan_runs(batch, batch_items(batch, bases=bases, reports=reports, variants=variants), repeats)
    fp = criteria.fingerprint()
    runner_mode = "live" if ctx.mode == "verify" else "dev"
    todo = R.pending(keys, ledger.rows(), fingerprint=fp, mode=runner_mode)
    seed = random.SystemRandom().randrange(1, 10**9) if seed is None else seed
    order = R.shuffled(todo, seed)
    est = R.estimate_usd(len(order)) if ctx.mode == "verify" else 0.0
    cap_usd = R.batch_budget_usd(len(order)) if batch_budget_usd is None else batch_budget_usd
    ctx.echo(f"[{ctx.mode}] 검증 묶음 {batch}: 계획 {len(keys)}회 · 끝남 {len(keys) - len(todo)}회(건너뜀) · 남음 {len(order)}회"
             f" · 순서 시드 {seed} · 묶음 예산 ${cap_usd:.2f}·{max_runs}회")
    if len(keys) > max_runs:
        raise RunnerError(f"묶음 회차 상한을 넘는다: 계획 {len(keys)}회 > {max_runs}회(--max-runs 로 바꾼다)")
    runner = jl.Runner(ledger, budget, runner_mode, confirm=ctx.confirm, out=ctx.echo)
    runner.start(jl.BatchPlan(f"validate_{batch}", runs=len(order), est_cost_usd=est), yes=yes)
    if not order:
        ctx.echo("남은 회차가 없다")
        return {"batch": batch, "ran": 0, "seed": seed}
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / f"plan-{batch}-{seed}.json").write_text(json.dumps(
        {"batch": batch, "seed": seed, "fingerprint": fp, "order": [k.__dict__ for k in order]},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    spent = 0.0
    gates = []
    for index, key in enumerate(order):
        meta, slots = variants.get(key.variant, ({}, None)) if key.variant != R.BASE_VARIANT else ({}, None)
        run, narrative_sha = await judge_one(client, bases[key.report], slots)
        run_id = ledger.next_run_id()
        target = ctx.out / f"{run_id}.json"
        body = json.dumps({
            "run_id": run_id, "key": key.__dict__, "order_seed": seed, "order_index": index, "mode": ctx.mode,
            **ctx.meta, "narrative_sha256": narrative_sha, "variant_meta": meta,
            "gate": run.gate, "failure": run.failure, "flagged_slots": run.flagged_slots, "unfixable": run.unfixable,
            "scorecard": run.scorecard, "record": run.record,
        }, ensure_ascii=False, indent=1) + "\n"
        target.write_text(body, encoding="utf-8")
        rel = target.relative_to(ctx.rel_root).as_posix() if ctx.rel_root else str(target)
        row = runner.record({**ctx.meta, **jl.judge_run_row(run), "run_id": run_id, "stage": key.stage,
                             "report": key.report, "variant": key.variant, "repeat": key.repeat,
                             "result_file": rel, "result_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()})
        gates.append(run.gate)
        spent += float(row["cost_usd"]) if row["cost_usd"] else 0.0
        if spent > cap_usd and index + 1 < len(order):
            raise RunnerError(f"묶음 예산 ${cap_usd:.2f} 를 넘었다(${spent:.2f}) — 멈춘다. 같은 명령으로 이어 한다")
    return {"batch": batch, "ran": len(order), "seed": seed, "gates": gates, "spent_usd": round(spent, 6)}


def load_variants(folder: Path | None) -> dict[str, tuple[dict, dict]]:
    if folder is None or not Path(folder).is_dir():
        return {}
    return {p.name: V.load_variant(p) for p in [*V.variant_folders(folder, "D"), *V.variant_folders(folder, "S")]}


@dataclass
class RunArgs:
    mode: str
    batch: str
    ledger: Path | None = None
    inputs: Path | None = None
    defects: Path | None = None
    out: Path | None = None
    live: bool = False
    yes: bool = False
    seed: int | None = None
    reports: list[str] = field(default_factory=list)
    repeats: int | None = None
    max_runs: int | None = None
    batch_budget_usd: float | None = None


def execute(a: RunArgs, *, client=None, confirm: Callable[[str], str] = input, echo: Callable[[str], None] = print,
            git: Callable[..., tuple[int, str]] = _git) -> dict:
    """run 하위 명령 본체. client 를 주면(시험) 그것으로 채점한다."""
    if a.mode not in MODES:
        raise RunnerError(f"모르는 모드: {a.mode}")
    if a.batch not in R.BATCHES:
        raise RunnerError(f"모르는 묶음: {a.batch}")
    vroot = validation_root()
    if a.mode == "dev":
        ledger = guard(a.ledger or Path(tempfile.mkdtemp(prefix="judge_validate_")) / "장부.csv", "dev")
        inputs = guard(a.inputs or DEV_INPUTS, "dev")
        defects = guard(a.defects, "dev") if a.defects else None
        out = guard(a.out or Path(tempfile.mkdtemp(prefix="judge_validate_runs_")), "dev")
        rel_root = None
    else:
        ledger = Path(a.ledger or vroot / "장부.csv")
        problems = verify_problems(live=a.live, ledger_path=ledger, git=git)
        if problems:
            raise RunnerError("verify 를 시작하지 않는다:\n  - " + "\n  - ".join(problems))
        inputs, defects, out, rel_root = vroot / SEALED, vroot / DEFECTS, Path(a.out or vroot / RUNS_OUT), vroot
    defaults = R.DEFAULTS[a.batch]
    repeats = a.repeats or defaults.repeats
    max_runs = a.max_runs if a.max_runs is not None else defaults.items * max(defaults.repeats, 1)
    variants = load_variants(defects) if a.batch in ("defects", "sham") else {}
    if a.mode == "dev":
        bases = asyncio.run(dev_bases(inputs))
        reports = a.reports or sorted(bases)
        if client is None:
            from dev_fake_llm_server import FakeOpenAI

            client = FakeOpenAI("pass")
    else:
        if a.batch == "reproducibility" and not a.reports:
            raise RunnerError("verify 재현성은 --reports 로 성적서 태그를 정한다(V1 에서 고른 3건)")
        needed = sorted(set(a.reports) | {m.get("base") for m, _ in variants.values()})
        bases = sealed_bases(inputs, needed)
        reports = a.reports
        if client is None:
            client = jl.make_clients(live=True).openai
            if client is None:
                raise RunnerError("OPENAI_API_KEY 가 없다")
    ctx = Ctx(a.mode, ledger, out, rel_root, confirm, echo, jl.run_meta())
    return asyncio.run(run_batch(ctx, a.batch, client=client, bases=bases, reports=reports, variants=variants,
                                 repeats=repeats, max_runs=max_runs, budget=jl.Budget.from_env(),
                                 batch_budget_usd=a.batch_budget_usd, seed=a.seed, yes=a.yes))


# ── report — 계산은 다른 모듈, 여기는 모아 넘기는 얇은 층 ─────────────────────

# 함수 이름 → 모듈(app.judge.validation.<모듈>). 실제 모듈과 합칠 때 서명이 다르면 이 표와 아래 report() 만 고친다.
STATS_API = {
    "icc_a1": "stats", "axis_scores": "stats", "reproducibility": "stats", "gate_stability": "stats",
    "defect_detected": "stats", "independence_ok": "stats", "sham_ok": "stats",
    "match_claims": "extraction", "recall_precision": "extraction",
    "agreement_rate": "agreement",
}


def load_stats_api() -> dict[str, Callable]:
    return {name: getattr(importlib.import_module(f"app.judge.validation.{mod}"), name) for name, mod in STATS_API.items()}


def collect_runs(ledger_rows: list[dict], *, fingerprint: str, mode: str, root: Path | None) -> list[dict]:
    """끝난 검증 회차의 결과 원본(같은 모드·같은 지문). 같은 회차가 여러 줄이면 마지막 것."""
    latest: dict = {}
    for row in ledger_rows:
        key = R.key_of_row(row)
        if key and row.get("mode") == mode and row.get("manifest_sha256") == fingerprint and row.get("result_file"):
            latest[key] = row
    runs = []
    for key, row in sorted(latest.items()):
        path = Path(row["result_file"])
        path = root / path if root and not path.is_absolute() else path
        body = path.read_bytes()
        if row.get("result_sha256") and hashlib.sha256(body).hexdigest() != row["result_sha256"]:
            raise RunnerError(f"결과 원본 해시가 장부와 다르다: {path}")
        runs.append({"key": key, **json.loads(body)})
    return runs


def report(runs: list[dict], api: dict[str, Callable], *, human_claims: dict[str, list[dict]] | None = None,
           human_pairs: list | None = None) -> dict:
    """7.6 판정값. runs = collect_runs 결과. human_claims = {성적서: 정답지 주장 목록}, human_pairs = 사람 일치도 짝."""
    by_batch: dict[str, list[dict]] = {}
    for r in runs:
        by_batch.setdefault(r["key"].batch, []).append(r)
    out: dict = {"runs": len(runs)}

    def scores(r):
        return api["axis_scores"](r["scorecard"]) if r.get("scorecard") else None

    repro = by_batch.get("reproducibility", [])
    base_scores: dict[str, list[dict]] = {}
    base_gates: dict[str, list] = {}
    for r in sorted(repro, key=lambda r: (r["key"].report, r["key"].repeat)):
        base_scores.setdefault(r["key"].report, []).append(scores(r))
        base_gates.setdefault(r["key"].report, []).append(r.get("gate"))
    if repro:
        axes = sorted({a for s in base_scores.values() for x in s if x for a in x})
        out["reproducibility"] = {
            axis: api["reproducibility"]([[x[axis] if x else None for x in s] for s in base_scores.values()]) for axis in axes}
        out["gate_stability"] = api["gate_stability"](list(base_gates.values()))
        out["base_gate_pass"] = {tag: all(g == "pass" for g in gs) for tag, gs in base_gates.items()}  # Q11 바탕 통과 확인
    for batch in ("defects", "sham"):
        rows = by_batch.get(batch, [])
        if not rows:
            continue
        per = {}
        for r in rows:
            vid, meta = r["key"].variant, r.get("variant_meta") or {}
            base = [s for s in base_scores.get(r["key"].report, []) if s]
            entry = per.setdefault(vid, {"base": r["key"].report, "axis": meta.get("axis"), "runs": []})
            entry["runs"].append({"gate": r.get("gate"), "failure": r.get("failure"), "scores": scores(r),
                                  **({"detected": api["defect_detected"](meta, r)} if batch == "defects" else {})})
            run_scores = [x["scores"] for x in entry["runs"] if x["scores"]]
            if batch == "defects":
                entry["independence"] = api["independence_ok"](meta.get("axis"), run_scores, base)
            else:
                entry["sham"] = api["sham_ok"](run_scores, base)
        out[batch] = per
    if human_claims:
        extraction = {}
        for r in repro:
            tag = r["key"].report
            if tag not in human_claims:
                continue
            system = ((r.get("record") or {}).get("extract") or {}).get("claims") or []
            matches = api["match_claims"](human_claims[tag], system)
            extraction.setdefault(tag, []).append(api["recall_precision"](matches, len(human_claims[tag]), len(system)))
        out["extraction"] = extraction
    if human_pairs:
        out["agreement"] = api["agreement_rate"](human_pairs)
    return out


def read_human_claims(folder: Path, tags: list[str]) -> dict[str, list[dict]]:
    import csv

    found = {}
    for tag in tags:
        path = Path(folder) / f"{tag}.claims.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if any(v.strip() for v in r.values() if v)]
            if rows:
                found[tag] = rows
    return found


def cmd_report(mode: str, *, ledger: Path | None, out: Path | None, api: dict[str, Callable] | None = None,
               answers: Path | None = None, echo: Callable[[str], None] = print) -> dict:
    vroot = validation_root()
    if mode == "dev":
        if ledger is None:
            raise RunnerError("dev report 는 --ledger(임시 장부)를 준다")
        ledger = guard(ledger, "dev")
        out = guard(out or Path(tempfile.mkdtemp(prefix="judge_validate_report_")), "dev")
        answers = guard(answers, "dev") if answers else None
        root = None
    else:
        ledger, out, root = Path(ledger or vroot / "장부.csv"), Path(out or vroot / RESULTS_OUT), vroot
        answers = Path(answers) if answers else vroot / ANSWERS
    runs = collect_runs(jl.Ledger(ledger).rows(), fingerprint=criteria.fingerprint(),
                        mode="live" if mode == "verify" else "dev", root=root)
    tags = sorted({r["key"].report for r in runs if r["key"].batch == "reproducibility"})
    human = read_human_claims(answers, tags) if answers and Path(answers).is_dir() else {}
    result = report(runs, api or load_stats_api(), human_claims=human)
    result.update({"mode": mode, "criteria_version": criteria.version(), "fingerprint": criteria.fingerprint(), "tag": TAG})
    out.mkdir(parents=True, exist_ok=True)
    target = out / "검증결과.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    echo(f"검증 결과 {target} — 회차 {result['runs']}개")
    return result


# ── 재검증 범위(정10) ─────────────────────────────────────────────────


def cmd_rescope(changed: list[str] | None, since: str | None, *, git: Callable[..., tuple[int, str]] = _git,
                echo: Callable[[str], None] = print) -> dict:
    from app.judge.validation.rescope import rescope

    files = list(changed or [])
    if since:
        code, out = git("diff", "--name-only", since, "HEAD")
        if code != 0:
            raise RunnerError(f"git diff 실패: {since}")
        files += [line for line in out.splitlines() if line.strip()]
    if not files:
        raise RunnerError("바뀐 파일이 없다(--changed 또는 --since)")
    result = rescope(files)
    echo(f"다시 돌 모듈: {', '.join(result['modules']) or '없음'}")
    echo(f"다시 돌 묶음: {', '.join(result['batches']) or '없음'}")
    echo(f"사람 일치도 다시 볼 축(그 축 LLM 판정이 달라졌으면): {', '.join(result['human_axes']) or '없음'}")
    for path, mods in result["by_file"].items():
        echo(f"  {path} → {', '.join(mods) or '영향 없음'}")
    return result


# ── 명령줄 ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="심판 검증 실행기(dev / verify)")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="묶음 하나 실행(회차마다 장부 한 줄)")
    run.add_argument("batch", choices=R.BATCHES)
    run.add_argument("--mode", choices=MODES, required=True)
    run.add_argument("--ledger")
    run.add_argument("--inputs", help="dev 전용 — 개발 입력 폴더(기본 tests/data/judge_dev)")
    run.add_argument("--defects", help="dev 전용 — 합성 결함·sham 폴더")
    run.add_argument("--out")
    run.add_argument("--reports", default="", help="재현성 성적서 태그(쉼표). verify 는 꼭 준다")
    run.add_argument("--repeats", type=int)
    run.add_argument("--max-runs", type=int, help="묶음 회차 상한(기본 = 줄인 V2 안)")
    run.add_argument("--batch-budget-usd", type=float, help="묶음 예산(기본 = 예상 × 1.5)")
    run.add_argument("--seed", type=int)
    run.add_argument("--live", action="store_true")
    run.add_argument("--yes", action="store_true")
    rep = sub.add_parser("report", help="장부·결과 원본으로 7.6 판정값 계산")
    rep.add_argument("--mode", choices=MODES, required=True)
    rep.add_argument("--ledger")
    rep.add_argument("--out")
    rep.add_argument("--answers")
    rs = sub.add_parser("rescope", help="재검증 범위(정10): 바뀐 파일 → 다시 돌 모듈·묶음")
    rs.add_argument("--changed", nargs="*", default=[])
    rs.add_argument("--since", help="이 커밋(예: judge-criteria-v1.0)부터 HEAD 까지 바뀐 파일")
    st = sub.add_parser("status", help="장부 누계와 상한")
    st.add_argument("--ledger", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            return jl.main(["status", "--ledger", args.ledger])
        if args.command == "rescope":
            cmd_rescope(args.changed, args.since)
            return 0
        if args.command == "report":
            cmd_report(args.mode, ledger=Path(args.ledger) if args.ledger else None, out=Path(args.out) if args.out else None,
                       answers=Path(args.answers) if args.answers else None)
            return 0
        if args.mode == "verify" and (args.inputs or args.defects):
            raise RunnerError("verify 는 입력·결함을 검증 폴더에서만 읽는다(--inputs·--defects 는 dev 전용)")
        summary = execute(RunArgs(
            mode=args.mode, batch=args.batch, ledger=Path(args.ledger) if args.ledger else None,
            inputs=Path(args.inputs) if args.inputs else None, defects=Path(args.defects) if args.defects else None,
            out=Path(args.out) if args.out else None, live=args.live, yes=args.yes, seed=args.seed,
            reports=[t for t in args.reports.split(",") if t], repeats=args.repeats, max_runs=args.max_runs,
            batch_budget_usd=args.batch_budget_usd))
        print(json.dumps({k: v for k, v in summary.items() if k != "gates"} | {"gates": summary.get("gates")},
                         ensure_ascii=False))
        return 0
    except RunnerError as exc:
        print(f"멈춤: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
