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
    - 이어 하기: 같은 모드·같은 기준 지문으로 결과 파일까지 적히고 게이트 판정이 난 회차는 건너뛴다(멈춘 뒤 같은 명령을
      다시 치면 이어 간다). 채점이 실패한 회차(gate 빈칸)는 다시 돈다 — 실패 줄의 비용은 장부 누계에 그대로 남는다.
    - 순서 섞기: 남은 회차를 시드로 섞는다(--seed, 없으면 새로 뽑아 찍음). 시드·순서는 plan-<묶음>-<시드>.json 과 결과 원본에.
    - 시작 때 예상 회차·비용(채점 1회 $0.15 실측)·장부 누계를 찍고 --yes 가 없으면 멈춘다. 묶음 회차 상한(기본 = 줄인 V2 안)을
      넘거나 라운드 남은 상한(JUDGE_LIVE_BUDGET_USD·JUDGE_LIVE_MAX_RUNS)보다 크면 시작하지 않는다. 돌다가 묶음 예산
      (예상 × 1.5)을 넘으면 멈춘다. 누계 $10 단위·비용 모름 20회마다 멈춰 확인(scripts/judge_live.py 의 Runner 그대로).
    - extraction 은 채점을 부르지 않는다(재현성 회차의 추출 기록을 report 가 정답지와 짝짓는다).
- report     : 장부·결과 원본으로 7.6 판정값을 계산한다. 계산은 app.judge.validation 의 stats·extraction·agreement 가 하고,
               이 명령은 그 함수들을 부르는 얇은 층이다(STATS_API). 결과는 verify 면 05_결과/검증결과.json, dev 면 출력 폴더.
- check-defects : 결함본·sham 폴더 꼴 검사(meta 칸·심은 사람 역할·칸 하나·문장 하나·before/after·diff.txt). verify 는 V1 뒤
               --allow-sealed 로만(봉인 서술을 연다), dev 는 합성 바탕(개발 입력)으로.
- make-diff <D01> : diff.txt 를 도구로 쓴다.
- sheets     : 사람 일치도 채점지(블라인드 — 시스템 판정·결함 여부 없음, 결함본 문항은 7.2 역할대로 인용/민서), 열쇠(따로 + 해시,
               읽기 전용), 읽을 문서(문서 번호만). 층 나눈 표본(--n 기본 80, --overlap 축마다 겹침 5, --seed 기록).
- reading    : 8절 블라인드 읽기 묶음(P1) — 사람마다 문서(결함본은 심지 않은 사람에게만)·빈 읽기표·열쇠. report 가 효용성 표로 센다.
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


def variant_problems(variants: dict[str, tuple[dict, dict]], bases: dict[str, Base], diffs: dict[str, str | None] | None = None,
                     ) -> list[str]:
    """변형마다 꼴 검사(variants.check_variant). diffs 를 주지 않으면 diff.txt 는 보지 않는다(실행 전 검사는 칸·문장만)."""
    out = []
    for vid, (meta, slots) in sorted(variants.items()):
        base = bases.get(meta.get("base"))
        if base is None:
            out.append(f"{vid}: 바탕 성적서 {meta.get('base')!r} 가 없다")
            continue
        diff = diffs.get(vid) if diffs is not None else None
        out += [f"{vid}: {p}" for p in V.check_variant(vid, meta, slots, diff, base.result, check_diff=diffs is not None)]
    return out


def _slot_ok(base: Base, meta: dict, slots: dict) -> bool:
    try:
        V.read_slot(base.result, meta.get("slot", ""))
    except (KeyError, IndexError, ValueError):
        return False
    return isinstance(slots.get(meta.get("slot")), str)


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
    diffs = {vid: ((defects / vid / "diff.txt").read_text(encoding="utf-8") if (defects / vid / "diff.txt").exists() else None)
             for vid in variants}
    bad = variant_problems(variants, bases, diffs)  # check-defects 와 같은 검사(diff.txt 까지)
    if bad:
        raise RunnerError("결함본·sham 꼴이 틀려 시작하지 않는다:\n  - " + "\n  - ".join(bad))
    ctx = Ctx(a.mode, ledger, out, rel_root, confirm, echo, jl.run_meta())
    return asyncio.run(run_batch(ctx, a.batch, client=client, bases=bases, reports=reports, variants=variants,
                                 repeats=repeats, max_runs=max_runs, budget=jl.Budget.from_env(),
                                 batch_budget_usd=a.batch_budget_usd, seed=a.seed, yes=a.yes))


# ── report — 계산은 다른 모듈, 여기는 모아 넘기는 얇은 층 ─────────────────────

# 함수 이름 → 모듈(app.judge.validation.<모듈>). 서명은 그 모듈 머리말(stats·extraction·agreement 갈래, 세션 5).
STATS_API = {
    "axis_scores": "stats", "reproducibility": "stats", "defect_detected": "stats", "detection_passed": "stats",
    "independence_ok": "stats", "sham_ok": "stats", "agreement_rate": "stats", "human_agreement_ok": "stats",
    "gold_from_csv": "extraction", "pred_from_extract": "extraction", "match_claims": "extraction",
    "recall_precision": "extraction", "runs_distribution": "extraction",
    "cohen_kappa": "agreement", "gwet_ac1": "agreement",
}


def load_stats_api() -> dict[str, Callable]:
    return {name: getattr(importlib.import_module(f"app.judge.validation.{mod}"), name) for name, mod in STATS_API.items()}


def collect_runs(ledger_rows: list[dict], *, fingerprint: str, mode: str, root: Path | None) -> list[dict]:
    """끝난 검증 회차의 결과 원본(같은 모드·같은 지문, 판정이 난 줄 — runs.finished_row). 같은 회차가 여러 줄이면 마지막
    성공 줄(실패 뒤 다시 돈 회차는 다시 돈 결과, 성공 뒤에 남은 실패 줄은 무시)."""
    latest: dict = {}
    for row in ledger_rows:
        key = R.finished_row(row, fingerprint=fingerprint, mode=mode)
        if key is not None:
            latest[key] = row
    runs = []
    for key, row in sorted(latest.items()):
        path = Path(row["result_file"])
        path = root / path if root and not path.is_absolute() else path
        body = path.read_bytes()
        if row.get("result_sha256") and hashlib.sha256(body).hexdigest() != row["result_sha256"]:
            raise RunnerError(f"결과 원본 해시가 장부와 다르다: {path}")
        runs.append({**json.loads(body), "key": key})  # 열쇠는 장부의 RunKey(결과 원본의 key 는 dict)
    return runs


def planted_sentence(meta: dict) -> str | None:
    """탐지를 볼 심은 문장 — 바꾸거나 끼운 문장(after), 지우기면 지운 문장(before)."""
    return meta.get("after") or meta.get("before") or None


def report(runs: list[dict], api: dict[str, Callable], *, human_claims: dict[str, tuple[list, list]] | None = None,
           human_scores: dict | None = None, human_reading: dict | None = None) -> dict:
    """7.6 판정값(+ 8절 효용성 — human_reading = sheets.reading_hits 결과). runs = collect_runs 결과(판정은 장부 순서가 아니라 회차 번호 순).
    human_claims = {성적서: (claims.csv 줄, sentences.csv 줄)}, human_scores = sheets.human_pairs 결과."""
    by_batch: dict[str, list[dict]] = {}
    for r in sorted(runs, key=lambda r: r["key"]):
        by_batch.setdefault(r["key"].batch, []).append(r)
    out: dict = {"runs": len(runs)}

    base_runs: dict[str, list[dict]] = {}
    for r in by_batch.get("reproducibility", []):
        base_runs.setdefault(r["key"].report, []).append(r)
    if base_runs:
        counts = {len(v) for v in base_runs.values()}
        repro = api["reproducibility"](base_runs, expected_runs=counts.pop() if len(counts) == 1 else None)
        out["reproducibility"] = repro
        out["gates_passed"] = repro["gates_passed"]
        out["base_gate_pass"] = {tag: all(r.get("gate") == "pass" for r in rs) for tag, rs in base_runs.items()}  # Q11

    def grouped(batch):
        per: dict[str, list[dict]] = {}
        for r in by_batch.get(batch, []):
            per.setdefault(r["key"].variant, []).append(r)
        return per

    defects = {}
    for vid, rs in sorted(grouped("defects").items()):
        meta = rs[0].get("variant_meta") or {}
        axis, base = meta.get("axis"), base_runs.get(rs[0]["key"].report, [])
        hits = [api["defect_detected"](r, axis, sentence=planted_sentence(meta), item_id=meta.get("item_id") or None)
                for r in rs]
        defects[vid] = {"base": rs[0]["key"].report, "axis": axis, "hits": hits, "scores": [api["axis_scores"](r) for r in rs],
                        "independence": api["independence_ok"](base, rs, axis) if base else None}
    if defects:
        out["defects"] = defects
        out["detection"] = api["detection_passed"]({v: {"axis": d["axis"], "hits": d["hits"]} for v, d in defects.items()})
        out["independence_passed"] = all(d["independence"] and d["independence"]["ok"] for d in defects.values())
    shams = {}
    for vid, rs in sorted(grouped("sham").items()):
        base = base_runs.get(rs[0]["key"].report, [])
        shams[vid] = {"base": rs[0]["key"].report, "sham": api["sham_ok"](base, rs) if base else None}
    if shams:
        out["sham"] = shams
        out["sham_passed"] = all(s["sham"] and s["sham"]["ok"] for s in shams.values())

    if human_claims:
        extraction, pairs, gold_n, pred_n = {}, 0, 0, 0
        for tag, (claim_rows, sentence_rows) in sorted(human_claims.items()):
            if tag not in base_runs:
                continue
            gold = api["gold_from_csv"](claim_rows, sentence_rows)
            preds = [api["pred_from_extract"](((r.get("record") or {}).get("extract") or {}).get("claims") or [], sentence_rows)
                     for r in base_runs[tag]]
            per_run = [api["recall_precision"](api["match_claims"](gold, p)) for p in preds]
            for m in per_run:
                pairs, gold_n, pred_n = pairs + m["n_pairs"], gold_n + m["n_gold"], pred_n + m["n_pred"]
            extraction[tag] = {"runs": per_run, "distribution": api["runs_distribution"](gold, preds)}
        out["extraction"] = {"reports": extraction, "pooled": {
            "recall": pairs / gold_n if gold_n else None, "precision": pairs / pred_n if pred_n else None,
            "n_pairs": pairs, "n_gold": gold_n, "n_pred": pred_n}}

    if human_reading and defects:
        hits = human_reading["defects"]
        out["effectiveness"] = {  # 8절 — 기준은 심어 둔 결함(사람이 아니라)
            "defects": len(defects), "system_found": sum(any(d["hits"]) for d in defects.values()),
            "human_found": sum(1 for v in defects if hits.get(v, {}).get("found")),
            "human_read": sum(1 for v in defects if v in hits), "human_false_alarms": human_reading["false_alarms"]}
    if human_scores:
        def rates(pairs: list[tuple]) -> dict | None:
            if not pairs:
                return None
            a, b = [x for x, _ in pairs], [y for _, y in pairs]
            return {"n": len(pairs), "rate": api["agreement_rate"](a, b), "kappa": api["cohen_kappa"](a, b),
                    "ac1": api["gwet_ac1"](a, b, ["예", "아니오"])}

        sys_h = {axis: rates(p) for axis, p in human_scores["system_human"].items()}
        h_h = {axis: rates(p) for axis, p in human_scores["human_human"].items()}
        all_sys = rates([p for ps in human_scores["system_human"].values() for p in ps])
        all_hh = rates([p for ps in human_scores["human_human"].values() for p in ps])
        out["agreement"] = {"system_human": all_sys, "human_human": all_hh, "by_axis": {"system_human": sys_h, "human_human": h_h},
                            "blank": human_scores["blank"],
                            "passed": api["human_agreement_ok"](all_sys["rate"], all_hh["rate"]) if all_sys and all_hh else None}
    return out


def _csv_rows(path: Path) -> list[dict]:
    import csv

    with Path(path).open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]


def read_human_claims(folder: Path, tags: list[str]) -> dict[str, tuple[list, list]]:
    """정답지(01_정답지) — 주장 표시가 있는 성적서만 {태그: (claims 줄, sentences 줄)}."""
    found = {}
    for tag in tags:
        claims, sentences = Path(folder) / f"{tag}.claims.csv", Path(folder) / f"{tag}.sentences.csv"
        if claims.exists() and sentences.exists():
            rows = _csv_rows(claims)
            if rows:
                found[tag] = (rows, _csv_rows(sentences))
    return found


def read_human_scores(folder: Path) -> dict | None:
    """사람 채점(04_사람평가) — 채점지 두 장과 열쇠가 있고 판정이 하나라도 적혔을 때만 짝으로."""
    from app.judge.validation import sheets as SH

    key = Path(folder) / "열쇠" / "열쇠.csv"
    sheets = {who: Path(folder) / f"채점지_{who}.csv" for who in ("인용", "민서")}
    if not key.exists() or not all(p.exists() for p in sheets.values()):
        return None
    filled = {who: _csv_rows(p) for who, p in sheets.items()}
    if not any((r.get(SH.SHEET_COLUMNS[6]) or "").strip() for rows in filled.values() for r in rows):
        return None
    return SH.human_pairs(_csv_rows(key), filled)


def read_human_reading(folder: Path) -> dict | None:
    """8절 읽기 결과(04_사람평가/읽기) — 열쇠와 읽기표가 있고 하나라도 적혔을 때만."""
    from app.judge.validation import sheets as SH

    key = Path(folder) / "읽기" / "열쇠" / "읽기_열쇠.csv"
    sheets = [Path(folder) / "읽기" / r / "읽기표.csv" for r in SH.READERS]
    if not key.exists() or not all(p.exists() for p in sheets):
        return None
    filled = [r for p in sheets for r in _csv_rows(p) if r["문서"].startswith("읽기-")]
    if not any((r.get(SH.READING_COLUMNS[1]) or "").strip() or (r.get("메모") or "").strip() for r in filled):
        return None
    from app.judge.validation.stats import touches  # 결함 탐지와 같은 '짚었나' 규칙

    return SH.reading_hits(_csv_rows(key), filled, touches)


def cmd_report(mode: str, *, ledger: Path | None, out: Path | None, api: dict[str, Callable] | None = None,
               answers: Path | None = None, human: Path | None = None, echo: Callable[[str], None] = print) -> dict:
    vroot = validation_root()
    if mode == "dev":
        if ledger is None:
            raise RunnerError("dev report 는 --ledger(임시 장부)를 준다")
        ledger = guard(ledger, "dev")
        out = guard(out or Path(tempfile.mkdtemp(prefix="judge_validate_report_")), "dev")
        answers = guard(answers, "dev") if answers else None
        human = guard(human, "dev") if human else None
        root = None
    else:
        ledger, out, root = Path(ledger or vroot / "장부.csv"), Path(out or vroot / RESULTS_OUT), vroot
        answers = Path(answers) if answers else vroot / ANSWERS
        human = Path(human) if human else vroot / HUMAN
    runs = collect_runs(jl.Ledger(ledger).rows(), fingerprint=criteria.fingerprint(),
                        mode="live" if mode == "verify" else "dev", root=root)
    tags = sorted({r["key"].report for r in runs if r["key"].batch == "reproducibility"})
    claims = read_human_claims(answers, tags) if answers and Path(answers).is_dir() else {}
    scores = read_human_scores(human) if human and Path(human).is_dir() else None
    reading = read_human_reading(human) if human and Path(human).is_dir() else None
    result = report(runs, api or load_stats_api(), human_claims=claims, human_scores=scores, human_reading=reading)
    result.update({"mode": mode, "criteria_version": criteria.version(), "fingerprint": criteria.fingerprint(), "tag": TAG})
    out.mkdir(parents=True, exist_ok=True)
    target = out / "검증결과.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    echo(f"검증 결과 {target} — 회차 {result['runs']}개")
    return result


# ── 결함본 검사·차이 만들기·사람 채점지(17-5) ────────────────────────────


def sealed_reading_problems(allow_sealed: bool) -> list[str]:
    """봉인 서술을 여는 명령(결함 검사·차이·채점지)의 조건 — 기준 고정 뒤(V1), 사람이 허락했을 때만(--allow-sealed)."""
    problems = []
    if not allow_sealed:
        problems.append("--allow-sealed 가 없다 — 봉인 서술 열람은 V1(기준 고정) 뒤 사람이 허락한 뒤에만")
    if criteria.version().endswith("-dev") or criteria.version_problem() is not None:
        problems.append(f"기준 버전이 고정되지 않았다(지금 {criteria.version()}) — 봉인은 기준 고정 전에 열지 않는다")
    return problems


def _variant_inputs(mode: str, *, defects: Path | None, inputs: Path | None, allow_sealed: bool):
    """(변형들, 바탕들, diff 글들, 결함 폴더). verify 는 검증 폴더 봉인·결함, dev 는 주어진 합성 폴더·개발 입력."""
    if mode == "verify":
        problems = sealed_reading_problems(allow_sealed)
        if problems:
            raise RunnerError("봉인을 열지 않는다:\n  - " + "\n  - ".join(problems))
        vroot = validation_root()
        folder, bases_of = vroot / DEFECTS, lambda tags: sealed_bases(vroot / SEALED, tags)
    else:
        if defects is None:
            raise RunnerError("dev 는 --defects(합성 결함 폴더)를 준다")
        folder = guard(defects, "dev")
        dev_in = guard(inputs or DEV_INPUTS, "dev")
        bases_of = lambda tags: {t: b for t, b in asyncio.run(dev_bases(dev_in)).items() if t in tags}  # noqa: E731
    variants = load_variants(folder)
    bases = bases_of(sorted({m.get("base") for m, _ in variants.values() if m.get("base")}))
    diffs = {vid: ((folder / vid / "diff.txt").read_text(encoding="utf-8") if (folder / vid / "diff.txt").exists() else None)
             for vid in variants}
    return variants, bases, diffs, folder


def cmd_check_defects(mode: str, *, defects: Path | None = None, inputs: Path | None = None, allow_sealed: bool = False,
                      echo: Callable[[str], None] = print) -> list[str]:
    """결함본·sham 폴더 전체 검사. 통과/문제만 찍는다(서술 글은 찍지 않음)."""
    variants, bases, diffs, _ = _variant_inputs(mode, defects=defects, inputs=inputs, allow_sealed=allow_sealed)
    problems = variant_problems(variants, bases, diffs)
    for vid in sorted(variants):
        mine = [p for p in problems if p.startswith(f"{vid}: ")]
        echo(f"{vid} {'통과' if not mine else '문제 ' + str(len(mine)) + '개'}")
        for p in mine:
            echo(f"  - {p[len(vid) + 2:]}")
    echo(f"결함본 {sum(v.startswith('D') for v in variants)}개 · sham {sum(v.startswith('S') for v in variants)}개 · 문제 {len(problems)}개")
    return problems


def cmd_make_diff(mode: str, variant: str, *, defects: Path | None = None, inputs: Path | None = None,
                  allow_sealed: bool = False, echo: Callable[[str], None] = print) -> Path:
    """diff.txt 를 도구로 쓴다(사람이 손으로 쓰지 않는다)."""
    variants, bases, _, folder = _variant_inputs(mode, defects=defects, inputs=inputs, allow_sealed=allow_sealed)
    if variant not in variants:
        raise RunnerError(f"없는 변형: {variant}")
    meta, slots = variants[variant]
    base = bases.get(meta.get("base"))
    if base is None or not _slot_ok(base, meta, slots):
        raise RunnerError(f"{variant}: 바탕 성적서·칸을 찾지 못했다")
    target = folder / variant / "diff.txt"
    target.write_text(V.make_diff(V.read_slot(base.result, meta["slot"]), slots[meta["slot"]], meta["slot"]), encoding="utf-8")
    echo(f"{variant} diff.txt 씀")
    return target


def doc_text(base: Base, slots: dict | None) -> str:
    """사람이 읽을 문서 — 심판이 읽는 것과 같은 조각(서술 칸 + 고정 문구 + 표). 결함 여부·변형 이름은 없다."""
    from app.judge.material import parse_material
    from app.judge.reading import build_pieces, render
    from app.narrative.baselines import build_benchmark_refs
    from app.narrative.schemas import NarrativeJobResult, NarrativeRequest

    req = NarrativeRequest.model_validate(base.request)
    result = NarrativeJobResult.model_validate(V.apply_slots(base.result, slots) if slots else base.result)
    material, _ = parse_material(req.judge_material)
    refs = build_benchmark_refs(req.task_type.value, req.fact_sheet.metrics, edition=4)
    pieces = build_pieces(result, result.section_sources, material, refs, req.fact_sheet.metrics)
    return f"# 바탕 성적서 데이터: {base.tag}\n\n" + render(pieces) + "\n"


def cmd_sheets(mode: str, *, ledger: Path | None, out: Path | None, n: int = 80, overlap: int = 5, seed: int | None = None,
               defects: Path | None = None, inputs: Path | None = None, allow_sealed: bool = False,
               echo: Callable[[str], None] = print) -> dict:
    """사람 일치도 채점지(블라인드, 역할대로)·열쇠(따로 + 해시)·읽을 문서. 표본은 검증 회차 결과에서 층 나눠 뽑는다(시드 기록)."""
    import csv
    import io
    import stat

    from app.judge.validation import sheets as SH

    vroot = validation_root()
    if mode == "verify":
        problems = sealed_reading_problems(allow_sealed)
        if problems:
            raise RunnerError("봉인을 열지 않는다:\n  - " + "\n  - ".join(problems))
        ledger, out, root = Path(ledger or vroot / "장부.csv"), Path(out or vroot / HUMAN), vroot
    else:
        if ledger is None:
            raise RunnerError("dev sheets 는 --ledger(임시 장부)를 준다")
        ledger = guard(ledger, "dev")
        out = guard(out or Path(tempfile.mkdtemp(prefix="judge_validate_sheets_")), "dev")
        root = None
    runs = collect_runs(jl.Ledger(ledger).rows(), fingerprint=criteria.fingerprint(),
                        mode="live" if mode == "verify" else "dev", root=root)
    # 같은 성적서·변형은 첫 회차만(카-4: 판정은 1회째로 본다 — V1 에서 바꾸면 여기)
    first = {}
    for r in sorted(runs, key=lambda r: r["key"]):
        first.setdefault((r["key"].report, r["key"].variant), r)
    types = criteria.load_json("contradictions.json")["types"]
    items = [i for r in first.values() for i in SH.items_from_run({**r, "key": r["key"].__dict__}, types)]
    seed = random.SystemRandom().randrange(1, 10**9) if seed is None else seed
    assigned = SH.assign(SH.sample(items, n, seed), overlap_per_axis=overlap, seed=seed)
    sheets, key, docs = SH.build(assigned, seed)
    variants = {}
    if any(r["key"].variant != R.BASE_VARIANT for r in first.values()):
        variants, _, _, _ = _variant_inputs(mode, defects=defects, inputs=inputs, allow_sealed=allow_sealed)
    needed = sorted({r["key"].report for r in first.values()})  # 변형의 바탕도 회차의 성적서다
    bases = (sealed_bases(vroot / SEALED, needed) if mode == "verify"
             else asyncio.run(dev_bases(guard(inputs or DEV_INPUTS, "dev"))))
    by_run = {r["run_id"]: r for r in first.values()}

    def table(columns, rows) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(columns)
        w.writerows(rows)
        return buf.getvalue()

    out.mkdir(parents=True, exist_ok=True)
    (out / "문서").mkdir(exist_ok=True)
    for label, (run_id, report_tag) in docs.items():
        variant = by_run[run_id]["key"].variant
        slots = variants[variant][1] if variant != R.BASE_VARIANT else None
        (out / "문서" / f"{label}.txt").write_text(doc_text(bases[report_tag], slots), encoding="utf-8")
    for who, rows in sheets.items():
        (out / f"채점지_{who}.csv").write_text(table(SH.SHEET_COLUMNS, rows), encoding="utf-8")
    checklist = criteria.load_json("checklist.json")["items"]
    (out / "필수내용_항목.md").write_text(
        f"# 필수 내용 항목 (기준 {criteria.version()})\n\n| 항목 | 독자가 알아야 할 것 |\n|---|---|\n"
        + "".join(f"| {i['id']} | {i['reader_learns']} |\n" for i in checklist), encoding="utf-8")
    key_dir = out / "열쇠"
    key_dir.mkdir(exist_ok=True)
    key_body = table(SH.KEY_COLUMNS, key)
    key_sha = hashlib.sha256(key_body.encode("utf-8")).hexdigest()
    (key_dir / "열쇠.csv").write_text(key_body, encoding="utf-8")
    (key_dir / "열쇠.sha256").write_text(f"{key_sha}  열쇠.csv\n", encoding="utf-8")
    strata: dict[str, int] = {}
    for item in assigned:
        strata[SH.stratum(item)] = strata.get(SH.stratum(item), 0) + 1
    draw = {"seed": seed, "n": n, "overlap_per_axis": overlap, "candidates": len(items), "drawn": len(assigned),
            "strata": dict(sorted(strata.items())), "per_scorer": {w: len(r) for w, r in sheets.items()}, "key_sha256": key_sha}
    (out / "뽑기.json").write_text(json.dumps(draw, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for f in (key_dir / "열쇠.csv", key_dir / "열쇠.sha256"):
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    echo(f"채점지: 인용 {len(sheets['인용'])}문항 · 민서 {len(sheets['민서'])}문항 · 문서 {len(docs)}개 · 시드 {seed} · 열쇠 sha256 {key_sha}")
    return draw


def cmd_reading(mode: str, *, out: Path | None, seed: int | None = None, defects: Path | None = None,
                inputs: Path | None = None, allow_sealed: bool = False, echo: Callable[[str], None] = print) -> dict:
    """8절 블라인드 읽기 묶음(P1) — 사람마다 읽을 문서(결함본은 심지 않은 사람에게만, 바탕·sham 은 둘 다)와 빈 읽기표, 열쇠(따로 + 해시)."""
    import csv
    import io
    import stat

    from app.judge.validation import sheets as SH

    variants, _, _, _ = _variant_inputs(mode, defects=defects, inputs=inputs, allow_sealed=allow_sealed)
    tags = sorted({m["base"] for m, _ in variants.values()})
    vroot = validation_root()
    bases = sealed_bases(vroot / SEALED, tags) if mode == "verify" else asyncio.run(dev_bases(guard(inputs or DEV_INPUTS, "dev")))
    out = Path(out or vroot / HUMAN / "읽기") if mode == "verify" else guard(out or Path(tempfile.mkdtemp(prefix="judge_reading_")), "dev")
    seed = random.SystemRandom().randrange(1, 10**9) if seed is None else seed
    lists, key = SH.build_reading(tags, {v: m for v, (m, _) in variants.items()}, seed)

    def table(columns, rows) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(columns)
        w.writerows(rows)
        return buf.getvalue()

    for reader, docs in lists.items():
        folder = out / reader
        (folder / "문서").mkdir(parents=True, exist_ok=True)
        for label, tag, vid in docs:
            slots = variants[vid][1] if vid != R.BASE_VARIANT else None
            (folder / "문서" / f"{label}.txt").write_text(doc_text(bases[tag], slots), encoding="utf-8")
        (folder / "읽기표.csv").write_text(table(SH.READING_COLUMNS, [(label, "", "", "") for label, _, _ in docs]
                                                + [("시작 시각", "", "", ""), ("끝 시각", "", "", "")]), encoding="utf-8")
    key_dir = out / "열쇠"
    key_dir.mkdir(parents=True, exist_ok=True)
    body = table(SH.READING_KEY_COLUMNS, key)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (key_dir / "읽기_열쇠.csv").write_text(body, encoding="utf-8")
    (key_dir / "읽기_열쇠.sha256").write_text(f"{sha}  읽기_열쇠.csv\n", encoding="utf-8")
    for f in key_dir.iterdir():
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    echo(f"읽기 묶음: " + " · ".join(f"{r} {len(d)}문서" for r, d in lists.items()) + f" · 시드 {seed} · 열쇠 sha256 {sha}")
    return {"seed": seed, "docs": {r: len(d) for r, d in lists.items()}, "key_sha256": sha}


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
    rep.add_argument("--answers", help="정답지 폴더(기본 verify 는 검증/01_정답지)")
    rep.add_argument("--human", help="사람 채점 폴더(기본 verify 는 검증/04_사람평가)")
    for name, helptext in (("check-defects", "결함본·sham 꼴 검사(verify 는 V1 뒤 --allow-sealed)"),
                           ("make-diff", "결함본·sham 의 diff.txt 를 도구로 쓴다")):
        c = sub.add_parser(name, help=helptext)
        c.add_argument("--mode", choices=MODES, required=True)
        c.add_argument("--defects", help="dev 전용 — 합성 결함 폴더")
        c.add_argument("--inputs", help="dev 전용 — 개발 입력 폴더")
        c.add_argument("--allow-sealed", action="store_true", help="봉인 서술 열람 허락(V1 뒤, 사람)")
        if name == "make-diff":
            c.add_argument("variant")
    rd = sub.add_parser("reading", help="8절 블라인드 읽기 묶음(P1)")
    rd.add_argument("--mode", choices=MODES, required=True)
    rd.add_argument("--out")
    rd.add_argument("--seed", type=int)
    rd.add_argument("--defects")
    rd.add_argument("--inputs")
    rd.add_argument("--allow-sealed", action="store_true")
    sh = sub.add_parser("sheets", help="사람 일치도 채점지·열쇠(층 나눈 표본, 시드 기록)")
    sh.add_argument("--mode", choices=MODES, required=True)
    sh.add_argument("--ledger")
    sh.add_argument("--out")
    sh.add_argument("--n", type=int, default=80, help="표본 크기(기본 80 — 줄인 안에 맞게 바꿀 수 있다)")
    sh.add_argument("--overlap", type=int, default=5, help="축마다 두 사람이 같이 채점할 문항 수")
    sh.add_argument("--seed", type=int)
    sh.add_argument("--defects")
    sh.add_argument("--inputs")
    sh.add_argument("--allow-sealed", action="store_true")
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
        if args.command in ("check-defects", "make-diff", "sheets", "reading") and args.mode == "verify" and (args.defects or args.inputs):
            raise RunnerError("verify 는 입력·결함을 검증 폴더에서만 읽는다(--inputs·--defects 는 dev 전용)")
        opt = lambda v: Path(v) if v else None  # noqa: E731
        if args.command == "check-defects":
            problems = cmd_check_defects(args.mode, defects=opt(args.defects), inputs=opt(args.inputs),
                                         allow_sealed=args.allow_sealed)
            return 1 if problems else 0
        if args.command == "make-diff":
            cmd_make_diff(args.mode, args.variant, defects=opt(args.defects), inputs=opt(args.inputs),
                          allow_sealed=args.allow_sealed)
            return 0
        if args.command == "reading":
            cmd_reading(args.mode, out=opt(args.out), seed=args.seed, defects=opt(args.defects), inputs=opt(args.inputs),
                        allow_sealed=args.allow_sealed)
            return 0
        if args.command == "sheets":
            cmd_sheets(args.mode, ledger=opt(args.ledger), out=opt(args.out), n=args.n, overlap=args.overlap, seed=args.seed,
                       defects=opt(args.defects), inputs=opt(args.inputs), allow_sealed=args.allow_sealed)
            return 0
        if args.command == "rescope":
            cmd_rescope(args.changed, args.since)
            return 0
        if args.command == "report":
            cmd_report(args.mode, ledger=Path(args.ledger) if args.ledger else None, out=Path(args.out) if args.out else None,
                       answers=Path(args.answers) if args.answers else None, human=Path(args.human) if args.human else None)
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
