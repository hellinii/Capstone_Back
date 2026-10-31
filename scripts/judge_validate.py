"""
judge_validate.py — 심판 검증 실행기(계획 17-3, 설계 7절). pytest 시험이 아니다.

두 모드
- dev    : 가짜 LLM(scripts/dev_fake_llm_server 의 가짜 OpenAI, 심판 '문제 없음' 답)으로 흐름만 돈다. 입력은 개발 판 4 입력
           (tests/data/judge_dev — 규칙 문장 서술을 키 없이 만든다)과 합성 결함·sham 뿐. 결과는 임시 폴더. 읽고 쓰는 경로는 허용 목록
           (개발 입력·임시 폴더·JUDGE_DEV_DIRS) 안이어야 하고, 검증 폴더(JUDGE_VALIDATION_DIR 와 저장소 옆 JUDGE_ROUND/검증 둘 다) 안이면
           어떤 철자로 적어도(scripts/judge_paths) 거부한다(오염 방지 규칙 2) — 입력·결함·장부·출력·정답지·사람 채점·결과 원본 경로.
- verify : 진짜 검증(V2). 아래 다섯이 모두 맞아야 시작한다 — 사유는 하나씩 따로 찍는다.
           ① 작업 트리가 깨끗하다(git status 빈칸)  ② 기준 버전이 고정 버전이고 기록된 지문과 같다(1.0-dev 면 거부)
           ③ 지금 MANIFEST == 태그 judge-criteria-v<지금 VERSION> 의 MANIFEST(파일 ↔ MANIFEST 도)  ④ --live  ⑤ 장부에 쓸 수 있다
           입력은 검증 폴더의 봉인 서술(00_봉인/<태그>.json — SHA256SUMS 와 맞아야), 결함·sham 은 02_결함/, 결과 원본은
           03_실행/<run_id>.json, 장부는 검증/장부.csv. 서술 글은 화면에 찍지 않는다(개수·해시·게이트만).

하위 명령(레포 루트에서)
- run <묶음> : reproducibility | defects | sham | extraction. 회차마다 장부 한 줄 + 결과 원본 파일 하나.
    - 이어 하기: 같은 모드·같은 기준 지문으로 결과 파일까지 적히고 게이트 판정이 난 회차는 건너뛴다(멈춘 뒤 같은 명령을
      다시 치면 이어 간다). 채점이 실패한 회차(gate 빈칸)는 다시 돈다 — 실패 줄의 비용은 장부 누계에 그대로 남는다.
    - 순서 섞기: 남은 회차를 시드로 섞는다(--seed, 없으면 새로 뽑아 찍음). 시드·순서는 plan-<묶음>-<시드>.json 과 결과 원본에.
    - 기본 회차(V1 결정 1·2, 2026-10-29): 재현성 3건(bc_strong·digits_weak·yeast_strong — verify 에서 --reports 를 안 주면
      이것) × 2회, 결함 3개 × 1회, sham 1 → 채점 10회 ≈ $1.5. 시작 때 V2 전체 계획·이번 묶음 예상 회차·비용(채점 1회 $0.15
      실측)·장부 누계를 찍고 --yes 가 없으면 멈춘다. 묶음 회차 상한(기본 = 위 회차)을 넘거나 라운드 남은 상한
      (JUDGE_LIVE_BUDGET_USD·JUDGE_LIVE_MAX_RUNS, 장부 누계 포함 — V2 승인 $4.5·40회)보다 크면 시작하지 않는다. 돌다가 묶음 예산
      (예상 × 1.5)을 넘으면 멈춘다. 누계 $10 단위·비용 모름 20회마다 멈춰 확인(scripts/judge_live.py 의 Runner 그대로).
    - extraction 은 채점을 부르지 않는다(재현성 회차의 추출 기록을 report 가 정답지와 짝짓는다).
- report     : 장부·결과 원본으로 7.6 판정값을 계산한다. 계산은 app.judge.validation 의 stats·extraction·agreement 가 하고,
               이 명령은 그 함수들을 부르는 얇은 층이다(STATS_API). 결과는 verify 면 05_결과/검증결과.json, dev 면 출력 폴더.
               정답지가 있는 성적서마다 사람이 놓친 주장 판정표(놓친주장_<태그>.csv — 시스템만 찾은 주장, 회차를 합쳐 한 줄씩)를
               같은 출력 폴더에 쓴다. 사람이 판정해 정답지 폴더에 <태그>.missed.csv 로 두면 "예" 줄을 보탤 목록
               (보탤주장_<태그>.csv, claims.csv 꼴)으로 내고, 그 줄을 정답지에 보태 추출 재현율·정밀도를 다시 잰다(V1 결정
               10④ — 검증결과.json 에 보태기 전 extraction.pooled 와 뒤 extraction.pooled_with_additions 둘 다). 정답지
               파일 자체는 고치지 않는다.
               판정 기준(V1 결정 4~7): 게이트 재현성 = 3건 모두 2회 같은 판정, 축 ICC 0.75, 결함 탐지 = 결함 모두 1회에 탐지
               (계획 개수보다 적게 돌았으면 계산 불가), 독립성·sham 여유 = max(흔들림 폭, 건수 1건·비율 0.05), 사람 일치 여유 =
               한 문항(1 / 겹침 문항 수).
- check-defects : 결함본·sham 폴더 꼴 검사(meta 칸·심은 사람 역할·칸 하나·문장 하나·before/after·diff.txt). verify 는 V1 뒤
               --allow-sealed 로만(봉인 서술을 연다), dev 는 합성 바탕(개발 입력)으로.
- make-diff <D01> : diff.txt 를 도구로 쓴다.
- sheets     : 사람 일치도 채점지(블라인드 — 시스템 판정·결함 여부 없음, 결함본 문항은 7.2 역할대로 인용/민서), 열쇠(따로 + 해시,
               읽기 전용), 읽을 문서(문서 번호만). 층 나눈 표본(--n 기본 40 = 축마다 10, --overlap 겹침 모두 10 — 축마다
               고르게 3·3·2·2, --seed 기록. V1 결정 7).
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
import judge_paths  # noqa: E402

from app.judge import criteria  # noqa: E402
from app.judge.validation import annotation as A  # noqa: E402
from app.judge.validation import runs as R  # noqa: E402
from app.judge.validation.extraction import MISSED_COLUMNS  # noqa: E402 — 표 열 이름(계산 함수는 STATS_API 로)
from app.judge.validation import variants as V  # noqa: E402

RunnerError = jl.RunnerError
MODES = ("dev", "verify")
TAG_PREFIX = "judge-criteria-v"


def criteria_tag() -> str:
    """지금 기준 버전의 고정 태그(1.1 부터 — 1.0 이름을 박아 두어 1.1 검증이 시작되지 않았다)."""
    return f"{TAG_PREFIX}{criteria.version()}"


MANIFEST_IN_REPO = "app/judge/criteria/MANIFEST.sha256"
DEV_INPUTS = ROOT / "tests" / "data" / "judge_dev"
SEALED = "00_봉인"
DEFECTS = "02_결함"
RUNS_OUT = "03_실행"
RESULTS_OUT = "05_결과"
ANSWERS = "01_정답지"
HUMAN = "04_사람평가"


# ── 읽기 경계 ─────────────────────────────────────────────────────────


DEFAULT_VALIDATION = (ROOT.parent / "JUDGE_ROUND" / "검증").resolve()
DEV_DIRS_ENV = "JUDGE_DEV_DIRS"  # dev 가 더 읽고 쓸 폴더(os.pathsep 로 여럿) — 허용 목록에 보탠다


def validation_root() -> Path:
    """검증 폴더 — 환경변수 JUDGE_VALIDATION_DIR, 없으면 저장소 옆 JUDGE_ROUND/검증(없어도 경로로만 쓴다)."""
    return jl.validation_dir() or DEFAULT_VALIDATION


def dev_allowed_roots() -> list[Path]:
    """dev 허용 목록 — 개발 입력(tests/data/judge_dev), 임시 폴더(시스템 임시·/tmp), JUDGE_DEV_DIRS 로 적은 폴더."""
    extra = [Path(p) for p in os.getenv(DEV_DIRS_ENV, "").split(os.pathsep) if p.strip()]
    return [DEV_INPUTS, Path(tempfile.gettempdir()), Path("/tmp"), *extra]


def guard(path: Path | str, mode: str) -> Path:
    """dev 모드 읽기·쓰기 경계 — 열기 전에 거부한다. ① 검증 폴더(환경변수 값과 저장소 옆 기본 자리 둘 다) 안이면 거부,
    ② 허용 목록(dev_allowed_roots) 밖이면 거부. 비교는 judge_paths(resolve·NFC·casefold·samefile — 철자 우회 막기)."""
    resolved = Path(path).resolve()
    if mode == "dev":
        for vdir in (validation_root(), DEFAULT_VALIDATION):
            if judge_paths.inside(resolved, vdir):
                raise RunnerError(f"dev 모드는 검증 폴더를 읽거나 쓰지 않는다: {resolved}")
        if not any(judge_paths.inside(resolved, root) for root in dev_allowed_roots()):
            raise RunnerError(f"dev 모드는 허용 목록 밖을 읽거나 쓰지 않는다: {resolved} — 개발 입력·임시 폴더·{DEV_DIRS_ENV}")
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
    tag = criteria_tag()
    code, tagged = git("show", f"{tag}:{MANIFEST_IN_REPO}")
    if code != 0:
        problems.append(f"태그 {tag} 가 없다 — 기준 고정 뒤 태그를 붙인다(정8)")
    elif tagged != criteria.recorded_manifest() or criteria.recorded_manifest() != criteria.manifest_text():
        problems.append(f"지금 기준 묶음 MANIFEST 가 태그 {tag} 의 MANIFEST 와 다르다 — 고정한 기준으로만 검증한다(7.4)")
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
    ctx.echo(f"[{ctx.mode}] {R.plan_summary()}")
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


def base_problems(rows: list[dict], tags: list[str], *, fingerprint: str, mode: str, repeats: int) -> list[str]:
    """결함·sham 을 돌리기 전 조건(Q11 — 실행기 첫 단계에서 바탕이 게이트를 통과하는지): 바탕 성적서마다 재현성 회차가
    계획 회차(repeats)만큼 끝났고(같은 모드·지문) 그 회차가 모두 pass. 흔들림 폭을 잴 바탕이 없으면 독립성·sham 은 셀 수 없다."""
    done: dict[str, dict[int, str]] = {}
    for row in rows:
        key = R.finished_row(row, fingerprint=fingerprint, mode=mode)
        if key is not None and key.batch == "reproducibility":
            done.setdefault(key.report, {})[key.repeat] = row["gate"]  # 같은 회차가 여러 줄이면 마지막 성공 줄
    out = []
    for tag in sorted(set(tags)):
        gates = done.get(tag, {})
        if len(gates) < repeats:
            out.append(f"바탕 {tag}: 재현성 회차가 {len(gates)}/{repeats}회만 끝났다 — 재현성 묶음을 먼저 다 돌린다")
        elif any(g != "pass" for g in gates.values()):
            out.append(f"바탕 {tag}: 재현성 회차가 모두 pass 가 아니다({sorted(gates.values())}) — 게이트를 통과한 바탕에만 결함을 심는다(Q11)")
    return out


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
    base_repeats: int | None = None  # 결함·sham 의 바탕 확인에 쓰는 재현성 계획 회차(기본 runs.DEFAULTS)
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
        reports = a.reports or (list(R.DEFAULT_REPORTS) if a.batch == "reproducibility" else [])  # V1 결정 2
        needed = sorted(set(reports) | {m.get("base") for m, _ in variants.values()})
        bases = sealed_bases(inputs, needed)
        if client is None:
            client = jl.make_clients(live=True).openai
            if client is None:
                raise RunnerError("OPENAI_API_KEY 가 없다")
    diffs = {vid: ((defects / vid / "diff.txt").read_text(encoding="utf-8") if (defects / vid / "diff.txt").exists() else None)
             for vid in variants}
    bad = variant_problems(variants, bases, diffs)  # check-defects 와 같은 검사(diff.txt 까지)
    if bad:
        raise RunnerError("결함본·sham 꼴이 틀려 시작하지 않는다:\n  - " + "\n  - ".join(bad))
    if a.batch in ("defects", "sham"):
        prefix = "D" if a.batch == "defects" else "S"
        tags = [m.get("base") for vid, (m, _) in variants.items() if vid.startswith(prefix)]
        rows = jl.Ledger(ledger).rows() if Path(ledger).exists() else []
        bad = base_problems(rows, tags, fingerprint=criteria.fingerprint(), mode="live" if a.mode == "verify" else "dev",
                            repeats=a.base_repeats or R.DEFAULTS["reproducibility"].repeats)
        if bad:
            raise RunnerError("바탕 성적서가 준비되지 않아 시작하지 않는다:\n  - " + "\n  - ".join(bad))
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
    "missed_rows": "extraction", "confirmed_additions": "extraction",
    "cohen_kappa": "agreement", "gwet_ac1": "agreement",
}


def load_stats_api() -> dict[str, Callable]:
    return {name: getattr(importlib.import_module(f"app.judge.validation.{mod}"), name) for name, mod in STATS_API.items()}


def variant_axes() -> tuple[str, ...]:
    from app.judge.validation.variants import AXES

    return AXES


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
        if mode == "dev":
            path = guard(path, "dev")  # 장부가 가리키는 결과 원본도 dev 경계 안이어야(장부를 고쳐 검증 폴더를 읽히지 않게)
        body = path.read_bytes()
        if row.get("result_sha256") and hashlib.sha256(body).hexdigest() != row["result_sha256"]:
            raise RunnerError(f"결과 원본 해시가 장부와 다르다: {path}")
        runs.append({**json.loads(body), "key": key})  # 열쇠는 장부의 RunKey(결과 원본의 key 는 dict)
    return runs


def _tri(values) -> bool | None:
    """세 값 묶기(stats.tri 와 같은 규칙 — 계산 모듈을 report 배선 시험에서 가짜로 갈아 끼우므로 여기 따로 둔다).
    값 None(계산 불가 칸)은 None 으로, dict 없는 칸(False·None)은 그대로."""
    values = [None if v is None else bool(v) for v in values]
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def planted_sentence(meta: dict) -> str | None:
    """탐지를 볼 심은 문장 — 바꾸거나 끼운 문장(after), 지우기면 지운 문장(before)."""
    return meta.get("after") or meta.get("before") or None


UNCOMPUTABLE = "계산 불가"
EXTRACTION_MIN = 0.95  # V1 결정 11 — 재현율·정밀도 모두, 3건 × 회차를 합친 값·놓친 주장을 보탠 뒤로 판정


def extraction_passed(ext: dict) -> bool | None:
    """추출 판정(V1 결정 11) — 보태기 전 값이 아니라 놓친 주장을 보탠 뒤 합친 값의 재현율·정밀도가 모두 기준 이상."""
    judged_on = ext["pooled_with_additions"]
    if judged_on["recall"] is None or judged_on["precision"] is None:
        return None
    return judged_on["recall"] >= EXTRACTION_MIN and judged_on["precision"] >= EXTRACTION_MIN


def verdict_word(value: bool | None) -> str:
    return {True: "통과", False: "미달", None: UNCOMPUTABLE}[value]


def merge_missed(per_run: list[list[tuple]]) -> list[tuple]:
    """회차마다의 놓친 주장 줄(extraction.missed_rows) → 한 표. 같은 문장·칸·종류·인용(공백 정리)은 한 줄, 번호는 1부터 다시."""
    seen: dict[tuple, tuple] = {}
    for rows in per_run:
        for row in rows:
            key = (str(row[1]), row[2], row[3], " ".join(str(row[4]).split()))
            seen.setdefault(key, row)
    return [(i, *row[1:]) for i, row in enumerate(seen.values(), 1)]


def report(runs: list[dict], api: dict[str, Callable], *, human_claims: dict[str, tuple[list, list]] | None = None,
           human_scores: dict | None = None, human_reading: dict | None = None,
           human_missed: dict[str, list[dict]] | None = None, plan: dict[str, R.BatchDefault] | None = None) -> dict:
    """7.6 판정값(+ 8절 효용성 — human_reading = sheets.reading_hits 결과). runs = collect_runs 결과(판정은 장부 순서가 아니라 회차 번호 순).
    human_claims = {성적서: (claims.csv 줄, sentences.csv 줄)}, human_scores = sheets.human_pairs 결과,
    human_missed = {성적서: 사람이 판정한 놓친 주장 표 줄}(Q12② — "예" 줄만 보탤 목록으로).
    plan = 묶음별 계획 회차(기본 runs.DEFAULTS — V1 확정값은 인자로). 회차 수는 관측값이 아니라 이 계획값으로 맞춘다.
    판정값은 세 값이다: True 통과 · False 미달 · None 계산 불가(덜 돎·바탕 미통과 등 — 이유는 그 칸의 note). verdicts 에 글로 모은다."""
    plan = {**R.DEFAULTS, **(plan or {})}
    by_batch: dict[str, list[dict]] = {}
    for r in sorted(runs, key=lambda r: r["key"]):
        by_batch.setdefault(r["key"].batch, []).append(r)
    out: dict = {"runs": len(runs)}

    base_runs: dict[str, list[dict]] = {}
    for r in by_batch.get("reproducibility", []):
        base_runs.setdefault(r["key"].report, []).append(r)
    want = plan["reproducibility"]
    if base_runs:
        counts = {tag: len(v) for tag, v in sorted(base_runs.items())}
        if len(counts) < want.items or any(c != want.repeats for c in counts.values()):
            # 덜 돈 묶음(성적서·회차가 계획보다 적거나 섞임) — 통계를 부르지 않고 계산 불가(미달과 따로)
            out["reproducibility"] = {"verdict": UNCOMPUTABLE, "axes_passed": None, "gates_passed": None,
                                      "note": f"덜 돎 — 계획 {want.items}건 × {want.repeats}회, 돈 회차 {counts}"}
            out["gates_passed"] = None
        else:
            repro = api["reproducibility"](base_runs, expected_runs=want.repeats)
            out["reproducibility"] = repro
            out["gates_passed"] = repro["gates_passed"]
        out["base_gate_pass"] = {tag: all(r.get("gate") == "pass" for r in rs) for tag, rs in base_runs.items()}  # Q11
    base_ok = {tag: out.get("gates_passed") is not None and len(rs) == want.repeats and all(r.get("gate") == "pass" for r in rs)
               for tag, rs in base_runs.items()}  # 흔들림 폭을 잴 수 있는 바탕(다 돌았고 모두 통과)

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
        ok_base = base_ok.get(rs[0]["key"].report, False)
        defects[vid] = {"base": rs[0]["key"].report, "axis": axis, "hits": hits, "scores": [api["axis_scores"](r) for r in rs],
                        "independence": api["independence_ok"](base, rs, axis) if ok_base else None}
        if not ok_base:
            defects[vid]["note"] = f"{UNCOMPUTABLE} — 바탕 {rs[0]['key'].report} 의 재현성 회차가 다 돌지 않았거나 모두 통과가 아니다"
    if defects:
        out["defects"] = defects
        out["detection"] = api["detection_passed"]({v: {"axis": d["axis"], "hits": d["hits"]} for v, d in defects.items()})
        want_d = plan["defects"]
        short = [v for v, d in defects.items() if len(d["hits"]) != want_d.repeats]
        if len(defects) < want_d.items or short:
            # V1 결정 4 는 '결함 모두'다 — 계획보다 적게 돌았으면 통과로 세지 않는다(계산 불가, 미달과 따로)
            out["detection"] = {**out["detection"], "passed": None,
                                "note": f"덜 돎 — 계획 결함 {want_d.items}개 × {want_d.repeats}회, 돈 결함 "
                                        f"{ {v: len(d['hits']) for v, d in defects.items()} }"}
        out["independence_passed"] = _tri(d["independence"] and d["independence"]["ok"] for d in defects.values())
    shams = {}
    for vid, rs in sorted(grouped("sham").items()):
        base = base_runs.get(rs[0]["key"].report, [])
        ok_base = base_ok.get(rs[0]["key"].report, False)
        shams[vid] = {"base": rs[0]["key"].report, "sham": api["sham_ok"](base, rs) if ok_base else None}
        if not ok_base:
            shams[vid]["note"] = f"{UNCOMPUTABLE} — 바탕 {rs[0]['key'].report} 의 재현성 회차가 다 돌지 않았거나 모두 통과가 아니다"
    if shams:
        out["sham"] = shams
        out["sham_passed"] = _tri(s["sham"] and s["sham"]["ok"] for s in shams.values())

    if human_claims:
        extraction, missed, pairs, gold_n, pred_n = {}, {}, 0, 0, 0
        pairs2, gold_n2, pred_n2, added = 0, 0, 0, 0  # 놓친 주장을 보탠 뒤(V1 결정 10④)
        for tag, (claim_rows, sentence_rows) in sorted(human_claims.items()):
            if tag not in base_runs:
                continue
            gold = api["gold_from_csv"](claim_rows, sentence_rows)
            preds = [api["pred_from_extract"](((r.get("record") or {}).get("extract") or {}).get("claims") or [], sentence_rows)
                     for r in base_runs[tag]]
            matches = [api["match_claims"](gold, p) for p in preds]
            per_run = [api["recall_precision"](m) for m in matches]
            for m in per_run:
                pairs, gold_n, pred_n = pairs + m["n_pairs"], gold_n + m["n_gold"], pred_n + m["n_pred"]
            extraction[tag] = {"runs": per_run, "distribution": api["runs_distribution"](gold, preds)}
            # 사람이 놓친 주장(Q12②) — 시스템만 찾은 주장을 판정표로. 판정한 표가 있으면 "예" 줄을 보탤 목록으로.
            judged = (human_missed or {}).get(tag)
            try:
                additions = api["confirmed_additions"](judged) if judged else []
            except ValueError as exc:
                raise RunnerError(f"{tag} 놓친 주장 판정표: {exc}") from exc
            missed[tag] = {"rows": merge_missed([api["missed_rows"](m) for m in matches]), "judged": bool(judged),
                           "additions": additions}
            # V1 결정 10④ — "예" 줄을 정답지에 보태 다시 잰다(원 정답지 파일은 그대로, 보탠 줄은 보탤주장_<태그>.csv)
            if additions:
                gold2 = api["gold_from_csv"]([*claim_rows, *additions], sentence_rows)
                per_run2 = [api["recall_precision"](api["match_claims"](gold2, p)) for p in preds]
                extraction[tag]["with_additions"] = {"added": len(additions), "runs": per_run2,
                                                     "distribution": api["runs_distribution"](gold2, preds)}
            else:
                per_run2 = per_run
                extraction[tag]["with_additions"] = {"added": 0, "runs": per_run, "distribution": extraction[tag]["distribution"]}
            for m in per_run2:
                pairs2, gold_n2, pred_n2 = pairs2 + m["n_pairs"], gold_n2 + m["n_gold"], pred_n2 + m["n_pred"]
            added += len(additions)
        out["extraction"] = {"reports": extraction, "pooled": {
            "recall": pairs / gold_n if gold_n else None, "precision": pairs / pred_n if pred_n else None,
            "n_pairs": pairs, "n_gold": gold_n, "n_pred": pred_n}, "pooled_with_additions": {
            "recall": pairs2 / gold_n2 if gold_n2 else None, "precision": pairs2 / pred_n2 if pred_n2 else None,
            "n_pairs": pairs2, "n_gold": gold_n2, "n_pred": pred_n2, "added": added}}
        out["extraction"]["passed"] = extraction_passed(out["extraction"])
        if missed:
            out["missed_claims"] = missed

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
                            # 여유 = 한 문항(1 / 겹침 문항 수 — V1 결정 7)
                            "passed": api["human_agreement_ok"](all_sys["rate"], all_hh["rate"], overlap_n=all_hh["n"])
                            if all_sys and all_hh else None}
    out["verdicts"] = {name: verdict_word(value) for name, value in (
        ("gates_passed", out.get("gates_passed", None)),
        ("axes_passed", (out.get("reproducibility") or {}).get("axes_passed")),
        ("detection", (out.get("detection") or {}).get("passed")) if "detection" in out else ("detection", None),
        ("independence_passed", out.get("independence_passed")), ("sham_passed", out.get("sham_passed")),
        ("extraction", (out.get("extraction") or {}).get("passed")),
        ("agreement", (out.get("agreement") or {}).get("passed")))}
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


def read_human_missed(folder: Path, tags: list[str]) -> dict[str, list[dict]]:
    """사람이 판정한 놓친 주장 표(정답지 폴더의 <태그>.missed.csv — report 가 쓴 놓친주장_<태그>.csv 를 판정해 옮긴 것)."""
    found = {}
    for tag in tags:
        path = Path(folder) / f"{tag}.missed.csv"
        if path.exists():
            rows = _csv_rows(path)
            if rows:
                found[tag] = rows
    return found


def write_missed(out: Path, missed: dict[str, dict]) -> list[Path]:
    """놓친 주장 판정표(놓친주장_<태그>.csv)와, 판정한 줄이 있으면 보탤 목록(보탤주장_<태그>.csv)을 출력 폴더에 쓴다."""
    written = []
    for tag, m in sorted(missed.items()):
        path = Path(out) / f"놓친주장_{tag}.csv"
        path.write_text(A.to_csv(MISSED_COLUMNS, m["rows"]), encoding="utf-8")
        written.append(path)
        if m["additions"]:
            path = Path(out) / f"보탤주장_{tag}.csv"
            path.write_text(A.to_csv(A.CLAIM_COLUMNS, [[row[c] for c in A.CLAIM_COLUMNS] for row in m["additions"]]),
                            encoding="utf-8")
            written.append(path)
    return written


def read_human_scores(folder: Path) -> dict | None:
    """사람 채점(04_사람평가) — 채점지 두 장과 열쇠가 있고 판정이 하나라도 적혔을 때만 짝으로."""
    from app.judge.validation import sheets as SH

    key = Path(folder) / "열쇠" / "열쇠.csv"
    sheets = {who: Path(folder) / f"채점지_{who}.csv" for who in ("인용", "민서")}
    if not key.exists() or not all(p.exists() for p in sheets.values()):
        return None
    read = {who: read_keyed_sheet(p) for who, p in sheets.items()}
    filled = {who: rows for who, (_, rows) in read.items()}
    if not any((r.get(SH.SHEET_COLUMNS[6]) or "").strip() for rows in filled.values() for r in rows):
        return None
    draw = Path(folder) / "뽑기.json"
    check_key(key, {f"채점지_{who}": sha for who, (sha, _) in read.items()},
              recorded={"열쇠.sha256": _sha_file(key.parent / "열쇠.sha256"),
                        "뽑기.json": json.loads(draw.read_text(encoding="utf-8")).get("key_sha256") if draw.exists() else None})
    return SH.human_pairs(_csv_rows(key), filled)


def read_human_reading(folder: Path) -> dict | None:
    """8절 읽기 결과(04_사람평가/읽기) — 열쇠와 읽기표가 있고 하나라도 적혔을 때만."""
    from app.judge.validation import sheets as SH

    key = Path(folder) / "읽기" / "열쇠" / "읽기_열쇠.csv"
    sheets = [Path(folder) / "읽기" / r / "읽기표.csv" for r in SH.READERS]
    if not key.exists() or not all(p.exists() for p in sheets):
        return None
    read = {p.parent.name: read_keyed_sheet(p) for p in sheets}
    filled = [r for _, rows in read.values() for r in rows if r["문서"].startswith("읽기-")]
    if not any((r.get(SH.READING_COLUMNS[1]) or "").strip() or (r.get("메모") or "").strip() for r in filled):
        return None
    check_key(key, {f"읽기표_{who}": sha for who, (sha, _) in read.items()},
              recorded={"읽기_열쇠.sha256": _sha_file(key.parent / "읽기_열쇠.sha256")})
    from app.judge.validation.stats import touches  # 결함 탐지와 같은 '짚었나' 규칙

    return SH.reading_hits(_csv_rows(key), filled, touches)


def cmd_report(mode: str, *, ledger: Path | None, out: Path | None, api: dict[str, Callable] | None = None,
               answers: Path | None = None, human: Path | None = None, echo: Callable[[str], None] = print,
               plan: dict[str, R.BatchDefault] | None = None) -> dict:
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
    judged_missed = read_human_missed(answers, sorted(claims)) if claims else {}
    scores = read_human_scores(human) if human and Path(human).is_dir() else None
    reading = read_human_reading(human) if human and Path(human).is_dir() else None
    result = report(runs, api or load_stats_api(), human_claims=claims, human_scores=scores, human_reading=reading,
                    human_missed=judged_missed, plan=plan)
    result.update({"mode": mode, "criteria_version": criteria.version(), "fingerprint": criteria.fingerprint(), "tag": criteria_tag()})
    out.mkdir(parents=True, exist_ok=True)
    for path in write_missed(out, result.get("missed_claims") or {}):
        echo(f"놓친 주장 표 {path}")
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


KEY_HEAD = "# 열쇠 sha256: "  # 채점지·읽기표 첫 줄 — 어느 열쇠와 짝인지(report 가 맞대 본다)


def refuse_nonempty(out: Path, *, ignore: tuple[str, ...] = ()) -> None:
    """출력 폴더가 비어 있지 않으면 시작 전에 거부한다 — 채점지·열쇠를 덮어쓰지 않는다(이미 나눠 준 표와 열쇠가 어긋나지 않게)."""
    if Path(out).exists():
        if not Path(out).is_dir():
            raise RunnerError(f"출력 자리가 폴더가 아니다: {out}")
        left = sorted(p.name for p in Path(out).iterdir() if p.name not in ignore)
        if left:
            raise RunnerError(f"출력 폴더가 비어 있지 않다(덮어쓰지 않는다 — 새 폴더를 준다): {out} — {', '.join(left[:5])}")


def read_keyed_sheet(path: Path) -> tuple[str | None, list[dict]]:
    """머리 줄(KEY_HEAD)이 있는 채점지·읽기표 → (열쇠 해시, 줄)."""
    import csv
    import io

    text = Path(path).read_text(encoding="utf-8")
    first, _, rest = text.partition("\n")
    sha = first[len(KEY_HEAD):].strip() if first.startswith(KEY_HEAD) else None
    body = rest if sha is not None else text
    rows = [r for r in csv.DictReader(io.StringIO(body)) if any((v or "").strip() for v in r.values())]
    return sha, rows


def check_key(key_path: Path, sheet_shas: dict[str, str | None], *, recorded: dict[str, str | None]) -> str:
    """열쇠 파일의 실제 해시가 채점지 머리·기록(열쇠.sha256·뽑기.json)과 모두 같은가. 다르면 거부(사유 문장)."""
    actual = hashlib.sha256(Path(key_path).read_bytes()).hexdigest()
    bad = [f"{name}({sha or '없음'})" for name, sha in {**sheet_shas, **recorded}.items() if sha != actual]
    if bad:
        raise RunnerError(f"열쇠 {Path(key_path).name}(sha256 {actual[:12]}…)가 짝과 맞지 않는다 — {', '.join(bad)}. "
                          "다른 묶음의 채점지·열쇠를 섞지 않는다")
    return actual


def _sha_file(path: Path) -> str | None:
    return Path(path).read_text(encoding="utf-8").split()[0] if Path(path).exists() else None


def doc_text(base: Base, slots: dict | None, label: str) -> str:
    """사람이 읽을 문서 — 심판이 읽는 것과 같은 조각(서술 칸 + 고정 문구 + 표). 머리는 문서 번호뿐이다 — 결함 여부·변형 이름·
    바탕 성적서 태그는 없다(태그가 보이면 같은 성적서의 두 판본을 맞대 볼 수 있다, J10 검토)."""
    from app.judge.material import parse_material
    from app.judge.reading import build_pieces, render
    from app.narrative.baselines import build_benchmark_refs
    from app.narrative.schemas import NarrativeJobResult, NarrativeRequest

    req = NarrativeRequest.model_validate(base.request)
    result = NarrativeJobResult.model_validate(V.apply_slots(base.result, slots) if slots else base.result)
    material, _ = parse_material(req.judge_material)
    refs = build_benchmark_refs(req.task_type.value, req.fact_sheet.metrics, edition=4)
    pieces = build_pieces(result, result.section_sources, material, refs, req.fact_sheet.metrics)
    return f"# {label}\n\n" + render(pieces) + "\n"


SHEETS_N = 40  # V1 결정 7 — 사람 채점 40문항(축마다 10)
SHEETS_OVERLAP = 10  # 그중 두 사람이 같이 채점할 문항(모두, 축마다 고르게)


def cmd_sheets(mode: str, *, ledger: Path | None, out: Path | None, n: int = SHEETS_N, overlap: int = SHEETS_OVERLAP,
               seed: int | None = None,
               defects: Path | None = None, inputs: Path | None = None, allow_sealed: bool = False,
               echo: Callable[[str], None] = print) -> dict:
    """사람 일치도 채점지(블라인드, 역할대로)·열쇠(따로 + 해시)·읽을 문서. 표본은 검증 회차 결과에서 층 나눠 뽑는다(시드 기록).
    채점지 첫 줄과 뽑기.json 에 열쇠 해시 — report 가 열쇠 파일과 맞대 본다. 출력 폴더가 비어 있지 않으면 시작하지 않는다."""
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
    refuse_nonempty(out, ignore=("읽기",))  # 읽기 묶음(cmd_reading 기본 자리 04_사람평가/읽기)은 먼저 만들어 둘 수 있다
    runs = collect_runs(jl.Ledger(ledger).rows(), fingerprint=criteria.fingerprint(),
                        mode="live" if mode == "verify" else "dev", root=root)
    # 같은 성적서·변형은 첫 회차만(카-4: 판정은 1회째로 본다 — V1 에서 바꾸면 여기)
    first = {}
    for r in sorted(runs, key=lambda r: r["key"]):
        first.setdefault((r["key"].report, r["key"].variant), r)
    types = criteria.load_json("contradictions.json")["types"]
    items = [i for r in first.values() for i in SH.items_from_run({**r, "key": r["key"].__dict__}, types)]
    seed = random.SystemRandom().randrange(1, 10**9) if seed is None else seed
    assigned = SH.assign(SH.sample(items, n, seed), overlap=overlap, seed=seed)
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

    key_body = table(SH.KEY_COLUMNS, key)
    key_sha = hashlib.sha256(key_body.encode("utf-8")).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    (out / "문서").mkdir(exist_ok=True)
    for label, (run_id, report_tag) in docs.items():
        variant = by_run[run_id]["key"].variant
        slots = variants[variant][1] if variant != R.BASE_VARIANT else None
        (out / "문서" / f"{label}.txt").write_text(doc_text(bases[report_tag], slots, label), encoding="utf-8")
    for who, rows in sheets.items():
        (out / f"채점지_{who}.csv").write_text(KEY_HEAD + key_sha + "\n" + table(SH.SHEET_COLUMNS, rows), encoding="utf-8")
    checklist = criteria.load_json("checklist.json")["items"]
    (out / "필수내용_항목.md").write_text(
        f"# 필수 내용 항목 (기준 {criteria.version()})\n\n| 항목 | 독자가 알아야 할 것 |\n|---|---|\n"
        + "".join(f"| {i['id']} | {i['reader_learns']} |\n" for i in checklist), encoding="utf-8")
    key_dir = out / "열쇠"
    key_dir.mkdir(exist_ok=True)
    (key_dir / "열쇠.csv").write_text(key_body, encoding="utf-8")
    (key_dir / "열쇠.sha256").write_text(f"{key_sha}  열쇠.csv\n", encoding="utf-8")
    strata: dict[str, int] = {}
    for item in assigned:
        strata[SH.stratum(item)] = strata.get(SH.stratum(item), 0) + 1
    both = [i for i in assigned if len(i["scorers"]) == 2]
    draw = {"seed": seed, "n": n, "overlap": overlap, "overlap_drawn": len(both),
            "overlap_by_axis": {a: sum(i["axis"] == a for i in both) for a in variant_axes()},
            "candidates": len(items), "drawn": len(assigned),
            "strata": dict(sorted(strata.items())), "per_scorer": {w: len(r) for w, r in sheets.items()}, "key_sha256": key_sha}
    (out / "뽑기.json").write_text(json.dumps(draw, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for f in (key_dir / "열쇠.csv", key_dir / "열쇠.sha256"):
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    echo(f"채점지: 인용 {len(sheets['인용'])}문항 · 민서 {len(sheets['민서'])}문항 · 문서 {len(docs)}개 · 시드 {seed} · 열쇠 sha256 {key_sha}")
    return draw


def cmd_reading(mode: str, *, out: Path | None, seed: int | None = None, defects: Path | None = None,
                inputs: Path | None = None, allow_sealed: bool = False, echo: Callable[[str], None] = print) -> dict:
    """8절 블라인드 읽기 묶음(P1) — 사람마다 읽을 문서(결함본은 심지 않은 사람에게만, 한 사람에게 한 성적서의 한 판본만 —
    sheets.build_reading)와 빈 읽기표(머리에 열쇠 해시), 열쇠(따로 + 해시). 출력 폴더가 비어 있지 않으면 시작하지 않는다."""
    import csv
    import io
    import stat

    from app.judge.validation import sheets as SH

    variants, _, _, _ = _variant_inputs(mode, defects=defects, inputs=inputs, allow_sealed=allow_sealed)
    tags = sorted({m["base"] for m, _ in variants.values()})
    vroot = validation_root()
    bases = sealed_bases(vroot / SEALED, tags) if mode == "verify" else asyncio.run(dev_bases(guard(inputs or DEV_INPUTS, "dev")))
    out = Path(out or vroot / HUMAN / "읽기") if mode == "verify" else guard(out or Path(tempfile.mkdtemp(prefix="judge_reading_")), "dev")
    refuse_nonempty(out)
    seed = random.SystemRandom().randrange(1, 10**9) if seed is None else seed
    lists, key, unplaced = SH.build_reading(tags, {v: m for v, (m, _) in variants.items()}, seed)

    def table(columns, rows) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(columns)
        w.writerows(rows)
        return buf.getvalue()

    body = table(SH.READING_KEY_COLUMNS, key)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    for reader, docs in lists.items():
        folder = out / reader
        (folder / "문서").mkdir(parents=True, exist_ok=True)
        for label, tag, vid in docs:
            slots = variants[vid][1] if vid != R.BASE_VARIANT else None
            (folder / "문서" / f"{label}.txt").write_text(doc_text(bases[tag], slots, label), encoding="utf-8")
        (folder / "읽기표.csv").write_text(KEY_HEAD + sha + "\n" + table(SH.READING_COLUMNS, [(label, "", "", "") for label, _, _ in docs]
                                                + [("시작 시각", "", "", ""), ("끝 시각", "", "", "")]), encoding="utf-8")
    key_dir = out / "열쇠"
    key_dir.mkdir(parents=True, exist_ok=True)
    (key_dir / "읽기_열쇠.csv").write_text(body, encoding="utf-8")
    (key_dir / "읽기_열쇠.sha256").write_text(f"{sha}  읽기_열쇠.csv\n", encoding="utf-8")
    for f in key_dir.iterdir():
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    echo(f"읽기 묶음: " + " · ".join(f"{r} {len(d)}문서" for r, d in lists.items()) + f" · 시드 {seed} · 열쇠 sha256 {sha}")
    if unplaced:  # 한 사람에게 한 성적서의 한 판본만 — 자리가 없는 결함본은 이번 묶음에서 읽지 않는다(효용성 표의 '읽은 결함'에서 빠짐)
        echo(f"이번 읽기 묶음에 자리가 없는 결함본 {len(unplaced)}개: {', '.join(unplaced)}")
    return {"seed": seed, "docs": {r: len(d) for r, d in lists.items()}, "key_sha256": sha, "unplaced": unplaced}


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
    run.add_argument("--reports", default="", help="재현성 성적서 태그(쉼표). verify 기본 bc_strong,digits_weak,yeast_strong")
    run.add_argument("--repeats", type=int)
    run.add_argument("--base-repeats", type=int, help="결함·sham 전 바탕 확인의 재현성 계획 회차(기본 2)")
    run.add_argument("--max-runs", type=int, help="묶음 회차 상한(기본 = V1 결정 회차: 재현성 6·결함 3·sham 1)")
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
    rep.add_argument("--reports-n", type=int, help="재현성 계획 성적서 수(기본 3 — V1 결정 1)")
    rep.add_argument("--repeats", type=int, help="재현성 계획 회차(기본 2 — V1 결정 1)")
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
    sh.add_argument("--n", type=int, default=SHEETS_N, help="표본 크기(기본 40 = 축마다 10, V1 결정 7)")
    sh.add_argument("--overlap", type=int, default=SHEETS_OVERLAP,
                    help="두 사람이 같이 채점할 문항 수(모두, 축마다 고르게 — 기본 10)")
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
            d = R.DEFAULTS["reproducibility"]
            cmd_report(args.mode, ledger=Path(args.ledger) if args.ledger else None, out=Path(args.out) if args.out else None,
                       answers=Path(args.answers) if args.answers else None, human=Path(args.human) if args.human else None,
                       plan={"reproducibility": R.BatchDefault(args.reports_n or d.items, args.repeats or d.repeats)})
            return 0
        if args.mode == "verify" and (args.inputs or args.defects):
            raise RunnerError("verify 는 입력·결함을 검증 폴더에서만 읽는다(--inputs·--defects 는 dev 전용)")
        summary = execute(RunArgs(
            mode=args.mode, batch=args.batch, ledger=Path(args.ledger) if args.ledger else None,
            inputs=Path(args.inputs) if args.inputs else None, defects=Path(args.defects) if args.defects else None,
            out=Path(args.out) if args.out else None, live=args.live, yes=args.yes, seed=args.seed,
            reports=[t for t in args.reports.split(",") if t], repeats=args.repeats, base_repeats=args.base_repeats,
            max_runs=args.max_runs,
            batch_budget_usd=args.batch_budget_usd))
        print(json.dumps({k: v for k, v in summary.items() if k != "gates"} | {"gates": summary.get("gates")},
                         ensure_ascii=False))
        return 0
    except RunnerError as exc:
        print(f"멈춤: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
