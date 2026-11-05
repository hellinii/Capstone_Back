"""
judge_eval.py — 판정자 평가 실험 실행기(사전 등록 v1 — 평가/사전등록/v1/사전등록_v1.md, 채점 전에 바꾼 규칙은 v3/사전등록_v3.md 2절).
pytest 시험이 아니다.

두 모드(judge_validate 와 같은 뜻)
- dev    : 가짜 LLM(scripts/dev_fake_llm_server 의 가짜 OpenAI)과 개발 입력(tests/data/judge_dev — 규칙 문장 서술)만. 평가 폴더는
           임시 폴더(--root, 없으면 새로 만든다)다. 검증 폴더·평가 폴더(저장소 옆 JUDGE_ROUND/평가·JUDGE_VALIDATION_DIR) 안은 어떤
           철자로 적어도 거부한다(judge_validate.guard + 평가 폴더).
- verify : 진짜 평가. 평가 폴더 = --root, 없으면 환경변수 JUDGE_VALIDATION_DIR, 없으면 저장소 옆 JUDGE_ROUND/평가. 바탕 서술은
           00_봉인/<태그>.json(SHA256SUMS 와 맞아야). 실호출 명령(run·baseline2)은 .env 를 먼저 읽은 뒤(load_env) judge_validate.
           verify_problems 다섯 조건(깨끗한 작업 트리·고정 기준·태그 MANIFEST·--live·장부 쓰기) + 사전 등록 지문(980e9bcc…)·모델
           설정(§0)이 모두 맞아야 시작한다.

평가 폴더
- 00_봉인/      바탕 결과서 3건(pima_strong·segment_weak·scene_strong) — 봉인 서술(judge_live seal-generate 꼴)
- 01_구성/      precheck.json
- 02_결함/<ID>/ meta.json·slots.json·diff.txt — F01~F09·C01~C09(결함), V01~V12(변형). plant 가 쓴다
- 03_실행/      <run_id>.json(채점 결과 원본), plan-r<회차>.json(순서), B2-<문서>.json(베이스라인 2)
- 04_판정표/    판정표.csv(블라인드), 열쇠/열쇠.csv(+ .sha256, 읽기 전용), 기록.json
- 05_결과/      baseline1.json, 집계.json·집계.md, 최종보고.json·최종보고.md
- 장부.csv      judge_live.LEDGER_COLUMNS 24열 그대로. 단계 eval_r1·eval_r2·baseline2, 실행 번호 E-0001 …(두 단계가 같은 번호 줄)

하위 명령(레포 루트에서 python scripts/judge_eval.py <명령> --mode dev|verify …)
- plant --table <json>   : 결함·변형 표 → 02_결함/<ID>/. 표 줄 = {id, base, axis, type, slot(점 경로), old(칸 글에 한 번만 나오는
                           글 그대로), new(바꿀 글), grounds{field, value, note}, expected_axes, allowed_secondary_axes,
                           first_check_pass_by_design(참·거짓·null), author, note}. 사전 등록 §3 표(축·기대 축·허용 부차 축)와
                           다르면 거부. 폴더가 이미 있으면 거부(덮어쓰지 않는다). meta 에 주입 문장(evaluation.injected_sentences).
- precheck               : 변형(V)은 1차 점검(베이스라인 1과 같은 검사)·가독성 코드 규칙 결과가 원본과 같은지, 결함(F·C)은 가독성 코드
                           규칙만(§3.8 — 결함본에 1차 점검을 돌리지 않는다). 01_구성/precheck.json. 변형이 하나라도 다르면 끝 코드 1.
- run --round 1|2        : 33문서(원본 3 + 02_결함 전부) 채점 1회차·2회차. 순서는 시드(§7.2 — 20261105·20261106)로 섞어
                           plan-r<회차>.json 에. 동시에 --concurrency 개(최대 3, §7.2). 2회차는 1회차 33문서가 모두 끝난 뒤에만.
                           이어 하기: 판정이 난 회차(pass·blocked)와 API 실패가 아닌 판정자 실패(최종 실패)는 건너뛰고, API 실패·시간
                           초과(재시도 대상, §7.4)만 같은 문서·같은 회차로 다시 돈다(시도 횟수 attempt·장부 error 열). 확인 입력은 없다:
                           --yes 가 없으면 계획만 찍고 멈춘다. 상한 = JUDGE_LIVE_BUDGET_USD·JUDGE_LIVE_MAX_RUNS(없으면 §7.7 $30·110회,
                           장부 누계 포함, 회당 예상 $0.30 — verify 는 §7.7 값보다 올리면 거부). 시작 전 넘으면 거부, 돌다가 다음
                           회차가 넘으면 새 회차를 띄우지 않는다. 다시 도는 실패는 API 실패·시간 초과(호출 경계의 timeout·transient
                           ·api_error 등과 네트워크 계열 예외)뿐이고, 코드 예외는 최종 실패다(§7.4).
- baseline1              : 1차 점검만(코드, 비용 0) — 33문서 → 05_결과/baseline1.json(칸·문장 위치, 결함본 탐지).
- baseline2              : 문서 전체 판정(사전등록/v1/베이스라인2_프롬프트.md 의 지시문·입력 틀·스키마 — SHA256SUMS 와 맞아야). 문서마다
                           1회, 03_실행/B2-<문서>.json, 장부 단계 baseline2. --limit K·--concurrency N.
- aggregate [--partial]  : §6 집계 → 05_결과/집계.json·집계.md. 회차가 덜 끝났으면 --partial 일 때만(부분 집계 표시). 판정 전에 보는
                           블라인드 집계다 — 표시 인용·위치·종류, 회차별 차단 목록·표시 수처럼 판정표 문항과 출처를 잇는 것은 싣지
                           않는다(§6.4 '열쇠는 판정을 마친 뒤 연다'). 그런 것은 report 의 최종보고에만.
- sheet [--seed S]       : 과잉 차단 판정표(§6.4·판정표_양식.md) — 04_판정표/판정표.csv(첫 줄 '# 열쇠 sha256: …'), 열쇠는 따로(읽기 전용
                           — 열쇠.csv·열쇠.sha256·출처별 표시 수와 시드를 담은 열쇠/기록.json). 쓰기 전에 새는 것 검사
                           (evaluation.sheet_leaks). 판정표가 이미 있으면 거부.
- report --answers <csv> : 판정한 표(열쇠 해시 맞대기) → §6.4 셈 + 전체 집계(표시 세부 포함) → 05_결과/최종보고.md·.json. 봉인
                           (00_봉인/SHA256SUMS)과 사전등록 v3 해시(결함_SHA256SUMS — 02_결함 파일, SHA256SUMS — 목록·precheck 등)도
                           다시 맞대 본다.
- 차례(verify)            : plant → precheck → (v3 동결) → baseline1 → run 1 → run 2 → baseline2 → sheet → 판정 → report.
                           aggregate 는 그 사이 아무 때나(블라인드).
- status                 : 회차별 끝남·실패·남음, 장부 누계와 상한.

상호작용
- 의존(import): scripts/judge_validate.py(바탕 서술·채점 한 번·verify 조건·경계·열쇠 맞대기), scripts/judge_live.py(장부·열 목록·
  장부 줄·run_meta·클라이언트), app.judge.validation.evaluation(§6 계산), app.judge.validation.variants, app.narrative(1차 점검 — 베이스라인 1)
- 사용처: 사람이 직접 실행. 시험은 tests/test_judge_eval_tool.py
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import difflib
import fcntl
import hashlib
import io
import json
import os
import random
import re
import stat
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import judge_live as jl  # noqa: E402
import judge_paths  # noqa: E402
import judge_validate as jv  # noqa: E402

from app.judge import criteria  # noqa: E402
from app.judge.validation import evaluation as E  # noqa: E402
from app.judge.validation import variants as V  # noqa: E402

RunnerError = jl.RunnerError
MODES = ("dev", "verify")
DEFAULT_EVAL = (ROOT.parent / "JUDGE_ROUND" / "평가").resolve()
SEALED, BUILD, DEFECTS, RUNS_OUT, SHEET_DIR, RESULTS_OUT = "00_봉인", "01_구성", "02_결함", "03_실행", "04_판정표", "05_결과"
PREREG_V1, PREREG_V3 = Path("사전등록") / "v1", Path("사전등록") / "v3"
B2_PROMPT = "베이스라인2_프롬프트.md"
LEDGER = "장부.csv"
MAX_CONCURRENCY = 3  # §7.2 동시에 3개까지
RUN_ID = re.compile(r"E-(\d{4,})")
STAGE_OF_ROUND = {1: "eval_r1", 2: "eval_r2"}
B2_STAGE = "baseline2"
# §7.4 다시 돌리는 실패 — API 실패·시간 초과뿐: 호출 경계 llm.FAILURES 의 시간·전송 갈래와, 호출 밖으로 새어 나온 네트워크 계열
# 예외(api_exception:). 판정자 실패(거절·잘림·인용 없음·출력 틀림·안전장치 등)와 코드 예외(exception: — 매번 나는 버그이거나 판정자 출력
# 때문)는 결과를 이유로 다시 돌리지 않는다(최종 실패 — 집계에서 '실패').
RETRYABLE = frozenset({"timeout", "transient", "api_error", "error", "deadline", "no_client"})
EXCEPTION_PREFIX = "exception:"
API_EXCEPTION_PREFIX = "api_exception:"


def retryable(error: str) -> bool:
    error = (error or "").strip()
    return error in RETRYABLE or error.startswith(API_EXCEPTION_PREFIX)


def _network_exceptions() -> tuple[type, ...]:
    import httpx
    import openai

    return (TimeoutError, asyncio.TimeoutError, ConnectionError, openai.APIError, httpx.HTTPError)


def exception_error(exc: BaseException) -> str:
    """회차 하나를 끝낸 예외 → 장부 error 칸. 네트워크 계열(API 실패·시간 초과)만 다시 도는 api_exception:, 나머지는 exception:."""
    prefix = API_EXCEPTION_PREFIX if isinstance(exc, _network_exceptions()) else EXCEPTION_PREFIX
    return f"{prefix}{type(exc).__name__}: {str(exc)[:200]}"


def load_env() -> None:
    """.env 를 읽는다(judge_live.make_clients 와 같은 load_dotenv — 이미 있는 환경변수는 덮지 않는다). verify 의 실호출 명령은 모델
    설정 검사와 상한 계산 전에 부른다 — 검사는 통과하고 채점은 .env 의 다른 설정으로 도는 일이 없게."""
    from dotenv import load_dotenv

    load_dotenv()


# ── 평가 폴더·경계 ────────────────────────────────────────────────────────


def eval_root(root: Path | str | None) -> Path:
    """verify 평가 폴더 — --root, 없으면 JUDGE_VALIDATION_DIR, 없으면 저장소 옆 JUDGE_ROUND/평가."""
    return Path(root).resolve() if root else (jl.validation_dir() or DEFAULT_EVAL)


def dev_guard(path: Path | str) -> Path:
    """dev 경계 — judge_validate.guard(검증 폴더 거부·허용 목록) + 평가 폴더 거부."""
    # 평가 폴더 검사를 먼저 — 저장소 옆 평가 폴더는 허용 목록 밖이기도 해서, 순서가 거꾸로면 거부 사유가 저장소 위치마다 달라진다.
    if judge_paths.inside(Path(path).resolve(), DEFAULT_EVAL):
        raise RunnerError(f"dev 모드는 평가 폴더를 읽거나 쓰지 않는다: {Path(path).resolve()}")
    return jv.guard(path, "dev")


@dataclass
class Ctx:
    mode: str
    root: Path
    echo: Callable[[str], None] = print

    @property
    def ledger_path(self) -> Path:
        return self.root / LEDGER

    @property
    def runner_mode(self) -> str:
        return "live" if self.mode == "verify" else "dev"

    def path(self, *parts) -> Path:
        p = self.root.joinpath(*parts)
        return dev_guard(p) if self.mode == "dev" else p


def make_ctx(mode: str, root: Path | str | None, echo: Callable[[str], None] = print) -> Ctx:
    if mode not in MODES:
        raise RunnerError(f"모르는 모드: {mode}")
    if mode == "dev":
        return Ctx(mode, dev_guard(root or tempfile.mkdtemp(prefix="judge_eval_")), echo)
    return Ctx(mode, eval_root(root), echo)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def defect_files(root: Path) -> dict[str, str]:
    """02_결함 의 파일 → sha256(평가 폴더 기준 경로). 점으로 시작하는 이름(.DS_Store 등 — 파일 탐색기가 만드는 것)은 뺀다."""
    folder = Path(root) / DEFECTS
    if not folder.is_dir():
        return {}
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts)}


def read_sums(path: Path) -> dict[str, str]:
    sums = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(None, 1)
            sums[name.strip().lstrip("*")] = digest
    return sums


def eval_verify_problems(*, live: bool, ledger_path: Path, git=jv._git) -> list[str]:
    """실호출 명령(run·baseline2)의 시작 조건 — judge_validate 다섯 조건 + 사전 등록 §0(기준 지문·모델 설정)."""
    from app.judge import config

    problems = jv.verify_problems(live=live, ledger_path=ledger_path, git=git)
    if criteria.fingerprint() != E.PREREG_FINGERPRINT:
        problems.append(f"기준 지문이 사전 등록 값과 다르다({criteria.fingerprint()[:8]}… ≠ {E.PREREG_FINGERPRINT[:8]}…)")
    want = {"판정자 모델": (config.judge_model(), E.PREREG_JUDGE_MODEL), "판정자 추론 강도": (config.judge_effort(), E.PREREG_JUDGE_EFFORT),
            "추출 모델": (config.extract_model(), E.PREREG_EXTRACT_MODEL),
            "추출 추론 강도": (config.extract_effort(), E.PREREG_EXTRACT_EFFORT)}
    for name, (now, fixed) in want.items():
        if now != fixed:
            problems.append(f"{name}가 사전 등록(§0)과 다르다: {now} ≠ {fixed}(환경변수를 지운다)")
    return problems


# ── 바탕·결함·문서 ────────────────────────────────────────────────────────


def load_bases(ctx: Ctx) -> dict[str, jv.Base]:
    """바탕 결과서 — dev 는 개발 입력 3건(규칙 문장 서술), verify 는 00_봉인 3건(해시 확인)."""
    if ctx.mode == "dev":
        return asyncio.run(jv.dev_bases(jv.DEV_INPUTS))
    sealed = ctx.root / SEALED
    if not (sealed / "SHA256SUMS").exists():
        raise RunnerError(f"봉인 서술이 없다: {sealed}/SHA256SUMS")
    return jv.sealed_bases(sealed, list(E.EVAL_REPORTS))


def load_items(ctx: Ctx) -> dict[str, tuple[dict, dict]]:
    """02_결함 의 결함·변형(F·C·V 폴더) → {ID: (meta, slots)}."""
    folder = ctx.path(DEFECTS)
    if not folder.is_dir():
        return {}
    return {p.name: V.load_variant(p) for p in sorted(folder.iterdir()) if p.is_dir() and E.EVAL_ID.fullmatch(p.name)}


@dataclass
class Doc:
    report: str
    variant: str
    meta: dict = field(default_factory=dict)
    slots: dict | None = None

    @property
    def id(self) -> str:
        return E.doc_id(self.report, self.variant)

    @property
    def source(self) -> str:
        return E.source_of(self.variant)

    @property
    def is_defect(self) -> bool:
        return self.variant in E.PREREG_DEFECTS


def documents(bases: dict[str, jv.Base], items: dict[str, tuple[dict, dict]]) -> list[Doc]:
    """원본(결과서 태그 순) + 결함·변형(ID 순)."""
    docs = [Doc(tag, E.BASE) for tag in sorted(bases)]
    for vid, (meta, slots) in sorted(items.items()):
        if meta.get("base") not in bases:
            raise RunnerError(f"{vid}: 바탕 결과서 {meta.get('base')!r} 가 없다")
        docs.append(Doc(meta["base"], vid, meta, slots))
    return docs


def _request(base: jv.Base):
    from app.narrative.schemas import NarrativeRequest

    return NarrativeRequest.model_validate(base.request)


def doc_result(base: jv.Base, slots: dict | None) -> dict:
    return V.apply_slots(base.result, slots) if slots else base.result


def doc_pieces(base: jv.Base, slots: dict | None):
    """판정자가 읽는 조각(judge_validate.doc_text 와 같은 재료) — 서술 칸 + 고정 문구 + 표."""
    from app.judge.material import parse_material
    from app.judge.reading import build_pieces
    from app.narrative.baselines import build_benchmark_refs
    from app.narrative.schemas import NarrativeJobResult

    req = _request(base)
    result = NarrativeJobResult.model_validate(doc_result(base, slots))
    material, _ = parse_material(req.judge_material)
    refs = build_benchmark_refs(req.task_type.value, req.fact_sheet.metrics, edition=4)
    return build_pieces(result, result.section_sources, material, refs, req.fact_sheet.metrics)


def doc_index(base: jv.Base, slots: dict | None) -> E.DocIndex:
    return E.DocIndex.from_pieces(doc_pieces(base, slots))


def narrative_sha(base: jv.Base, slots: dict | None) -> str:
    """judge_validate.judge_one 과 같은 서술 해시."""
    body = doc_result(base, slots)
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def sources_of(base: jv.Base):
    """판정자 조회 자료(app.judge.sources.Sources) — 필드 이름 확인·판정표 필드 값."""
    from app.judge.sources import Sources
    from app.narrative.baselines import build_benchmark_refs
    from app.narrative.derived import compute_derived

    req = _request(base)
    fs = req.fact_sheet
    refs = build_benchmark_refs(req.task_type.value, fs.metrics, edition=4)
    return Sources(fs.model_dump(), refs, compute_derived(fs, task_type=req.task_type.value), req.client_inputs)


def injected_of(doc: Doc, base: jv.Base, index: E.DocIndex) -> list[int]:
    """결함본·변형본의 주입 문장 번호 — 바탕 칸 글과 바뀐 칸 글로 다시 계산해 meta 기록과 맞대 본 뒤 문서 문장에 붙인다."""
    slot = doc.meta["slot"]
    again = E.injected_sentences(V.read_slot(base.result, slot), doc.slots[slot])
    if [list(x) for x in again] != [list(x) for x in doc.meta.get("injected") or []]:
        raise RunnerError(f"{doc.id}: meta.injected 가 다시 계산한 주입 문장과 다르다(02_결함 을 손으로 고쳤나)")
    return E.injected_ids(index, slot, again)


def original_ids_of(doc: Doc, base: jv.Base, original_index: E.DocIndex) -> list[int]:
    """결함의 원래 문장(v3 2.1 — 바꾸기 전 원본 문장) → 원본 문서의 문장 번호. 덧붙이기만 한 결함은 []."""
    slot = doc.meta["slot"]
    replaced = E.replaced_sentences(V.read_slot(base.result, slot), doc.slots[slot])
    return E.injected_ids(original_index, slot, replaced) if replaced else []


# ── plant ─────────────────────────────────────────────────────────────

PLANT_FIELDS = ("id", "base", "axis", "type", "slot", "old", "new", "grounds", "expected_axes", "allowed_secondary_axes",
                "first_check_pass_by_design", "author")
NARRATIVE_SOURCES = ("llm", "fallback")


def _slot_source(result: dict, slot_path: str) -> str:
    from app.judge.validation.sheets import slot_of_path

    sid = slot_of_path(slot_path)
    if sid == "9":
        return result["section_sources"]["9"]
    return result["section_sources"].get(sid, "")


def check_plant_row(row: dict, bases: dict[str, jv.Base], *, mode: str, existing: set[str]) -> tuple[list[str], dict | None]:
    """표 한 줄 검사 → (문제 목록, 쓸 것). 문제 문장에는 칸 글을 옮기지 않는다(화면에 찍히므로)."""
    rid = str(row.get("id", "?"))
    problems = [f"'{f}' 칸이 없다" for f in PLANT_FIELDS if f not in row]
    if problems:
        return problems, None
    if not E.EVAL_ID.fullmatch(rid) or rid not in (*E.DEFECT_IDS, *E.VARIANT_IDS):
        return [f"ID 는 F01~F09·C01~C09·V01~V12 중 하나다: {rid}"], None
    if rid in existing:
        return [f"02_결함/{rid} 가 이미 있다 — 덮어쓰지 않는다"], None
    base = bases.get(row["base"])
    if base is None:
        return [f"바탕 결과서 {row['base']!r} 가 없다(있는 것: {', '.join(sorted(bases))})"], None
    is_defect = rid in E.PREREG_DEFECTS
    extra: dict = {}
    if is_defect:
        plan = E.PREREG_DEFECTS[rid]
        if row["axis"] != plan.axis:
            problems.append(f"축이 사전 등록과 다르다: {row['axis']} ≠ {plan.axis}")
        if row["type"] != plan.type:
            problems.append(f"유형이 사전 등록과 다르다: {row['type']} ≠ {plan.type}")
        if sorted(row["expected_axes"] or []) != sorted(plan.expected_axes):
            problems.append(f"기대 축이 사전 등록과 다르다: {row['expected_axes']} ≠ {list(plan.expected_axes)}")
        if sorted(row["allowed_secondary_axes"] or []) != sorted(plan.allowed_secondary_axes):
            problems.append(f"허용 부차 축이 사전 등록과 다르다: {row['allowed_secondary_axes']} ≠ {list(plan.allowed_secondary_axes)}")
        if mode == "verify" and row["base"] != plan.report:
            # §3.5 — 넣을 문장이 없으면 다른 결과서로 옮기고 기록한다
            if not str(row.get("note") or "").strip():
                problems.append(f"사전 등록 결과서({plan.report})와 다른 결과서로 옮겼으면 note 에 까닭을 적는다(§3.5)")
            extra["moved_from"] = plan.report
    else:
        if row["type"] not in E.VARIANT_TYPES:
            problems.append(f"변형 유형은 {', '.join(E.VARIANT_TYPES)} 중 하나다: {row['type']}")
        if row["expected_axes"] or row["allowed_secondary_axes"]:
            problems.append("변형은 기대 축·허용 부차 축이 없다(빈 목록)")
    if row["first_check_pass_by_design"] not in (True, False, None):
        problems.append("first_check_pass_by_design 은 true·false·null 이다")
    if not str(row["author"] or "").strip():
        problems.append("author(작성자)가 비었다")
    grounds = row["grounds"]
    if not isinstance(grounds, dict) or not {"field", "value", "note"} <= set(grounds):
        problems.append("grounds 는 {field, value, note} 다")
    slot = row["slot"]
    try:
        base_text = V.read_slot(base.result, slot)
    except (KeyError, IndexError, ValueError):
        return problems + [f"바탕 결과서에 없는 서술 칸이다: {slot}"], None
    if _slot_source(base.result, slot) not in NARRATIVE_SOURCES:
        return problems + [f"서술(AI·규칙 문장) 칸이 아니다(출처 {_slot_source(base.result, slot)}): {slot}"], None
    old, new = str(row["old"]), str(row["new"])
    count = base_text.count(old) if old else 0
    if count != 1:
        return problems + [f"old 가 칸 글에 정확히 한 번 나와야 한다(지금 {count}번)"], None
    if old == new:
        return problems + ["new 가 old 와 같다"], None
    new_text = base_text.replace(old, new, 1)
    injected = E.injected_sentences(base_text, new_text)
    if not injected:
        return problems + ["바뀐 문장이 없다(문장 나누기 기준)"], None
    a, b = V.sentences(base_text), V.sentences(new_text)
    ops = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if row["type"] != "문장 순서 교체" and len(ops) != 1:
        problems.append(f"바뀐 곳이 한 곳이어야 한다(지금 {len(ops)}곳 — §3.1 결함본 1개에 결함 1개)")
    if rid in E.PREREG_DEFECTS and rid != "C01" and E.PREREG_DEFECTS[rid].axis == "consistency":
        # v3 2.8 — 숫자는 §6.9 숫자 토큰 정의로 센다(F1·M4 같은 이름 속 숫자는 숫자가 아니다)
        if Counter(E.numeric_tokens(base_text)) != Counter(E.numeric_tokens(new_text)):
            problems.append("일관성 결함(C01 제외)의 변경문은 숫자를 바꾸지 않는다(§3.4·v3 2.8)")
    try:
        E.injected_ids(doc_index(base, {slot: new_text}), slot, injected)
    except ValueError as exc:
        problems.append(f"주입 문장을 문서 문장에 붙이지 못했다: {exc}")
    if problems:
        return problems, None
    warnings = []
    if is_defect and isinstance(grounds, dict) and str(grounds.get("field") or "").strip():
        known = sources_of(base)
        names = set(known.numbers) | set(known.texts)
        unknown = [f for f in re.split(r"[|,]", str(grounds["field"])) if f.strip() and E.canon_field(f) not in names]
        if unknown:
            warnings.append(f"근거 필드가 판정자 조회 자료에 없다: {', '.join(unknown)} — 미탐지 원인(6.3)의 필드 대조에 쓰인다")
    meta = {
        "id": rid, "kind": "defect" if is_defect else "variant", "base": row["base"], "axis": row["axis"], "type": row["type"],
        "slot": slot, "old": old, "new": new, "grounds": grounds, "expected_axes": list(row["expected_axes"] or []),
        "allowed_secondary_axes": list(row["allowed_secondary_axes"] or []),
        "first_check_pass_by_design": row["first_check_pass_by_design"], "author": row["author"], "note": row.get("note") or "",
        "injected": [list(x) for x in injected], "base_slot_sha256": hashlib.sha256(base_text.encode("utf-8")).hexdigest(),
        **extra,
    }
    return [], {"meta": meta, "new_text": new_text, "base_text": base_text, "warnings": warnings}


def cmd_plant(ctx: Ctx, table: Path, *, echo: Callable[[str], None] | None = None) -> list[str]:
    """결함·변형 표 → 02_결함/<ID>/{meta.json, slots.json, diff.txt}. 한 줄이라도 틀리면 아무것도 쓰지 않는다."""
    echo = echo or ctx.echo
    raw = json.loads(Path(table).read_text(encoding="utf-8"))
    rows = raw["rows"] if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise RunnerError("표가 비었다(줄 목록 또는 {\"rows\": [...]})")
    ids = [str(r.get("id")) for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise RunnerError(f"표에 같은 ID 가 두 번 있다: {', '.join(dup)}")
    bases = load_bases(ctx)
    folder = ctx.path(DEFECTS)
    existing = {p.name for p in folder.iterdir()} if folder.is_dir() else set()
    plans, problems = [], []
    for row in rows:
        bad, plan = check_plant_row(row, bases, mode=ctx.mode, existing=existing)
        problems += [f"{row.get('id', '?')}: {p}" for p in bad]
        if plan:
            plans.append(plan)
    if problems:
        raise RunnerError("표가 틀려 아무것도 쓰지 않는다:\n  - " + "\n  - ".join(problems))
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    written = []
    for plan in plans:
        meta = {**plan["meta"], "planted_at": stamp}
        target = folder / meta["id"]
        target.mkdir()  # 있으면 FileExistsError — 위에서 이미 걸렀다
        (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (target / "slots.json").write_text(json.dumps({meta["slot"]: plan["new_text"]}, ensure_ascii=False, indent=1) + "\n",
                                           encoding="utf-8")
        (target / "diff.txt").write_text(V.make_diff(plan["base_text"], plan["new_text"], meta["slot"]), encoding="utf-8")
        written.append(meta["id"])
        echo(f"{meta['id']} 씀 — 주입 문장 {len(meta['injected'])}개")
        for w in plan["warnings"]:
            echo(f"  ! {w}")
    return written


# ── 1차 점검(베이스라인 1·변형 확인) ───────────────────────────────────────────


def _section_parts(result: dict, sid: str) -> dict:
    from app.narrative.sections import SECTIONS_BY_ID

    if sid == "9":
        rn = result["recommendation_narrative"]
        return {"recommendation_narrative.data_quality": rn.get("data_quality") or "",
                "recommendation_narrative.model_ops": rn.get("model_ops") or "",
                "recommendations": [dict(r) for r in result.get("recommendations") or []]}
    path = SECTIONS_BY_ID[sid].response_paths[0]
    group, name = path.split(".")
    return {path: result[group][name] or ""}


class FirstCheck:
    """생성 단계 1차 점검(app.narrative.checks.check_section)과 최종 재확인(판정 모순)을 서술 칸마다 — 베이스라인 1(§5)·변형 확인(§4).
    요청 하나로 허용 목록·결론 목록을 한 번 만들고(generator._prepare 그대로), 칸 글은 문서마다 바꿔 끼운다.
    해석 메모: 생성 경로는 AI 칸만 검사하지만, 여기는 서술 칸(출처 AI·규칙 문장) 모두에 적용한다(§5 '서술 칸 단위로')."""

    def __init__(self, base: jv.Base):
        from app.narrative import checks
        from app.narrative.generator import _prepare

        self.req = _request(base)
        self.fs = self.req.fact_sheet
        self.edition = self.req.report_edition
        _, _, self.whitelist, self.conclusions, _, self.exempt, self.masks = _prepare(self.req)
        self.labels = checks.numeric_labels(self.fs)
        self.problem_ids = [p.id for p in self.fs.problems]

    def sections(self, result: dict) -> dict[str, dict]:
        from app.narrative import checks
        from app.narrative.generator import _align, _rec_dicts
        from app.narrative.grounding import find_verdict_contradictions
        from app.narrative.sections import SECTION_IDS

        out = {}
        for sid in SECTION_IDS:
            if result["section_sources"].get(sid) not in NARRATIVE_SOURCES:
                continue
            parts = _section_parts(result, sid)
            if sid == "9":
                parts = _align(parts, self.fs.problems, fill_missing=False, edition=self.edition)[0]
            parts = _rec_dicts(parts)
            rec = checks.check_section(sid, parts, whitelist=self.whitelist, verdict=self.fs.verdict,
                                       conclusions=self.conclusions, problem_ids=self.problem_ids, exempt=self.exempt,
                                       masks=self.masks, labels=self.labels, edition=self.edition).as_record()
            final = bool(find_verdict_contradictions(checks.mask_names(checks.section_texts(sid, parts), self.masks),
                                                     self.fs.verdict))
            out[sid] = {"source": result["section_sources"][sid], "check": rec, "final_recheck": final}
        return out

    def findings(self, result: dict, index: E.DocIndex) -> list[dict]:
        """검출 → 위치(칸·문장 번호). 숫자·판정 모순·예측 수치·코드 낱말은 문장마다 같은 검사로 다시 찾고, 권고 빠짐·선언 숫자 빠짐·
        빈 칸·7.3 인용 없음은 칸 단위(문장 번호 없음)."""
        from app.narrative import checks
        from app.narrative.grounding import _NUMBER_RE, find_verdict_contradictions, strip_context, verify_grounding

        out = []
        for sid, s in self.sections(result).items():
            rec = s["check"]
            sents = [x for x in index.sentences if x.slot == sid]

            def where(test) -> list[int]:
                return [x.id for x in sents if test(x.text)]

            def contradicts(text: str) -> bool:
                return bool(find_verdict_contradictions(checks.mask_names([text], self.masks), self.fs.verdict))

            for tok in rec["numbers"]:
                out.append({"section": sid, "check": "numbers", "value": tok, "sentence_ids": where(
                    lambda t, tok=tok: tok in verify_grounding([t], self.whitelist, edition=self.edition).violations)})
            if rec["verdict_contradiction"]:
                out.append({"section": sid, "check": "verdict_contradiction", "value": None, "sentence_ids": where(contradicts)})
            for tok in rec["impact_numbers"]:
                out.append({"section": sid, "check": "impact_number", "value": tok, "sentence_ids": where(
                    lambda t, tok=tok: tok in _NUMBER_RE.findall(strip_context(t, edition=self.edition)))})
            for word in rec["code_words"]:
                out.append({"section": sid, "check": "code_words", "value": word, "sentence_ids": where(
                    lambda t, word=word: word in checks.code_words([t], self.exempt))})
            for pid in rec["missing_recommendations"]:
                out.append({"section": sid, "check": "missing_recommendation", "value": pid, "sentence_ids": []})
            if rec["d24_missing"]:
                out.append({"section": sid, "check": "d24_missing", "value": None, "sentence_ids": []})
            for fmt in rec["format"]:
                if fmt != "code_words":
                    out.append({"section": sid, "check": f"format:{fmt}", "value": None, "sentence_ids": []})
            if s["final_recheck"]:
                out.append({"section": sid, "check": "final_verdict_contradiction", "value": None,
                            "sentence_ids": where(contradicts)})
        return out


def readability_rules(base: jv.Base, slots: dict | None) -> list[dict]:
    """가독성 코드 규칙 둘(로마자 노출·서술체) — app.judge.readability 그대로(LLM 표시는 부르지 않는다)."""
    from dataclasses import asdict

    from app.judge.readability import allowed_latin, romanization, sentence_style
    from app.judge.reading import for_module
    from app.narrative.baselines import build_benchmark_refs

    req = _request(base)
    refs = build_benchmark_refs(req.task_type.value, req.fact_sheet.metrics, edition=4)
    scope = for_module(doc_pieces(base, slots), "readability")
    allowed = allowed_latin(req.fact_sheet, req.client_inputs, [r.get("source_note", "") for r in refs])
    return [asdict(f) for f in romanization(scope, allowed) + sentence_style(scope)]


def readability_signature(findings: list[dict]) -> list[list[str]]:
    """원본과 견줄 꼴 — (규칙, 칸, 로마자면 그 낱말). 서술체 위반은 문장 글이 변형마다 달라 칸 단위로 센다."""
    return sorted([f["rule"], f["slot"], f["text"] if f["rule"] == "romanization" else "-"] for f in findings)


def first_check_signature(sections: dict[str, dict]) -> dict:
    return {sid: {**s["check"], "final_recheck": s["final_recheck"]} for sid, s in sorted(sections.items())}


def cmd_precheck(ctx: Ctx, *, echo: Callable[[str], None] | None = None) -> dict:
    """동결 전 확인(§3.8·§4) → 01_구성/precheck.json. 서술 글은 찍지 않는다(칸·검사 이름·건수만)."""
    echo = echo or ctx.echo
    bases = load_bases(ctx)
    items = load_items(ctx)
    if not items:
        raise RunnerError("02_결함 에 결함·변형이 없다 — plant 를 먼저 한다")
    originals = {}
    for tag, base in sorted(bases.items()):
        fc = FirstCheck(base)
        originals[tag] = {"first_check": first_check_signature(fc.sections(base.result)),
                          "readability": readability_signature(readability_rules(base, None))}
    out: dict = {"mode": ctx.mode, "criteria_version": criteria.version(), "fingerprint": criteria.fingerprint(),
                 "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "items": {}, "files": {}}
    variant_bad = []
    for vid, (meta, slots) in sorted(items.items()):
        base = bases.get(meta.get("base"))
        if base is None:
            raise RunnerError(f"{vid}: 바탕 결과서 {meta.get('base')!r} 가 없다")
        rea = readability_signature(readability_rules(base, slots))
        rea_same = rea == originals[meta["base"]]["readability"]
        entry: dict = {"base": meta["base"], "kind": meta.get("kind"), "type": meta.get("type"),
                       "readability_same": rea_same,
                       "readability_diff": None if rea_same else {"original": originals[meta["base"]]["readability"], "this": rea}}
        if vid.startswith("V"):
            fc = first_check_signature(FirstCheck(base).sections(doc_result(base, slots)))
            fc_same = fc == originals[meta["base"]]["first_check"]
            entry.update({"first_check_same": fc_same,
                          "first_check_diff": None if fc_same else {
                              sid: {"original": originals[meta["base"]]["first_check"].get(sid), "this": fc.get(sid)}
                              for sid in sorted(set(fc) | set(originals[meta["base"]]["first_check"]))
                              if fc.get(sid) != originals[meta["base"]]["first_check"].get(sid)}})
            ok = fc_same and rea_same
            if not ok:
                variant_bad.append(vid)
        else:
            entry["first_check"] = "돌리지 않음(§3.8 — 베이스라인 1 결과를 미리 보지 않는다)"
            ok = rea_same
        out["items"][vid] = entry
        echo(f"{vid} {'같음' if ok else '다름'}" + ("" if vid.startswith("V") else " (가독성 코드 규칙만)"))
    out["files"] = defect_files(ctx.root)  # 사전 등록 v3 결함_SHA256SUMS 에 옮겨 적을 값
    out["variants_differ"] = variant_bad
    target = ctx.path(BUILD, "precheck.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    echo(f"precheck {target} — 변형 다름 {len(variant_bad)}개" + (f": {', '.join(variant_bad)}" if variant_bad else ""))
    return out


# ── 장부·실행 번호·상한 ──────────────────────────────────────────────────


class LedgerWriter:
    """장부 쓰기 한 곳 — 프로세스 안은 asyncio.Lock, 프로세스 사이는 장부 파일 flock. 실행 번호(E-0001 …)는 잠근 채 장부의 가장 큰
    E 번호 + 1 로 정하고, 결과 원본 파일과 장부 줄을 같은 잠금 안에서 쓴다(동시에 끝난 회차가 같은 번호를 받지 않게)."""

    def __init__(self, path: Path):
        self.ledger = jl.Ledger(path)
        self.lock = asyncio.Lock()

    @contextlib.contextmanager
    def _flock(self):
        with open(self.ledger.path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    @staticmethod
    def next_id(rows: list[dict]) -> str:
        used = [int(m.group(1)) for r in rows if (m := RUN_ID.fullmatch(r.get("run_id", "")))]
        return f"E-{max(used, default=0) + 1:04d}"

    async def commit(self, build: Callable[[str, list[dict]], tuple[dict, list[tuple[Path, str]]]]) -> dict:
        """build(실행 번호, 지금 장부 줄) → (장부 줄, [(파일, 글)]). 파일을 먼저 쓰고 장부 줄을 단다."""
        async with self.lock:
            with self._flock():
                rows = self.ledger.rows()
                run_id = self.next_id(rows)
                row, files = build(run_id, rows)
                for path, text in files:
                    if path.exists():
                        raise RunnerError(f"결과 파일이 이미 있다(덮어쓰지 않는다): {path}")
                    path.write_text(text, encoding="utf-8")
                return self.ledger.append({**row, "run_id": run_id})


@dataclass(frozen=True)
class Cap:
    """이 실험 장부 상한(§7.7 $30·110회) — JUDGE_LIVE_BUDGET_USD·JUDGE_LIVE_MAX_RUNS 로 낮출 수 있다. verify 는 §7.7 값보다 높이면
    거부한다(상한은 사전 등록 값이다). dev 는 시험용으로 아무 값이나."""

    usd: float
    runs: int

    @classmethod
    def from_env(cls, mode: str = "verify") -> "Cap":
        usd = os.getenv("JUDGE_LIVE_BUDGET_USD", "").strip()
        runs = os.getenv("JUDGE_LIVE_MAX_RUNS", "").strip()
        cap = cls(float(usd) if usd else E.BUDGET_USD, int(runs) if runs else E.BUDGET_RUNS)
        if mode == "verify" and (cap.usd > E.BUDGET_USD + 1e-9 or cap.runs > E.BUDGET_RUNS):
            raise RunnerError(f"상한이 사전 등록 §7.7(${E.BUDGET_USD:.0f}·{E.BUDGET_RUNS}회)보다 높다: ${cap.usd:.2f}·{cap.runs}회 "
                              "(JUDGE_LIVE_BUDGET_USD·JUDGE_LIVE_MAX_RUNS — .env 포함)")
        return cap


def spent(rows: list[dict]) -> tuple[float, int]:
    """(장부 누계 달러, 줄 수). 비용을 모르는 줄은 회당 예상($0.30)으로 센다(상한을 넘지 않게 보수적으로)."""
    usd = sum(float(r["cost_usd"]) if (r.get("cost_usd") or "").strip() else E.EST_USD_PER_RUN for r in rows)
    return usd, len(rows)


def cap_problem(rows: list[dict], cap: Cap, more: int) -> str | None:
    usd, n = spent(rows)
    est = more * E.EST_USD_PER_RUN
    if n + more > cap.runs:
        return f"회차 상한을 넘는다: 장부 {n}회 + {more}회 > {cap.runs}회"
    if usd + est > cap.usd + 1e-9:
        return f"비용 상한을 넘는다: 장부 ${usd:.2f} + 예상 ${est:.2f}({more}회 × ${E.EST_USD_PER_RUN:.2f}) > ${cap.usd:.2f}"
    return None


def row_status(row: dict, *, fingerprint: str, mode: str) -> str | None:
    """장부 줄 하나의 상태 — done(판정이 남) · final(판정자 실패 — 다시 돌리지 않는다) · retry(API 실패·시간 초과·예외).
    다른 모드·다른 기준 지문의 줄은 None(이 실험 회차가 아니다)."""
    if row.get("mode") != mode or row.get("manifest_sha256") != fingerprint:
        return None
    if (row.get("result_file") or "").strip() and row.get("gate") in ("pass", "blocked"):
        return "done"
    if (row.get("result_file") or "").strip() and not retryable(row.get("error", "")):
        return "final"
    return "retry"


def stage_states(rows: list[dict], stage: str, *, fingerprint: str, mode: str) -> dict[tuple[str, str], dict]:
    """(결과서, 변형) → {status: done|final|retry, row: 끝낸 줄(없으면 마지막 줄), attempts}."""
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.get("stage") != stage:
            continue
        st = row_status(row, fingerprint=fingerprint, mode=mode)
        if st is None:
            continue
        key = (row["report"], row["variant"])
        cur = out.setdefault(key, {"status": "retry", "row": row, "attempts": 0})
        cur["attempts"] += 1
        if cur["status"] in ("done", "final"):
            continue  # 끝난 뒤의 줄은 없어야 한다 — 있어도 처음 끝낸 줄을 쓴다
        cur["status"], cur["row"] = st, row
    return out


def finished(state: dict | None) -> bool:
    return bool(state) and state["status"] in ("done", "final")


# ── run ──────────────────────────────────────────────────────────────


def plan_order(ctx: Ctx, round_no: int, docs: list[Doc], *, write: bool) -> list[Doc]:
    """회차 안 순서(§7.2 시드). plan-r<회차>.json 이 있으면 지금 문서 목록·순서가 같은지 보고, 없으면 write 일 때 쓴다."""
    seed = E.ROUND_SEEDS[round_no]
    order = sorted(docs, key=lambda d: (d.report, d.variant))
    random.Random(seed).shuffle(order)
    listed = [[d.report, d.variant, d.id] for d in order]
    path = ctx.path(RUNS_OUT, f"plan-r{round_no}.json")
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("seed") != seed or saved.get("order") != listed:
            raise RunnerError(f"{path.name} 의 순서가 지금 문서 목록과 다르다 — 02_결함 이 바뀌었나(동결 뒤에는 고치지 않는다)")
    elif write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"round": round_no, "seed": seed, "fingerprint": criteria.fingerprint(),
                                    "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "order": listed},
                                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return order


def full_set_problems(docs: list[Doc]) -> list[str]:
    """verify — 사전 등록 33문서(원본 3·결함 18·변형 12)가 모두 있어야 한다."""
    want = {*E.EVAL_REPORTS, *E.DEFECT_IDS, *E.VARIANT_IDS}
    have = {d.id for d in docs}
    out = []
    if want - have:
        out.append(f"없는 문서: {', '.join(sorted(want - have))}")
    if have - want:
        out.append(f"사전 등록에 없는 문서: {', '.join(sorted(have - want))}")
    return out


def frozen_problems(ctx: Ctx) -> list[str]:
    """verify 채점 전 — precheck 가 변형 다름 없이 끝났고 그 뒤 02_결함 이 바뀌지 않았으며, 사전등록 v3 해시가 있으면 맞다."""
    out = []
    pre = ctx.root / BUILD / "precheck.json"
    files = defect_files(ctx.root)
    if not pre.exists():
        out.append("01_구성/precheck.json 이 없다 — precheck 를 먼저 돌린다(§4 동결 전 확인)")
    else:
        saved = json.loads(pre.read_text(encoding="utf-8"))
        if saved.get("variants_differ"):
            out.append(f"precheck 에서 원본과 다른 변형이 있다: {', '.join(saved['variants_differ'])}")
        if saved.get("files") != files:
            out.append("precheck 뒤에 02_결함 파일이 바뀌었다 — precheck 를 다시 돌린다")
    out += defect_sum_problems(ctx)
    return out


V3_DEFECT_SUMS, V3_SUMS = "결함_SHA256SUMS", "SHA256SUMS"


def defect_sum_problems(ctx: Ctx) -> list[str]:
    """사전등록 v3 해시(§5)를 맞대 본다 — 있는 것만.
    - 결함_SHA256SUMS: 02_결함 파일마다(이름은 '02_결함/F01/meta.json' 또는 'F01/meta.json'). 적힌 파일이 모두 있고 같아야 하며,
      02_결함 의 파일(숨김 파일 빼고)이 모두 적혀 있어야 한다.
    - SHA256SUMS: 결함변형_목록.json·precheck.json·사전등록_v3.md 등. 평가 폴더·02_결함·01_구성·v3 폴더에서 찾은 것은 같아야 한다
      (찾지 못한 이름은 이 도구가 볼 수 없는 파일로 보고 넘어간다)."""
    out: list[str] = []
    folder = ctx.root / PREREG_V3
    dsums = folder / V3_DEFECT_SUMS
    if dsums.exists():
        listed = {}
        for name, digest in read_sums(dsums).items():
            rel = name[len(DEFECTS) + 1:] if name.startswith(f"{DEFECTS}/") else name
            listed[f"{DEFECTS}/{rel}"] = digest
        have = defect_files(ctx.root)
        for name, digest in sorted(listed.items()):
            if name not in have:
                out.append(f"사전등록 v3 {V3_DEFECT_SUMS} 에 적힌 파일이 없다: {name}")
            elif have[name] != digest:
                out.append(f"사전등록 v3 {V3_DEFECT_SUMS} 와 다르다: {name}")
        out += [f"사전등록 v3 {V3_DEFECT_SUMS} 에 없는 02_결함 파일: {name}" for name in sorted(set(have) - set(listed))]
    sums = folder / V3_SUMS
    if sums.exists():
        for name, digest in read_sums(sums).items():
            cands = [ctx.root / name, ctx.root / DEFECTS / name, ctx.root / BUILD / name, folder / name]
            path = next((p for p in cands if p.is_file()), None)
            if path is not None and sha256_file(path) != digest:
                out.append(f"사전등록 v3 {V3_SUMS} 와 다르다: {name}")
    return out


async def launch(ctx: Ctx, writer: LedgerWriter, todo: list[Doc], one: Callable, *, concurrency: int, cap: Cap,
                 what: str) -> dict:
    """todo 를 차례로 동시에 concurrency 개씩 띄운다(run·baseline2 공용). 새 회차를 띄우기 전마다 장부 누계 + 지금 도는 회차 + 이 회차가
    상한을 넘는지 보고, 넘으면 남은 것을 띄우지 않는다(stopped)."""
    sem = asyncio.Semaphore(concurrency)
    state = {"in_flight": 0}

    async def guarded(doc: Doc) -> dict:
        try:
            return await one(doc)
        finally:
            state["in_flight"] -= 1
            sem.release()

    ctx.path(RUNS_OUT).mkdir(parents=True, exist_ok=True)
    tasks, stopped = [], []
    for i, doc in enumerate(todo):
        await sem.acquire()
        async with writer.lock:
            problem = cap_problem(writer.ledger.rows(), cap, state["in_flight"] + 1)
        if problem:
            sem.release()
            stopped = [d.id for d in todo[i:]]
            ctx.echo(f"상한에 닿아 새 {what} 띄우지 않는다({problem}) — 남은 {len(stopped)}회")
            break
        state["in_flight"] += 1
        tasks.append(asyncio.create_task(guarded(doc)))
    return {"rows": await asyncio.gather(*tasks), "stopped": stopped}


async def run_round(ctx: Ctx, round_no: int, order: list[Doc], todo: list[Doc], bases: dict[str, jv.Base], *, client,
                    judge, concurrency: int, cap: Cap, meta: dict) -> dict:
    """남은 회차를 동시에 concurrency 개씩. 회차마다 결과 원본 03_실행/<E-…>.json + 장부 한 줄(잠금 안에서)."""
    stage = STAGE_OF_ROUND[round_no]
    seed = E.ROUND_SEEDS[round_no]
    writer = LedgerWriter(ctx.ledger_path)
    position = {d.id: i for i, d in enumerate(order)}

    async def one(doc: Doc) -> dict:
        started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        t0 = time.monotonic()
        run, sha, error = None, narrative_sha(bases[doc.report], doc.slots), ""
        try:
            run, sha = await judge(client, bases[doc.report], doc.slots)
            error = run.failure or ""
        except Exception as exc:  # noqa: BLE001 — 회차 하나의 예외로 나머지를 멈추지 않는다(네트워크 계열만 재시도 대상)
            error = exception_error(exc)
        seconds = round(time.monotonic() - t0, 3)

        def build(run_id: str, rows: list[dict]):
            attempt = 1 + sum(1 for r in rows if r.get("stage") == stage and r.get("report") == doc.report
                              and r.get("variant") == doc.variant and r.get("mode") == ctx.runner_mode)
            body = {
                "run_id": run_id, "key": {"stage": stage, "round": round_no, "report": doc.report, "variant": doc.variant,
                                          "doc": doc.id, "source": doc.source},
                "attempt": attempt, "order_seed": seed, "order_index": position[doc.id], "mode": ctx.mode, **meta,
                "narrative_sha256": sha, "variant_meta": doc.meta, "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "seconds": seconds,
                "gate": run.gate if run else None, "failure": run.failure if run else None, "error": error,
                "flagged_slots": run.flagged_slots if run else [], "unfixable": run.unfixable if run else [],
                "scorecard": run.scorecard if run else None, "record": run.record if run else {},
            }
            text = json.dumps(body, ensure_ascii=False, indent=1, default=str) + "\n"
            target = ctx.path(RUNS_OUT, f"{run_id}.json")
            cols = jl.judge_run_row(run) if run else {"model_requested": "gpt-6.1-sol+gpt-6-luna", "seconds": seconds}
            row = {**meta, **cols, "stage": stage, "report": doc.report, "variant": doc.variant, "repeat": round_no,
                   "mode": ctx.runner_mode, "result_file": target.relative_to(ctx.root).as_posix(),
                   "result_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "error": error}
            return row, [(target, text)]

        return await writer.commit(build)

    out = await launch(ctx, writer, todo, one, concurrency=concurrency, cap=cap, what="회차를")
    for r in out["rows"]:
        ctx.echo(f"  {r['run_id']} {r['stage']} {E.doc_id(r['report'], r['variant'])} — {r['gate'] or '실패 ' + r['error']}")
    return out


def cmd_run(ctx: Ctx, round_no: int, *, concurrency: int = MAX_CONCURRENCY, only: list[str] | None = None, live: bool = False,
            yes: bool = False, client=None, judge=None, git=jv._git) -> dict:
    """채점 한 회차(§7.2). 확인 입력이 없다 — --yes 가 없으면 계획만 찍고 멈춘다."""
    if round_no not in E.ROUND_SEEDS:
        raise RunnerError(f"회차는 1 또는 2: {round_no}")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise RunnerError(f"동시 실행은 1~{MAX_CONCURRENCY}개(§7.2): {concurrency}")
    if ctx.mode == "verify":
        load_env()  # 모델 설정 검사·상한 계산 전에(.env 의 JUDGE_* 가 검사를 비켜 가지 않게)
        problems = eval_verify_problems(live=live, ledger_path=ctx.ledger_path, git=git)
        if problems:
            raise RunnerError("verify 를 시작하지 않는다:\n  - " + "\n  - ".join(problems))
    cap = Cap.from_env(ctx.mode)
    bases = load_bases(ctx)
    docs = documents(bases, load_items(ctx))
    if ctx.mode == "verify":
        problems = full_set_problems(docs) + frozen_problems(ctx)
        if problems:
            raise RunnerError("채점을 시작하지 않는다:\n  - " + "\n  - ".join(problems))
    fp = criteria.fingerprint()
    rows = jl.Ledger(ctx.ledger_path).rows() if ctx.ledger_path.exists() else []
    if round_no == 2:
        first = stage_states(rows, STAGE_OF_ROUND[1], fingerprint=fp, mode=ctx.runner_mode)
        left = [d.id for d in docs if not finished(first.get((d.report, d.variant)))]
        if left:
            raise RunnerError(f"1회차가 끝나지 않았다(§7.2 — 1회차 33문서를 마친 뒤 2회차): 남은 {len(left)}개 {', '.join(left[:6])}")
    order = plan_order(ctx, round_no, docs, write=False)
    states = stage_states(rows, STAGE_OF_ROUND[round_no], fingerprint=fp, mode=ctx.runner_mode)
    todo = [d for d in order if not finished(states.get((d.report, d.variant)))]
    if only:
        unknown = sorted(set(only) - {d.id for d in docs})
        if unknown:
            raise RunnerError(f"없는 문서: {', '.join(unknown)}")
        todo = [d for d in todo if d.id in set(only)]
    usd, n = spent(rows)
    retries = sum(1 for d in todo if (states.get((d.report, d.variant)) or {}).get("status") == "retry")
    done = sum(finished(states.get((d.report, d.variant))) for d in order)
    ctx.echo(f"[{ctx.mode}] {STAGE_OF_ROUND[round_no]}: 계획 {len(order)}문서 · 끝남 {done}"
             f" · 이번 {len(todo)}회(다시 도는 것 {retries}) · 시드 {E.ROUND_SEEDS[round_no]} · 동시 {concurrency}"
             f" | 장부 ${usd:.2f}·{n}회 | 상한 ${cap.usd:.2f}·{cap.runs}회 | 예상 ${len(todo) * E.EST_USD_PER_RUN:.2f}")
    problem = cap_problem(rows, cap, len(todo))
    if problem:
        raise RunnerError(problem)
    if not yes:
        raise RunnerError("--yes 가 없어 시작하지 않는다(위 계획을 보고 다시 실행)")
    plan_order(ctx, round_no, docs, write=True)
    if not todo:
        ctx.echo("남은 회차가 없다")
        return {"round": round_no, "ran": 0, "stopped": [], "gates": {}}
    if client is None:
        if ctx.mode == "dev":
            from dev_fake_llm_server import FakeOpenAI

            client = FakeOpenAI("pass")
        else:
            client = jl.make_clients(live=True).openai
            if client is None:
                raise RunnerError("OPENAI_API_KEY 가 없다")
    out = asyncio.run(run_round(ctx, round_no, order, todo, bases, client=client, judge=judge or jv.judge_one,
                                concurrency=concurrency, cap=cap, meta=jl.run_meta()))
    gates = Counter(r["gate"] or ("재시도 대상" if retryable(r["error"]) else "최종 실패") for r in out["rows"])
    return {"round": round_no, "ran": len(out["rows"]), "stopped": out["stopped"], "gates": dict(gates),
            "run_ids": [r["run_id"] for r in out["rows"]]}


def cmd_status(ctx: Ctx) -> dict:
    if ctx.mode == "verify":
        load_env()  # 상한 표시가 run 과 같게
    rows = jl.Ledger(ctx.ledger_path).rows() if ctx.ledger_path.exists() else []
    fp = criteria.fingerprint()
    out = {}
    for stage in (*STAGE_OF_ROUND.values(), B2_STAGE):
        st = stage_states(rows, stage, fingerprint=fp, mode=ctx.runner_mode)
        c = Counter(v["status"] for v in st.values())
        out[stage] = {"done": c["done"], "final_failure": c["final"], "retry": c["retry"],
                      "attempts": sum(v["attempts"] for v in st.values())}
        ctx.echo(f"{stage}: 끝남 {c['done']} · 최종 실패 {c['final']} · 다시 돌 것 {c['retry']} · 줄 {out[stage]['attempts']}")
    usd, n = spent(rows)
    cap = Cap.from_env(ctx.mode)
    out["ledger"] = {"usd": round(usd, 6), "rows": n, "cap_usd": cap.usd, "cap_runs": cap.runs}
    ctx.echo(f"장부 누계 ${usd:.2f}·{n}회 | 상한 ${cap.usd:.2f}·{cap.runs}회")
    return out


# ── 베이스라인 1(1차 점검만, §5) ───────────────────────────────────────────


def cmd_baseline1(ctx: Ctx, *, echo: Callable[[str], None] | None = None) -> dict:
    """33문서에 1차 점검(코드, 비용 0) → 05_결과/baseline1.json. verify 는 동결(precheck·v3 해시) 뒤에만(§3.8 — 결함본에 1차 점검을
    미리 돌리지 않는다)."""
    from app.judge.validation.sheets import slot_of_path

    echo = echo or ctx.echo
    bases = load_bases(ctx)
    docs = documents(bases, load_items(ctx))
    if ctx.mode == "verify":
        problems = full_set_problems(docs) + frozen_problems(ctx)
        if problems:
            raise RunnerError("베이스라인 1 을 돌리지 않는다(동결 전):\n  - " + "\n  - ".join(problems))
    checks = {tag: FirstCheck(b) for tag, b in bases.items()}
    out_docs = {}
    for doc in docs:
        base = bases[doc.report]
        index = doc_index(base, doc.slots)
        findings = checks[doc.report].findings(doc_result(base, doc.slots), index)
        entry = {"report": doc.report, "variant": doc.variant, "source": doc.source, "sentences": len(index.sentences),
                 "n_findings": len(findings), "findings": findings}
        if doc.is_defect:
            inj = injected_of(doc, base, index)
            det = E.baseline1_detection(findings, inj, [slot_of_path(doc.meta["slot"])])
            entry.update({"injected": inj, "status": det["status"], "detected": det["detected"],
                          "first_check_pass_by_design": doc.meta.get("first_check_pass_by_design")})
        out_docs[doc.id] = entry
        echo(f"{doc.id} 1차 점검 검출 {len(findings)}건" + (f" — {entry['status']}" if doc.is_defect else ""))
    out = {"mode": ctx.mode, "criteria_version": criteria.version(), "fingerprint": criteria.fingerprint(),
           "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "docs": out_docs}
    target = ctx.path(RESULTS_OUT, "baseline1.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


# ── 베이스라인 2(문서 전체 판정, §5) ─────────────────────────────────────────

B2_TIMEOUT_SECONDS = 300.0
B2_PLACEHOLDER_HEADS = ("[문서]", "[데이터]", "[고객 입력]")
_FENCE = re.compile(r"^```[A-Za-z]*\s*$")
_B2_MODEL = re.compile(r"모델:\s*`([^`]+)`\s*,\s*추론 강도\s*`([^`]+)`")


def _fenced(section: str) -> str:
    lines = section.splitlines()
    start = next((i for i, ln in enumerate(lines) if _FENCE.match(ln)), None)
    if start is None:
        raise RunnerError("베이스라인 2 프롬프트 파일에서 코드 블록을 찾지 못했다")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == "```"), None)
    if end is None:
        raise RunnerError("베이스라인 2 프롬프트 파일의 코드 블록이 닫히지 않았다")
    return "\n".join(lines[start + 1:end])


def load_b2_prompt(ctx: Ctx) -> dict:
    """사전등록/v1/베이스라인2_프롬프트.md — 먼저 같은 폴더 SHA256SUMS 와 맞대 보고(다르면 거부), 지시문·입력 틀·스키마·모델을 읽는다."""
    folder = ctx.path(PREREG_V1)
    path, sums = folder / B2_PROMPT, folder / "SHA256SUMS"
    if not path.exists() or not sums.exists():
        raise RunnerError(f"베이스라인 2 프롬프트 또는 SHA256SUMS 가 없다: {folder}")
    digest = sha256_file(path)
    if read_sums(sums).get(B2_PROMPT) != digest:
        raise RunnerError(f"{B2_PROMPT} 의 sha256 이 사전 등록 SHA256SUMS 와 다르다({digest[:12]}…) — 돌리지 않는다")
    text = path.read_text(encoding="utf-8")
    parts = {}
    for chunk in re.split(r"(?m)^## ", text)[1:]:
        title, _, body = chunk.partition("\n")
        parts[title.strip()] = body

    def part(prefix: str) -> str:
        found = [body for title, body in parts.items() if title.startswith(prefix)]
        if len(found) != 1:
            raise RunnerError(f"베이스라인 2 프롬프트 파일에 '## {prefix}' 절이 하나가 아니다")
        return _fenced(found[0])

    template = part("입력")
    lines = template.splitlines()
    heads = [i for i, ln in enumerate(lines) if ln.strip() in B2_PLACEHOLDER_HEADS]
    if [lines[i].strip() for i in heads] != list(B2_PLACEHOLDER_HEADS) or any(
            i + 1 >= len(lines) or not (lines[i + 1].startswith("<") and lines[i + 1].rstrip().endswith(">")) for i in heads):
        raise RunnerError("베이스라인 2 입력 틀이 [문서]·[데이터]·[고객 입력] 과 그 아래 <자리> 줄 꼴이 아니다")
    m = _B2_MODEL.search(text)
    if not m:
        raise RunnerError("베이스라인 2 프롬프트 파일에서 모델·추론 강도를 읽지 못했다")
    return {"instructions": part("지시문"), "template": template, "schema": json.loads(part("출력 스키마")),
            "model": m.group(1), "effort": m.group(2), "sha256": digest}


def b2_input(prompt: dict, base: jv.Base, slots: dict | None) -> str:
    """입력 틀의 세 자리 줄을 차례로 채운다 — reading.render(조각)(판정자가 읽는 글, 머리 줄 없음), fact_sheet JSON, client_inputs JSON."""
    from app.judge.reading import render

    fills = [render(doc_pieces(base, slots)), json.dumps(base.request.get("fact_sheet"), ensure_ascii=False),
             json.dumps(base.request.get("client_inputs"), ensure_ascii=False)]
    lines = prompt["template"].splitlines()
    for k, i in enumerate([i for i, ln in enumerate(lines) if ln.strip() in B2_PLACEHOLDER_HEADS]):
        lines[i + 1] = fills[k]
    return "\n".join(lines)


def b2_shape_ok(data: dict) -> bool:
    """스키마 꼴 확인(구조화 출력이 지키지만 다시 본다) — decision·issues(axis·type·quotes 1~2개·reason)."""
    if data.get("decision") not in ("issue", "block") or not isinstance(data.get("issues"), list):
        return False
    for i in data["issues"]:
        q = i.get("quotes") if isinstance(i, dict) else None
        if (not isinstance(i, dict) or i.get("axis") not in ("faithfulness", "consistency") or not isinstance(q, list)
                or not 1 <= len(q) <= 2 or not all(isinstance(x, str) for x in q)):
            return False
    return True


class DevBaseline2Fake:
    """dev 전용 가짜 OpenAI — 베이스라인 2 형식에 '문제 없음' 답(실호출 없음)."""

    def __init__(self):
        self.responses = self
        self.calls = 0

    def with_options(self, **_kw):
        return self

    async def create(self, **kwargs):
        from openai.types.responses import Response

        self.calls += 1
        text = json.dumps({"decision": "issue", "issues": []})
        return Response.model_validate({
            "id": f"resp_b2_{self.calls}", "object": "response", "created_at": 1, "model": kwargs["model"], "status": "completed",
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "output": [{"id": f"msg_b2_{self.calls}", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 1050}})


def b2_file(ctx: Ctx, doc: Doc) -> Path:
    return ctx.path(RUNS_OUT, f"B2-{doc.id}.json")


async def run_baseline2(ctx: Ctx, todo: list[Doc], bases: dict[str, jv.Base], prompt: dict, *, client, concurrency: int,
                        cap: Cap, meta: dict) -> dict:
    from app.judge import llm

    writer = LedgerWriter(ctx.ledger_path)
    fmt = llm.json_schema_format("baseline2", prompt["schema"])

    async def one(doc: Doc) -> dict:
        text = b2_input(prompt, bases[doc.report], doc.slots)
        out = await llm.create_response(client, purpose="baseline2", model=prompt["model"], instructions=prompt["instructions"],
                                        input=[{"role": "user", "content": text}], timeout_s=B2_TIMEOUT_SECONDS,
                                        text_format=fmt, effort=prompt["effort"])
        data, failure = llm.parse_json(out)
        if data is not None and not b2_shape_ok(data):
            data, failure = None, "bad_output"
        error = failure or ""

        def build(run_id: str, rows: list[dict]):
            attempt = 1 + sum(1 for r in rows if r.get("stage") == B2_STAGE and r.get("report") == doc.report
                              and r.get("variant") == doc.variant and r.get("mode") == ctx.runner_mode)
            body = {"run_id": run_id, "key": {"stage": B2_STAGE, "report": doc.report, "variant": doc.variant, "doc": doc.id,
                                              "source": doc.source},
                    "attempt": attempt, "mode": ctx.mode, **meta, "narrative_sha256": narrative_sha(bases[doc.report], doc.slots),
                    "prompt_sha256": prompt["sha256"], "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "input_chars": len(text), "call": out.record.as_dict(), "failure": failure, "result": data}
            body_text = json.dumps(body, ensure_ascii=False, indent=1, default=str) + "\n"
            target = b2_file(ctx, doc) if not retryable(error) else ctx.path(RUNS_OUT, f"B2-{doc.id}-{run_id}.json")
            row = {**jl.ledger_row(out, stage=B2_STAGE, report=doc.report, variant=doc.variant, repeat=1, meta=meta),
                   "mode": ctx.runner_mode, "gate": {"block": "blocked", "issue": "pass"}.get((data or {}).get("decision"), ""),
                   "result_file": target.relative_to(ctx.root).as_posix(),
                   "result_sha256": hashlib.sha256(body_text.encode("utf-8")).hexdigest(), "error": error}
            return row, [(target, body_text)]

        return await writer.commit(build)

    out = await launch(ctx, writer, todo, one, concurrency=concurrency, cap=cap, what="호출을")
    for r in out["rows"]:
        ctx.echo(f"  {r['run_id']} baseline2 {E.doc_id(r['report'], r['variant'])} — {r['gate'] or '실패 ' + r['error']}")
    return out


def cmd_baseline2(ctx: Ctx, *, live: bool = False, yes: bool = False, limit: int | None = None,
                  concurrency: int = MAX_CONCURRENCY, client=None, git=jv._git) -> dict:
    """문서마다 1회(§5). 이미 끝난 문서(B2-<문서>.json)는 건너뛰고 API 실패만 다시 부른다. 순서는 1회차 시드 순서."""
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise RunnerError(f"동시 실행은 1~{MAX_CONCURRENCY}개(§7.2): {concurrency}")
    prompt = load_b2_prompt(ctx)
    if ctx.mode == "verify":
        load_env()  # 검사·상한 전에(cmd_run 과 같음)
        problems = eval_verify_problems(live=live, ledger_path=ctx.ledger_path, git=git)
        if (prompt["model"], prompt["effort"]) != (E.PREREG_JUDGE_MODEL, E.PREREG_JUDGE_EFFORT):
            problems.append(f"베이스라인 2 모델 설정이 판정자와 다르다: {prompt['model']}·{prompt['effort']}")
        if problems:
            raise RunnerError("verify 를 시작하지 않는다:\n  - " + "\n  - ".join(problems))
    bases = load_bases(ctx)
    docs = documents(bases, load_items(ctx))
    if ctx.mode == "verify":
        problems = full_set_problems(docs) + frozen_problems(ctx)
        if problems:
            raise RunnerError("베이스라인 2 를 시작하지 않는다:\n  - " + "\n  - ".join(problems))
    rows = jl.Ledger(ctx.ledger_path).rows() if ctx.ledger_path.exists() else []
    states = stage_states(rows, B2_STAGE, fingerprint=criteria.fingerprint(), mode=ctx.runner_mode)
    order = plan_order(ctx, 1, docs, write=False)
    todo = [d for d in order if not finished(states.get((d.report, d.variant)))]
    if limit is not None:
        todo = todo[:max(limit, 0)]
    cap = Cap.from_env(ctx.mode)
    usd, n = spent(rows)
    done = sum(finished(states.get((d.report, d.variant))) for d in order)
    ctx.echo(f"[{ctx.mode}] baseline2: 문서 {len(order)} · 끝남 {done}"
             f" · 이번 {len(todo)}회 · 모델 {prompt['model']}·{prompt['effort']} | 장부 ${usd:.2f}·{n}회 | 상한 ${cap.usd:.2f}·{cap.runs}회")
    problem = cap_problem(rows, cap, len(todo))
    if problem:
        raise RunnerError(problem)
    if not yes:
        raise RunnerError("--yes 가 없어 시작하지 않는다(위 계획을 보고 다시 실행)")
    if not todo:
        ctx.echo("남은 문서가 없다")
        return {"ran": 0, "stopped": [], "gates": {}}
    if client is None:
        client = DevBaseline2Fake() if ctx.mode == "dev" else jl.make_clients(live=True).openai
        if client is None:
            raise RunnerError("OPENAI_API_KEY 가 없다")
    out = asyncio.run(run_baseline2(ctx, todo, bases, prompt, client=client, concurrency=concurrency, cap=cap,
                                    meta=jl.run_meta()))
    gates = Counter(r["gate"] or ("재시도 대상" if retryable(r["error"]) else "최종 실패") for r in out["rows"])
    return {"ran": len(out["rows"]), "stopped": out["stopped"], "gates": dict(gates)}


# ── 집계(§6) ────────────────────────────────────────────────────────────

FINAL = ("done", "final")
NOT_RUN = "미실행"


def _load_result(ctx: Ctx, row: dict) -> dict:
    path = ctx.root / row["result_file"]
    if ctx.mode == "dev":
        path = dev_guard(path)
    body = path.read_bytes()
    if row.get("result_sha256") and hashlib.sha256(body).hexdigest() != row["result_sha256"]:
        raise RunnerError(f"결과 원본 해시가 장부와 다르다: {row['result_file']}")
    return json.loads(body)


def collect_runs(ctx: Ctx, docs: list[Doc], rows: list[dict]) -> dict[tuple[int, str], dict]:
    """(회차, 문서) → {status: done|final|retry|미실행, run: 결과 원본(끝난 회차만), attempts, row}."""
    fp = criteria.fingerprint()
    out = {}
    for round_no, stage in STAGE_OF_ROUND.items():
        states = stage_states(rows, stage, fingerprint=fp, mode=ctx.runner_mode)
        for doc in docs:
            s = states.get((doc.report, doc.variant))
            if s is None:
                out[(round_no, doc.id)] = {"status": NOT_RUN, "run": None, "attempts": 0, "row": None}
                continue
            run = _load_result(ctx, s["row"]) if s["status"] in FINAL else None
            out[(round_no, doc.id)] = {"status": s["status"], "run": run, "attempts": s["attempts"], "row": s["row"]}
    return out


def collect_b2(ctx: Ctx, docs: list[Doc], rows: list[dict]) -> dict[str, dict]:
    states = stage_states(rows, B2_STAGE, fingerprint=criteria.fingerprint(), mode=ctx.runner_mode)
    out = {}
    for doc in docs:
        s = states.get((doc.report, doc.variant))
        if s is None:
            out[doc.id] = {"status": NOT_RUN, "body": None, "attempts": 0, "row": None}
        else:
            out[doc.id] = {"status": s["status"], "body": _load_result(ctx, s["row"]) if s["status"] in FINAL else None,
                           "attempts": s["attempts"], "row": s["row"]}
    return out


def ledger_costs(rows: list[dict], mode: str) -> dict:
    """단계별 줄 수·비용 합·평균 시간·재시도(같은 문서·회차의 두 번째 이후 줄)·API 실패 줄."""
    out = {}
    fp = criteria.fingerprint()
    for stage in (*STAGE_OF_ROUND.values(), B2_STAGE):
        mine = [r for r in rows if r.get("stage") == stage and r.get("mode") == mode and r.get("manifest_sha256") == fp]
        secs = [float(r["seconds"]) for r in mine if (r.get("seconds") or "").strip()]
        keys = Counter((r["report"], r["variant"]) for r in mine)
        out[stage] = {"rows": len(mine), "cost_usd": round(sum(float(r["cost_usd"]) for r in mine if (r.get("cost_usd") or "").strip()), 6),
                      "unknown_cost_rows": sum(1 for r in mine if not (r.get("cost_usd") or "").strip()),
                      "mean_seconds": round(sum(secs) / len(secs), 3) if secs else None,
                      "retries": sum(n - 1 for n in keys.values()),
                      "api_failures": sum(1 for r in mine if retryable(r.get("error", ""))),
                      "models_served": sorted({m for r in mine for m in (r.get("model_served") or "").split(",") if m})}
    out["total"] = {"rows": sum(v["rows"] for v in out.values()), "cost_usd": round(sum(v["cost_usd"] for v in out.values()), 6)}
    return out


class World:
    """집계에 쓰는 재료 한 벌 — 바탕·문서·문장 목록·주입 문장·회차 결과·베이스라인."""

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.bases = load_bases(ctx)
        self.docs = documents(self.bases, load_items(ctx))
        if not self.docs:
            raise RunnerError("문서가 없다")
        self.index = {d.id: doc_index(self.bases[d.report], d.slots) for d in self.docs}
        self.injected = {d.id: injected_of(d, self.bases[d.report], self.index[d.id]) for d in self.docs if d.variant != E.BASE}
        # v3 2.1 결함의 원래 문장 — 원본 문서의 문장 번호(덧붙이기 결함은 빈 목록)
        self.original_ids = {d.id: original_ids_of(d, self.bases[d.report], self.index[d.report]) for d in self.docs if d.is_defect}
        self.rows = jl.Ledger(ctx.ledger_path).rows() if ctx.ledger_path.exists() else []
        self.runs = collect_runs(ctx, self.docs, self.rows)
        self.b2 = collect_b2(ctx, self.docs, self.rows)
        b1 = ctx.root / RESULTS_OUT / "baseline1.json"
        self.b1 = json.loads(b1.read_text(encoding="utf-8")) if b1.exists() else None
        self._sources: dict = {}

    def run(self, doc: Doc, round_no: int) -> dict | None:
        return self.runs[(round_no, doc.id)]["run"]

    def pair(self, doc: Doc) -> list[dict | None]:
        return [self.run(doc, 1), self.run(doc, 2)]

    def original(self, report: str) -> Doc:
        return next(d for d in self.docs if d.report == report and d.variant == E.BASE)

    def missing(self) -> list[str]:
        return [f"{doc.id} {n}회차({self.runs[(n, doc.id)]['status']})" for n in (1, 2) for doc in self.docs
                if self.runs[(n, doc.id)]["status"] not in FINAL]

    def value_of(self, report: str):
        if report not in self._sources:
            self._sources[report] = sources_of(self.bases[report])
        src = self._sources[report]

        def lookup(name: str):
            got = src.lookup(name)
            return got.get("value") if got.get("found", True) else None
        return lookup


def _status_of(world: World, doc: Doc, n: int) -> str:
    st = world.runs[(n, doc.id)]["status"]
    return {"done": "완료", "final": "실패", "retry": "재시도 대상", NOT_RUN: NOT_RUN}[st]


def _flag_detail(f: dict) -> dict:
    return {"index": f["index"], "axis": f["axis"], "label": f["label"], "slots": f["slots"], "quotes": f["quotes"],
            "field": f["field"], **({"sentence_ids": f["sentence_ids"]} if "sentence_ids" in f else {})}


def aggregate(ctx: Ctx, *, partial: bool = False, world: World | None = None, blind: bool = True) -> dict:
    """§6 집계 전부(규칙에 없는 것은 넣지 않는다). partial 이 아니면 33문서 × 2회차가 모두 끝나고 베이스라인 1 이 있어야 한다.
    blind(판정 전 — aggregate 명령)는 판정표 문항과 출처를 잇는 것(표시 인용·위치·종류·필드, 문서·회차별 표시 목록과 표시 수, 재현성
    검출 문장 집합, 미탐지 원인의 모순 인용)을 싣지 않는다(§6.4). report 는 blind=False 로 모두 싣는다."""
    w = world or World(ctx)
    missing = w.missing()
    if not partial and (missing or w.b1 is None):
        raise RunnerError("집계할 회차가 덜 끝났다(--partial 로 부분 집계): "
                          + ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "") + ("" if w.b1 else " · baseline1.json 없음"))
    defects_docs = [d for d in w.docs if d.is_defect]
    variant_docs = [d for d in w.docs if d.source == E.SOURCE_VARIANT]
    originals = {tag: w.pair(w.original(tag)) for tag in w.bases}

    def run_entry(d: Doc, n: int) -> dict:
        r, info = w.run(d, n), w.runs[(n, d.id)]
        entry = {"round": n, "status": _status_of(w, d, n), "attempts": info["attempts"], "run_id": (info["row"] or {}).get("run_id"),
                 "model_served": (info["row"] or {}).get("model_served"), "at": (info["row"] or {}).get("at"),
                 "gate": (r or {}).get("gate"), "failure": (r or {}).get("failure"), "error": (info["row"] or {}).get("error") or None,
                 "claims": len((((r or {}).get("record") or {}).get("extract") or {}).get("claims") or []),
                 "scores": E.axis_scores(r) if E.run_ok(r) else None}
        if not blind:
            entry["flags"] = [_flag_detail(E.locate_flag(f, w.index[d.id])) for f in E.flags_of(r)] if E.run_ok(r) else []
        return entry

    out: dict = {"mode": ctx.mode, "blind": blind, "partial": bool(partial and (missing or w.b1 is None)), "missing": missing,
                 "criteria_version": criteria.version(), "fingerprint": criteria.fingerprint(),
                 "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "docs": {d.id: {"report": d.report, "variant": d.variant, "source": d.source, "sentences": len(w.index[d.id].sentences),
                                 "runs": [run_entry(d, n) for n in (1, 2)]} for d in w.docs}}

    # 6.2·6.3·6.7·6.8 결함본 (+ v3 2.1 원본에서 이미 표시)
    defects = {}
    for doc in defects_docs:
        idx, inj, meta = w.index[doc.id], w.injected[doc.id], doc.meta
        expected = list(meta["expected_axes"])
        fields = (meta.get("grounds") or {}).get("field")
        dets = [E.detect(r, idx, inj, expected) for r in w.pair(doc)]
        causes = [{"round": n, **E.miss_cause(doc.variant, w.run(doc, n), idx, inj, fields)}
                  for n, d in zip((1, 2), dets) if d["gate_detected"] is False]
        axis_causes = [{"round": n, **E.miss_cause(doc.variant, w.run(doc, n), idx, inj, fields, axis=a)}
                       for n, d in zip((1, 2), dets) for a in expected if d["axes"][a] is False]
        if blind:
            for c in causes + axis_causes:
                c["contradictions"] = len(c["contradictions"])  # 위치 불일치 모순의 인용은 판정표 문항이다
        pre = E.original_flagged(originals[doc.report], w.index[doc.report], w.original_ids[doc.id])
        defects[doc.id] = {
            "report": doc.report, "type": meta["type"], "axis": meta["axis"], "expected_axes": expected,
            "allowed_secondary_axes": meta["allowed_secondary_axes"], "injected": inj,
            "first_check_pass_by_design": meta.get("first_check_pass_by_design"),
            "original_flagged": {k: pre[k] for k in ("applies", "gate", "axes", "checked_runs")},
            "gate": [E.word(d["gate_detected"], d["status"]) for d in dets],
            "gate_class": E.two_run_class([d["gate_detected"] for d in dets]),
            "axes": {a: {"runs": [E.word(d["axes"][a], d["status"]) for d in dets],
                         "class": E.two_run_class([d["axes"][a] for d in dets])} for a in expected},
            "out_of_place": [{"round": n, "count": len(d["out_of_place"]),
                              **({} if blind else {"flags": [_flag_detail(f) for f in d["out_of_place"]]})}
                             for n, d in zip((1, 2), dets)],
            "miss_causes": causes, "axis_miss_causes": axis_causes,
            "independence": E.independence(originals[doc.report], w.pair(doc), expected, meta["allowed_secondary_axes"],
                                           E.EXPECTED_CHANGES.get(doc.variant, ())),
            "extraction": [E.extraction_hit(doc.variant, r, idx, inj) for r in w.pair(doc)],
        }
    # v3 2.1 — 원본에서 이미 표시된 결함은 분모에서 빼고 따로(게이트 수준: 차단 축 어느 것이든, 축 수준: 그 기대 축)
    counted = {v: d for v, d in defects.items() if not d["original_flagged"]["gate"]}
    both = sum(d["gate_class"] == E.BOTH for d in counted.values())
    by_type: dict[str, list] = {}
    for vid, d in defects.items():
        by_type.setdefault(d["type"], []).append({"defect": vid, "class": d["gate_class"] if vid in counted else E.ORIGINAL_FLAGGED})
    out["q1_gate"] = {"defects": len(defects), "counted": len(counted),
                      "original_flagged": sorted(set(defects) - set(counted)),
                      "classes": dict(Counter(d["gate_class"] for d in counted.values())),
                      "both_detected": E.wilson(both, len(counted)), "by_type": by_type}
    by_axis = {}
    for a in E.GATE_AXES:
        mine = {v: d for v, d in defects.items() if a in d["axes"]}
        kept = {v: d for v, d in mine.items() if not d["original_flagged"]["axes"][a]}
        by_axis[a] = {"defects": len(mine), "counted": len(kept),
                      "original_flagged": sorted(set(mine) - set(kept)),
                      "both_detected": sum(d["axes"][a]["class"] == E.BOTH for d in kept.values()),
                      "any_run_missed": sorted(v for v, d in kept.items() if any(x == E.MISSED for x in d["axes"][a]["runs"]))}
    out["q2_axis"] = {"by_axis": by_axis}
    out["defects"] = defects
    related = [v for vid in E.EXTRACTION_IDS if vid in defects for v in defects[vid]["extraction"]]
    reference = [v for vid, d in defects.items() if E.RELATED_KINDS[vid] is None for v in d["extraction"]]
    out["extraction_rate"] = {"related": E.rate(related), "reference_c02_c09": E.rate(reference)}

    # 6.6 불변성 (v3 2.4 — V02·V06 은 결정과 차단 축만)
    variants = {}
    for doc in variant_docs:
        judged = E.GATE_AXES if doc.variant in E.INVARIANCE_GATE_ONLY else E.AXES
        inv = E.invariance(originals[doc.report], w.pair(doc), judged)
        variants[doc.id] = {"report": doc.report, "type": doc.meta.get("type"), **inv}
    out["q3_invariance"] = {"variants": variants, "by_type": {
        t: dict(Counter(v["status"] for v in variants.values() if v["type"] == t)) for t in E.VARIANT_TYPES}}
    blocked = [{"doc": d.id, "round": n, "run_id": (w.runs[(n, d.id)]["row"] or {}).get("run_id"),
                "flags": len(E.flags_of(w.run(d, n)))}
               for d in w.docs if d.source in (E.SOURCE_ORIGINAL, E.SOURCE_VARIANT) for n in (1, 2)
               if (w.run(d, n) or {}).get("gate") == "blocked"]
    out["q3_blocked_before_judgement"] = {"count": len(blocked), **({} if blind else {"blocked_runs": blocked})}

    # 6.5 재현성
    rep = E.reproducibility({d.id: w.pair(d) for d in w.docs}, w.index)
    if blind:
        rep["docs"] = {doc: {k: v for k, v in x.items() if k != "sets"} for doc, x in rep["docs"].items()}
    out["q4_reproducibility"] = rep

    # 5 베이스라인
    b1 = (w.b1 or {}).get("docs", {})
    b2 = {}
    for doc in w.docs:
        st = w.b2[doc.id]["status"]
        result = w.b2[doc.id]["body"]["result"] if st == "done" else None
        if st != "done":
            b2[doc.id] = {"status": {"final": E.FAILED, "retry": "재시도 대상"}.get(st, NOT_RUN), "detected": None, "issues": None,
                          "decision": None}
        elif doc.is_defect:
            det = E.baseline2_detection(result, w.index[doc.id], w.injected[doc.id])
            b2[doc.id] = {k: det.get(k) for k in ("status", "detected", "issues", "decision")}
        else:
            b2[doc.id] = {"status": "완료", "detected": None, "issues": len(E.baseline2_issues(result, w.index[doc.id])),
                          "decision": result.get("decision")}
    out["q5_baselines"] = {
        "defects": {vid: {"first_check_pass_by_design": d["first_check_pass_by_design"],
                          "baseline1": (b1.get(vid) or {}).get("status", NOT_RUN), "baseline2": b2[vid]["status"],
                          "judge_gate": d["gate_class"]} for vid, d in defects.items()},
        "originals_variants": {d.id: {"baseline1_findings": (b1.get(d.id) or {}).get("n_findings"), "baseline2_issues": b2[d.id]["issues"],
                                      "baseline2_decision": b2[d.id]["decision"],
                                      **({} if blind else {"judge_flags": [len(E.flags_of(r)) if E.run_ok(r) else None
                                                                           for r in w.pair(d)]})}
                               for d in w.docs if not d.is_defect},
        "baseline1_present": w.b1 is not None, "baseline2": b2}

    # 6.9 숫자 토큰 재현율 — 원본 3건 × 2회
    recall_rows = {}
    for tag in sorted(w.bases):
        doc = w.original(tag)
        for n, r in zip((1, 2), w.pair(doc)):
            if E.run_ok(r):
                got = E.numeric_recall(r, w.index[doc.id])
                recall_rows[f"{doc.id} {n}회차"] = {k: got[k] for k in ("tokens", "matched", "recall")}
    tokens = sum(v["tokens"] for v in recall_rows.values())
    matched = sum(v["matched"] for v in recall_rows.values())
    out["numeric_recall"] = {"runs": recall_rows, "tokens": tokens, "matched": matched, "recall": matched / tokens if tokens else None}
    out["costs"] = ledger_costs(w.rows, ctx.runner_mode)
    return out


# ── 보고 글(마크다운) ──────────────────────────────────────────────────────


def _cell(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_cell(x) for x in v) or "—"
    return str(v).replace("|", "\\|").replace("\n", " ")


def md_table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "(없음)\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def aggregate_markdown(agg: dict) -> str:
    q1, q4 = agg["q1_gate"], agg["q4_reproducibility"]
    wil = q1["both_detected"]
    parts = [f"# 판정자 평가 집계 (사전 등록 v1 §6 · v3 2절)\n\n- 모드 {agg['mode']} · 기준 {agg['criteria_version']} · 지문 {agg['fingerprint'][:8]}… · {agg['at']}\n"
             + (f"- **부분 집계** — 덜 끝난 회차: {', '.join(agg['missing']) or '없음(베이스라인 1 없음)'}\n" if agg["partial"] else "")
             + ("- 판정 전 블라인드 집계 — 표시 인용·위치와 회차별 차단 목록은 판정 뒤 최종보고에만 싣는다(§6.4).\n" if agg["blind"] else "")]
    parts.append("\n## Q1 게이트 탐지 (6.2)\n\n" + md_table(
        ["결함", "결과서", "유형", "1회차", "2회차", "분류", "위치 불일치 표시(1·2회차)", "원본에서 이미 표시(v3 2.1)"],
        [[v, d["report"], d["type"], d["gate"][0], d["gate"][1], d["gate_class"],
          f"{d['out_of_place'][0]['count']}·{d['out_of_place'][1]['count']}",
          d["original_flagged"]["gate"] if d["original_flagged"]["applies"] else "해당 없음"] for v, d in agg["defects"].items()]))
    parts.append(f"\n두 회차 모두 탐지 {wil['k']}/{wil['n']}"
                 + (f" — Wilson 95% [{wil['low']:.2f}, {wil['high']:.2f}]" if wil["low"] is not None else "")
                 + (f" (분모에서 뺀 '{E.ORIGINAL_FLAGGED}' 결함: {', '.join(q1['original_flagged'])})" if q1["original_flagged"] else "")
                 + "\n")
    parts.append("\n유형별(표본 2~3개 — 기술 통계만)\n\n" + md_table(
        ["유형", "결함·분류"], [[t, ", ".join(f"{x['defect']} {x['class']}" for x in xs)] for t, xs in q1["by_type"].items()]))
    parts.append("\n## Q2 축 수준 탐지 (6.2)\n\n" + md_table(
        ["결함", "기대 축", "1회차", "2회차", "분류", "원본에서 이미 표시(v3 2.1)"],
        [[v, a, x["runs"][0], x["runs"][1], x["class"],
          d["original_flagged"]["axes"][a] if d["original_flagged"]["applies"] else "해당 없음"]
         for v, d in agg["defects"].items() for a, x in d["axes"].items()]))
    parts.append("\n" + md_table(["축", "기대 축 결함 수", "분모(원본에서 이미 표시 뺌)", "두 회차 모두 탐지", "한 회차라도 축 수준 미탐지",
                                   "원본에서 이미 표시"],
                                 [[a, x["defects"], x["counted"], x["both_detected"], x["any_run_missed"], x["original_flagged"]]
                                  for a, x in agg["q2_axis"]["by_axis"].items()]))
    parts.append("\n## 독립성 (6.7) — 기대 축이 아닌 축, 원본 두 회차 범위 ± 허용 폭\n\n" + md_table(
        ["결함", "축", "원본 두 회차", "결함본 두 회차", "판정"],
        [[v, a, x["base"], x["defect"], x["status"]] for v, d in agg["defects"].items() for a, x in d["independence"].items()]))
    parts.append("\n## 미탐지 원인 (6.3) — 게이트 수준·축 수준에서 놓친 회차\n\n" + md_table(
        ["결함", "회차", "수준", "원인", "근거"],
        [[v, c["round"], "게이트" if c["axis"] is None else f"축 {c['axis']}", c["cause"],
          c["note"] or "; ".join(f"{x['kind']} '{x['quote']}' 필드 {x['field'] or '-'} 판정 {x['verdict'] or '-'}" for x in c["claims"][:3])]
         for v, d in agg["defects"].items() for c in d["miss_causes"] + d["axis_miss_causes"]]))
    parts.append("\n## Q3 불변성 (6.6)\n\n" + md_table(
        ["변형", "결과서", "유형", "원본 결정", "변형 결정", "판정", "벗어난 축", "보고만 하는 축의 벗어남(v3 2.4)"],
        [[v, x["report"], x["type"], x["base_gates"], x["variant_gates"], x["status"],
          [a for a, s in x["axes"].items() if s["ok"] is False and s["judged"]],
          [a for a, s in x["axes"].items() if s["ok"] is False and not s["judged"]]]
         for v, x in agg["q3_invariance"]["variants"].items()]))
    parts.append("\n변형 유형별\n\n" + md_table(["유형", "판정별 수"], [[t, ", ".join(f"{k} {v}" for k, v in c.items()) or "—"]
                                                                  for t, c in agg["q3_invariance"]["by_type"].items()]))
    blocked = agg["q3_blocked_before_judgement"]
    parts.append(f"\n## Q3 과잉 차단 — 판정 전\n\n차단된 원본·변형 회차 {blocked['count']}개(과잉 차단 여부는 판정표 판정 뒤 report 가 센다).\n"
                 + ("" if "blocked_runs" not in blocked else "\n" + md_table(
                     ["문서", "회차", "표시 수"], [[b["doc"], b["round"], b["flags"]] for b in blocked["blocked_runs"]])))
    parts.append("\n## Q4 재현성 (6.5)\n\n" + md_table(
        ["묶음", "문서 수", "발급 결정 일치", "검출 문장 집합 완전 일치", "평균 Jaccard"],
        [["전체", q4["overall"]["n"], q4["overall"]["decision_agreement"], q4["overall"]["exact_rate"], q4["overall"]["mean_jaccard"]]]
        + [[g, x["n"], x["decision_agreement"], x["exact_rate"], x["mean_jaccard"]] for g, x in q4["groups"].items()]))
    if q4["excluded"]:
        parts.append(f"\n실패·미실행 회차가 있어 뺀 문서: {', '.join(q4['excluded'])}\n")
    split = [d for d, x in q4["docs"].items() if x["group"] == "갈림"]
    parts.append(f"\n발급 결정이 회차에 따라 갈린 문서: {', '.join(split) or '없음'}\n")
    q5 = agg["q5_baselines"]
    parts.append("\n## Q5 베이스라인 (§5)\n\n" + md_table(
        ["결함", "설계상 1차 점검 통과", "베이스라인 1", "베이스라인 2", "판정자(게이트)"],
        [[v, x["first_check_pass_by_design"], x["baseline1"], x["baseline2"], x["judge_gate"]] for v, x in q5["defects"].items()]))
    blind = agg["blind"]
    parts.append("\n원본·변형 — 문서당 지적 수\n\n" + md_table(
        ["문서", "베이스라인 1 검출", "베이스라인 2 지적", "베이스라인 2 결정"] + ([] if blind else ["판정자 표시(1·2회차)"]),
        [[d, x["baseline1_findings"], x["baseline2_issues"], x["baseline2_decision"]] + ([] if blind else [x["judge_flags"]])
         for d, x in q5["originals_variants"].items()]))
    ex, nr = agg["extraction_rate"], agg["numeric_recall"]
    parts.append("\n## 보조 측정\n\n" + md_table(
        ["측정", "맞음", "분모", "비율", "기록 없음(실패)"],
        [["주입 문장 추출률 F01~F09·C01(관련 종류, 6.8)", ex["related"]["k"], ex["related"]["n"], ex["related"]["rate"], ex["related"]["no_record"]],
         ["참고: C02~C09 어떤 종류든", ex["reference_c02_c09"]["k"], ex["reference_c02_c09"]["n"], ex["reference_c02_c09"]["rate"],
          ex["reference_c02_c09"]["no_record"]],
         ["숫자 토큰 재현율(원본 × 2회, 6.9)", nr["matched"], nr["tokens"], nr["recall"], None]]))
    costs = agg["costs"]
    parts.append("\n## 비용·시간·재시도\n\n" + md_table(
        ["단계", "장부 줄", "비용($)", "비용 모름 줄", "평균 시간(초)", "재시도", "API 실패 줄", "응답 모델"],
        [[s, c["rows"], c["cost_usd"], c["unknown_cost_rows"], c["mean_seconds"], c["retries"], c["api_failures"], c["models_served"]]
         for s, c in costs.items() if s != "total"]) + f"\n합계 {costs['total']['rows']}줄 · ${costs['total']['cost_usd']:.2f}\n")
    return "".join(parts)


def cmd_aggregate(ctx: Ctx, *, partial: bool = False) -> dict:
    agg = aggregate(ctx, partial=partial)
    folder = ctx.path(RESULTS_OUT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "집계.json").write_text(json.dumps(agg, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    (folder / "집계.md").write_text(aggregate_markdown(agg), encoding="utf-8")
    ctx.echo(f"집계 {folder}/집계.json·집계.md" + (" — 부분 집계" if agg["partial"] else ""))
    return agg


# ── 과잉 차단 판정표(§6.4) ──────────────────────────────────────────────────


def _csv_text(columns, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    w.writerows(rows)
    return buf.getvalue()


def sheet_entries_of(world: World, *, include_b2: bool) -> list[dict]:
    """판정 대상 — 원본·변형의 모든 위반·모순 표시(두 회차), 결함본의 주입 문장 밖 표시, (원하면) 원본·변형의 베이스라인 2 지적.
    앞뒤 문장은 같은 결과서 원본에서(표시 문장이 원본에 있으면 — 바뀐 문장이 앞뒤 칸으로 새지 않게)."""
    entries = []
    for doc in world.docs:
        idx = world.index[doc.id]
        context = world.index[doc.report]
        for n in (1, 2):
            run = world.run(doc, n)
            if not E.run_ok(run):
                continue
            if doc.is_defect:
                located = E.detect(run, idx, world.injected[doc.id], doc.meta["expected_axes"])["out_of_place"]
            else:
                located = [E.locate_flag(f, idx) for f in E.flags_of(run)]
            entries += E.sheet_entries(run["run_id"], doc.id, doc.source, n, located, idx, value_of=world.value_of(doc.report),
                                       context=context)
        body = world.b2[doc.id]["body"]
        if include_b2 and not doc.is_defect and body and not body.get("failure"):
            entries += E.b2_entries(body["run_id"], doc.id, E.SOURCE_B2, E.baseline2_issues(body["result"], idx), idx,
                                    context=context)
    return entries


def cmd_sheet(ctx: Ctx, *, seed: int | None = None, include_b2: bool = False) -> dict:
    """블라인드 판정표·열쇠. 33문서 × 2회차가 끝나야 만든다(표시가 빠지지 않게). 쓰기 전에 새는 것 검사 — 걸리면 쓰지 않는다."""
    target = ctx.path(SHEET_DIR, "판정표.csv")
    if target.exists():
        raise RunnerError(f"판정표가 이미 있다(덮어쓰지 않는다 — 나눠 준 표와 열쇠가 어긋나지 않게): {target}")
    world = World(ctx)
    if world.missing():
        raise RunnerError(f"채점이 덜 끝나 판정표를 만들지 않는다: {', '.join(world.missing()[:8])}")
    seed = random.SystemRandom().randrange(1, 10**9) if seed is None else seed
    entries = sheet_entries_of(world, include_b2=include_b2)
    rows, key_rows, items = E.build_sheet(entries, seed)
    # 출처 낱말 검사에서 걷어 낼 글 — 문서 문장, 데이터 값, 베이스라인 2 의 사유(LLM 이 쓴 글 — '원본 데이터와 다르다' 같은 말은 출처가
    # 아니다). 문서 이름·태그·ID·실행 번호·회차는 이 글들 안이라도 모두 잡는다.
    allowed = ([s.text for idx in world.index.values() for s in idx.sentences] + [e["value"] for e in entries if e["value"]]
               + [e["reason"] for e in entries if e["source"] == E.SOURCE_B2 and e["reason"]])
    leaks = E.sheet_leaks(rows, doc_names=[d.id for d in world.docs] + list(world.bases), allowed_texts=allowed)
    if leaks:
        raise RunnerError("판정표에 출처를 알려 주는 것이 있어 쓰지 않는다:\n  - " + "\n  - ".join(leaks[:20]))
    key_body = _csv_text(E.KEY_COLUMNS, key_rows)
    key_sha = hashlib.sha256(key_body.encode("utf-8")).hexdigest()
    key_dir = ctx.path(SHEET_DIR, "열쇠")
    key_dir.mkdir(parents=True, exist_ok=True)
    (key_dir / "열쇠.csv").write_text(key_body, encoding="utf-8")
    (key_dir / "열쇠.sha256").write_text(f"{key_sha}  열쇠.csv\n", encoding="utf-8")
    at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # 출처별 표시 수와 시드(문항 번호를 다시 만들 수 있다)는 열쇠와 같이 둔다 — 판정을 마친 뒤 연다(§6.4)
    key_record = {"seed": seed, "by_source": dict(Counter(e["source"] for e in entries)), "key_sha256": key_sha, "at": at}
    (key_dir / "기록.json").write_text(json.dumps(key_record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    target.write_text(jv.KEY_HEAD + key_sha + "\n" + _csv_text(E.SHEET_COLUMNS, rows), encoding="utf-8")
    record = {"items": len(rows), "flags": len(entries), "include_baseline2": include_b2, "key_sha256": key_sha, "at": at}
    (ctx.path(SHEET_DIR, "기록.json")).write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for f in (key_dir / "열쇠.csv", key_dir / "열쇠.sha256", key_dir / "기록.json"):
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    ctx.echo(f"판정표 {len(rows)}문항(표시 {len(entries)}개) · 열쇠 sha256 {key_sha} · 시드·출처별 수는 열쇠/기록.json(판정 뒤에 연다)")
    return record


# ── 최종 보고(§6.4 셈) ──────────────────────────────────────────────────────


def integrity(ctx: Ctx) -> dict:
    """봉인(00_봉인/SHA256SUMS)과 사전등록 v3 해시(결함_SHA256SUMS·SHA256SUMS — 있으면) 를 다시 맞대 본다."""
    out: dict = {"sealed": None, "defects_v3": None}
    sums = ctx.root / SEALED / "SHA256SUMS"
    if sums.exists():
        bad = [name for name, digest in read_sums(sums).items()
               if not (ctx.root / SEALED / name).is_file() or sha256_file(ctx.root / SEALED / name) != digest]
        out["sealed"] = {"ok": not bad, "mismatch": bad}
    elif ctx.mode == "verify":
        out["sealed"] = {"ok": False, "mismatch": ["00_봉인/SHA256SUMS 없음"]}
    if any((ctx.root / PREREG_V3 / name).exists() for name in (V3_DEFECT_SUMS, V3_SUMS)):
        bad = defect_sum_problems(ctx)
        out["defects_v3"] = {"ok": not bad, "mismatch": bad}
    return out


def _read_keyed(path: Path) -> tuple[str | None, list[dict]]:
    """judge_validate.read_keyed_sheet 와 같되 BOM(표 프로그램이 UTF-8 로 저장할 때 붙이는 것)을 걷는다."""
    text = Path(path).read_text(encoding="utf-8-sig")
    first, _, rest = text.partition("\n")
    sha = first[len(jv.KEY_HEAD):].strip() if first.startswith(jv.KEY_HEAD) else None
    body = rest if sha is not None else text
    return sha, [r for r in csv.DictReader(io.StringIO(body)) if any((v or "").strip() for v in r.values())]


def read_answers(ctx: Ctx, answers: Path) -> tuple[list[dict], dict[str, str], list[dict]]:
    """판정한 표 → (열쇠 줄, {문항: 판정}, 판정 줄). 열쇠 해시(표 첫 줄·열쇠.sha256·열쇠 파일)가 모두 같아야 하고, 도구가 쓴 칸은
    만든 판정표와 글자까지 같아야 하며, 모든 문항에 판정과 판정 사유가 있어야 한다(판정을 마친 뒤에만 열쇠를 연다)."""
    key_path = ctx.path(SHEET_DIR, "열쇠", "열쇠.csv")
    sheet_path = ctx.path(SHEET_DIR, "판정표.csv")
    if not key_path.exists() or not sheet_path.exists():
        raise RunnerError("판정표·열쇠가 없다 — sheet 를 먼저 만든다")
    sha, rows = _read_keyed(Path(answers))
    jv.check_key(key_path, {"판정한 표": sha}, recorded={"열쇠.sha256": jv._sha_file(key_path.parent / "열쇠.sha256")})
    _, made = jv.read_keyed_sheet(sheet_path)
    made_by_id = {r["문항"]: r for r in made}
    got_by_id = {r.get("문항"): r for r in rows}
    if set(got_by_id) != set(made_by_id):
        raise RunnerError(f"판정한 표의 문항이 만든 판정표와 다르다: 빠짐 {sorted(set(made_by_id) - set(got_by_id))[:5]} · "
                          f"더함 {sorted(set(got_by_id) - set(made_by_id))[:5]}")
    changed = [q for q, r in got_by_id.items() if any(r.get(c, "") != made_by_id[q].get(c, "") for c in E.SHEET_COLUMNS[:9])]
    if changed:
        raise RunnerError(f"판정 칸 밖을 고친 문항이 있다: {', '.join(sorted(changed)[:5])}")
    verdicts = {q: (r.get(E.VERDICT_COLUMN) or "").strip() for q, r in got_by_id.items()}
    bad = sorted(q for q, v in verdicts.items() if v not in (E.REAL_ERROR, E.JUDGE_ERROR))
    no_reason = sorted(q for q, r in got_by_id.items() if not (r.get(E.VERDICT_REASON_COLUMN) or "").strip())
    if bad or no_reason:
        raise RunnerError(f"판정을 마치지 않았다 — 열쇠를 열지 않는다(판정 빈칸·틀린 글 {bad[:5]}, 판정 사유 빈칸 {no_reason[:5]})")
    return jv._csv_rows(key_path), verdicts, rows


def cmd_report(ctx: Ctx, answers: Path) -> dict:
    key_rows, verdicts, filled = read_answers(ctx, answers)
    world = World(ctx)
    agg = aggregate(ctx, partial=False, world=world, blind=False)  # 판정을 마친 뒤 — 표시 세부까지
    runs_info = {}
    for doc in world.docs:
        for n in (1, 2):
            run = world.run(doc, n)
            if E.run_ok(run):
                faith = (run.get("scorecard") or {}).get("faithfulness") or {}
                runs_info[run["run_id"]] = {"doc": doc.id, "source": doc.source, "gate": run["gate"], "checked": faith.get("checked", 0),
                                            "inference_checked": int(faith.get("inference_ok") or 0)
                                            + int(faith.get("inference_violations") or 0)}
    tally = E.tally_overblocking(key_rows, verdicts, runs_info)
    checks = integrity(ctx)
    out = {**agg, "overblocking": tally, "integrity": checks, "answers_file_sha256": sha256_file(Path(answers)),
           "sheet": [{c: r.get(c, "") for c in E.SHEET_COLUMNS} for r in filled]}
    folder = ctx.path(RESULTS_OUT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "최종보고.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    c, cl = tally["counts"], tally["claim_level"]
    md = aggregate_markdown(agg).replace("# 판정자 평가 집계 (사전 등록 v1 §6 · v3 2절)", "# 판정자 평가 최종 보고 (사전 등록 v1 · v3)", 1)
    md += ("\n## Q3 과잉 차단 — 판정 뒤 (6.4)\n\n"
           + md_table(["셈", "값"], [["차단된 원본·변형 회차", c["차단"]], ["과잉 차단(판정자 오판만)", c["과잉 차단"]],
                                    ["자연 발생 오류 검출(실제 오류 있음)", c["자연 발생 오류 검출"]], ["통과한 원본·변형 회차", c["통과"]],
                                    ["주장 단위: 판정자 오판 충실성 위반 / 검사한 주장(6.4 글자 그대로)",
                                     f"{cl['wrong_faithfulness_violations']} / {cl['checked']}"],
                                    ["주장 단위 오판율(6.4 글자 그대로)", cl["rate"]],
                                    ["  나눠 보기 — 사실 일치 위반 오판 / 검사한 주장", f"{cl['violation']['wrong']} / {cl['violation']['checked']}"],
                                    ["  나눠 보기 — 단정형 추론 오판 / 추론 판정(inference_ok + inference_violations)",
                                     f"{cl['inference']['wrong']} / {cl['inference']['checked']}"]])
           + f"\n{cl['note']}\n"
           + "\n" + md_table(["회차", "문서", "출처", "판정", "실제 오류", "판정자 오판"],
                             [[r, x["doc"], x["source"], x["status"], x["verdicts"][E.REAL_ERROR], x["verdicts"][E.JUDGE_ERROR]]
                              for r, x in tally["runs"].items()])
           + "\n결함본 주입 문장 밖 표시: " + ", ".join(f"{k} {v}" for k, v in tally["defect_out_of_place"].items())
           + "\n\n베이스라인 2 지적(섞었을 때): " + ", ".join(f"{k} {v}" for k, v in tally["baseline2"].items()) + "\n")
    md += ("\n## 해시 다시 맞대기\n\n" + md_table(["대상", "결과", "다른 것"], [
        ["00_봉인/SHA256SUMS", None if checks["sealed"] is None else checks["sealed"]["ok"], (checks["sealed"] or {}).get("mismatch")],
        ["사전등록 v3(결함_SHA256SUMS·SHA256SUMS)", "없음" if checks["defects_v3"] is None else checks["defects_v3"]["ok"],
         (checks["defects_v3"] or {}).get("mismatch")]]))
    (folder / "최종보고.md").write_text(md, encoding="utf-8")
    ctx.echo(f"최종 보고 {folder}/최종보고.md·.json — 과잉 차단 {c['과잉 차단']}회, 주장 단위 오판 {cl['wrong_faithfulness_violations']}/{cl['checked']}")
    return out


# ── 명령줄 ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="판정자 평가 실험 실행기(dev / verify)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, helptext: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--mode", choices=MODES, required=True)
        p.add_argument("--root", help="평가 폴더(verify 기본: JUDGE_VALIDATION_DIR, 없으면 저장소 옆 JUDGE_ROUND/평가 · dev: 임시 폴더)")
        return p

    p = add("plant", "결함·변형 표 → 02_결함/<ID>/")
    p.add_argument("--table", required=True)
    add("precheck", "동결 전 확인 → 01_구성/precheck.json")
    p = add("run", "채점 한 회차(33문서, 시드 순서, 동시 실행)")
    p.add_argument("--round", type=int, required=True, choices=(1, 2))
    p.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    p.add_argument("--only", default="", help="이 문서만(쉼표 — F01·V03·pima_strong 꼴)")
    p.add_argument("--live", action="store_true")
    p.add_argument("--yes", action="store_true")
    add("status", "회차별 끝남·실패·남음과 장부 누계")
    add("baseline1", "베이스라인 1(1차 점검만) → 05_결과/baseline1.json")
    p = add("baseline2", "베이스라인 2(문서 전체 판정, 문서마다 1회)")
    p.add_argument("--live", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    p = add("aggregate", "§6 집계 → 05_결과/집계.json·집계.md")
    p.add_argument("--partial", action="store_true", help="덜 끝난 회차가 있어도 있는 것으로(부분 집계 표시)")
    p = add("sheet", "과잉 차단 판정표(블라인드)·열쇠 → 04_판정표/")
    p.add_argument("--seed", type=int)
    p.add_argument("--include-baseline2", action="store_true", help="원본·변형의 베이스라인 2 지적도 섞는다(종류는 '지적')")
    p = add("report", "판정한 표 → §6.4 셈 + 집계 → 05_결과/최종보고.md·.json")
    p.add_argument("--answers", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ctx = make_ctx(args.mode, args.root)
        if args.command == "plant":
            cmd_plant(ctx, Path(args.table))
            return 0
        if args.command == "precheck":
            out = cmd_precheck(ctx)
            return 1 if out["variants_differ"] else 0
        if args.command == "run":
            out = cmd_run(ctx, args.round, concurrency=args.concurrency, only=[x for x in args.only.split(",") if x],
                          live=args.live, yes=args.yes)
            print(json.dumps({k: v for k, v in out.items() if k != "run_ids"}, ensure_ascii=False))
            return 3 if out["stopped"] or out["gates"].get("재시도 대상") else 0
        if args.command == "status":
            cmd_status(ctx)
            return 0
        if args.command == "baseline1":
            cmd_baseline1(ctx)
            return 0
        if args.command == "baseline2":
            out = cmd_baseline2(ctx, live=args.live, yes=args.yes, limit=args.limit, concurrency=args.concurrency)
            print(json.dumps(out, ensure_ascii=False))
            return 3 if out["stopped"] or out["gates"].get("재시도 대상") else 0
        if args.command == "aggregate":
            cmd_aggregate(ctx, partial=args.partial)
            return 0
        if args.command == "sheet":
            cmd_sheet(ctx, seed=args.seed, include_b2=args.include_baseline2)
            return 0
        if args.command == "report":
            out = cmd_report(ctx, Path(args.answers))
            bad = [k for k, v in out["integrity"].items() if v is not None and not v["ok"]]
            return 1 if bad else 0
        raise RunnerError(f"모르는 명령: {args.command}")
    except RunnerError as exc:
        print(f"멈춤: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
