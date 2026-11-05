"""
judge_eval.py — 판정자 평가 실험 실행기(사전 등록 v1 — 평가/사전등록/v1/사전등록_v1.md). pytest 시험이 아니다.

두 모드(judge_validate 와 같은 뜻)
- dev    : 가짜 LLM(scripts/dev_fake_llm_server 의 가짜 OpenAI)과 개발 입력(tests/data/judge_dev — 규칙 문장 서술)만. 평가 폴더는
           임시 폴더(--root, 없으면 새로 만든다)다. 검증 폴더·평가 폴더(저장소 옆 JUDGE_ROUND/평가·JUDGE_VALIDATION_DIR) 안은 어떤
           철자로 적어도 거부한다(judge_validate.guard + 평가 폴더).
- verify : 진짜 평가. 평가 폴더 = --root, 없으면 환경변수 JUDGE_VALIDATION_DIR, 없으면 저장소 옆 JUDGE_ROUND/평가. 바탕 서술은
           00_봉인/<태그>.json(SHA256SUMS 와 맞아야). 실호출 명령(run·baseline2)은 judge_validate.verify_problems 다섯 조건(깨끗한
           작업 트리·고정 기준·태그 MANIFEST·--live·장부 쓰기) + 사전 등록 지문(980e9bcc…)·모델 설정(§0)이 모두 맞아야 시작한다.

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
                           장부 누계 포함, 회당 예상 $0.30). 시작 전 넘으면 거부, 돌다가 다음 회차가 넘으면 새 회차를 띄우지 않는다.
- baseline1              : 1차 점검만(코드, 비용 0) — 33문서 → 05_결과/baseline1.json(칸·문장 위치, 결함본 탐지).
- baseline2              : 문서 전체 판정(사전등록/v1/베이스라인2_프롬프트.md 의 지시문·입력 틀·스키마 — SHA256SUMS 와 맞아야). 문서마다
                           1회, 03_실행/B2-<문서>.json, 장부 단계 baseline2. --limit K·--concurrency N.
- aggregate [--partial]  : §6 집계 전부 → 05_결과/집계.json·집계.md. 회차가 덜 끝났으면 --partial 일 때만(부분 집계 표시).
- sheet [--seed S]       : 과잉 차단 판정표(§6.4·판정표_양식.md) — 04_판정표/판정표.csv(첫 줄 '# 열쇠 sha256: …'), 열쇠는 따로(읽기 전용).
                           쓰기 전에 새는 것 검사(evaluation.sheet_leaks). 판정표가 이미 있으면 거부.
- report --answers <csv> : 판정한 표(열쇠 해시 맞대기) → §6.4 셈 + 집계 → 05_결과/최종보고.md·.json. 봉인(00_봉인/SHA256SUMS)과
                           결함(사전등록/v3/SHA256SUMS 가 있으면) 해시도 다시 맞대 본다.
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
# §7.4 다시 돌리는 실패 — API 실패·시간 초과(호출 경계 llm.FAILURES 의 시간·전송 갈래)와 실행기 예외. 판정자 실패(거절·잘림·인용 없음·
# 출력 틀림·안전장치 등)는 결과를 이유로 다시 돌리지 않는다(최종 실패 — 집계에서 '실패').
RETRYABLE = frozenset({"timeout", "transient", "api_error", "error", "deadline", "no_client"})
EXCEPTION_PREFIX = "exception:"


def retryable(error: str) -> bool:
    error = (error or "").strip()
    return error in RETRYABLE or error.startswith(EXCEPTION_PREFIX)


# ── 평가 폴더·경계 ────────────────────────────────────────────────────────


def eval_root(root: Path | str | None) -> Path:
    """verify 평가 폴더 — --root, 없으면 JUDGE_VALIDATION_DIR, 없으면 저장소 옆 JUDGE_ROUND/평가."""
    return Path(root).resolve() if root else (jl.validation_dir() or DEFAULT_EVAL)


def dev_guard(path: Path | str) -> Path:
    """dev 경계 — judge_validate.guard(검증 폴더 거부·허용 목록) + 평가 폴더 거부."""
    resolved = jv.guard(path, "dev")
    if judge_paths.inside(resolved, DEFAULT_EVAL):
        raise RunnerError(f"dev 모드는 평가 폴더를 읽거나 쓰지 않는다: {resolved}")
    return resolved


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
        if Counter(E.numeric_tokens(base_text, exclude=False)) != Counter(E.numeric_tokens(new_text, exclude=False)):
            problems.append("일관성 결함(C01 제외)의 변경문은 숫자를 바꾸지 않는다(§3.4)")
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
    folder = ctx.path(DEFECTS)
    out["files"] = {p.relative_to(ctx.root).as_posix(): sha256_file(p)
                    for p in sorted(folder.rglob("*")) if p.is_file()}  # 사전 등록 v3 SHA256SUMS 에 옮겨 적을 값
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
    """이 실험 장부 상한(§7.7 $30·110회) — JUDGE_LIVE_BUDGET_USD·JUDGE_LIVE_MAX_RUNS 로 바꾼다."""

    usd: float
    runs: int

    @classmethod
    def from_env(cls) -> "Cap":
        usd = os.getenv("JUDGE_LIVE_BUDGET_USD", "").strip()
        runs = os.getenv("JUDGE_LIVE_MAX_RUNS", "").strip()
        return cls(float(usd) if usd else E.BUDGET_USD, int(runs) if runs else E.BUDGET_RUNS)


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
    """verify 채점 전 — precheck 가 변형 다름 없이 끝났고 그 뒤 02_결함 이 바뀌지 않았으며, 사전등록/v3/SHA256SUMS 가 있으면 맞다."""
    out = []
    pre = ctx.root / BUILD / "precheck.json"
    files = {p.relative_to(ctx.root).as_posix(): sha256_file(p) for p in sorted((ctx.root / DEFECTS).rglob("*")) if p.is_file()} \
        if (ctx.root / DEFECTS).is_dir() else {}
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


def defect_sum_problems(ctx: Ctx) -> list[str]:
    """사전등록/v3/SHA256SUMS(있으면)의 줄 가운데 평가 폴더(또는 02_결함) 안 파일을 가리키는 것이 모두 맞는가."""
    sums_path = ctx.root / PREREG_V3 / "SHA256SUMS"
    if not sums_path.exists():
        return []
    out, checked = [], 0
    for name, digest in read_sums(sums_path).items():
        cands = [ctx.root / name, ctx.root / DEFECTS / name, ctx.root / PREREG_V3 / name]
        path = next((p for p in cands if p.is_file()), None)
        if path is None:
            if name.startswith(f"{DEFECTS}/"):
                out.append(f"사전등록 v3 에 적힌 파일이 없다: {name}")
            continue
        checked += 1
        if sha256_file(path) != digest:
            out.append(f"사전등록 v3 해시와 다르다: {name}")
    return out


async def run_round(ctx: Ctx, round_no: int, order: list[Doc], todo: list[Doc], bases: dict[str, jv.Base], *, client,
                    judge, concurrency: int, cap: Cap, meta: dict) -> dict:
    """남은 회차를 동시에 concurrency 개씩. 회차마다 결과 원본 03_실행/<E-…>.json + 장부 한 줄(잠금 안에서)."""
    stage = STAGE_OF_ROUND[round_no]
    seed = E.ROUND_SEEDS[round_no]
    writer = LedgerWriter(ctx.ledger_path)
    sem = asyncio.Semaphore(concurrency)
    state = {"in_flight": 0}
    position = {d.id: i for i, d in enumerate(order)}

    async def one(doc: Doc) -> dict:
        started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        t0 = time.monotonic()
        run, sha, error = None, narrative_sha(bases[doc.report], doc.slots), ""
        try:
            run, sha = await judge(client, bases[doc.report], doc.slots)
            error = run.failure or ""
        except Exception as exc:  # noqa: BLE001 — 회차 하나의 예외로 나머지를 멈추지 않는다(재시도 대상으로 남긴다)
            error = f"{EXCEPTION_PREFIX}{type(exc).__name__}: {str(exc)[:200]}"
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
            ctx.echo(f"상한에 닿아 새 회차를 띄우지 않는다({problem}) — 남은 {len(stopped)}회")
            break
        state["in_flight"] += 1
        tasks.append(asyncio.create_task(guarded(doc)))
    rows = await asyncio.gather(*tasks)
    for r in rows:
        ctx.echo(f"  {r['run_id']} {r['stage']} {E.doc_id(r['report'], r['variant'])} — {r['gate'] or '실패 ' + r['error']}")
    return {"rows": rows, "stopped": stopped}


def cmd_run(ctx: Ctx, round_no: int, *, concurrency: int = MAX_CONCURRENCY, only: list[str] | None = None, live: bool = False,
            yes: bool = False, client=None, judge=None, git=jv._git) -> dict:
    """채점 한 회차(§7.2). 확인 입력이 없다 — --yes 가 없으면 계획만 찍고 멈춘다."""
    if round_no not in E.ROUND_SEEDS:
        raise RunnerError(f"회차는 1 또는 2: {round_no}")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise RunnerError(f"동시 실행은 1~{MAX_CONCURRENCY}개(§7.2): {concurrency}")
    if ctx.mode == "verify":
        problems = eval_verify_problems(live=live, ledger_path=ctx.ledger_path, git=git)
        if problems:
            raise RunnerError("verify 를 시작하지 않는다:\n  - " + "\n  - ".join(problems))
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
    cap = Cap.from_env()
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
    cap = Cap.from_env()
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
    sem = asyncio.Semaphore(concurrency)
    state = {"in_flight": 0}
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
            ctx.echo(f"상한에 닿아 새 호출을 띄우지 않는다({problem}) — 남은 {len(stopped)}회")
            break
        state["in_flight"] += 1
        tasks.append(asyncio.create_task(guarded(doc)))
    rows = await asyncio.gather(*tasks)
    for r in rows:
        ctx.echo(f"  {r['run_id']} baseline2 {E.doc_id(r['report'], r['variant'])} — {r['gate'] or '실패 ' + r['error']}")
    return {"rows": rows, "stopped": stopped}


def cmd_baseline2(ctx: Ctx, *, live: bool = False, yes: bool = False, limit: int | None = None,
                  concurrency: int = MAX_CONCURRENCY, client=None, git=jv._git) -> dict:
    """문서마다 1회(§5). 이미 끝난 문서(B2-<문서>.json)는 건너뛰고 API 실패만 다시 부른다. 순서는 1회차 시드 순서."""
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise RunnerError(f"동시 실행은 1~{MAX_CONCURRENCY}개(§7.2): {concurrency}")
    prompt = load_b2_prompt(ctx)
    if ctx.mode == "verify":
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
    cap = Cap.from_env()
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
        raise RunnerError(f"모르는 명령: {args.command}")
    except RunnerError as exc:
        print(f"멈춤: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
