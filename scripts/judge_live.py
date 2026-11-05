"""
judge_live.py — 심판 라운드 실호출 실행기(초안, 심판 라운드 세션 1 J0).

pytest 테스트가 아니다. 실호출(진짜 Claude·OpenAI 호출)은 키가 있는 사람 컴퓨터에서 `--live` 를 줄 때만 한다.
레포 루트에서: python scripts/judge_live.py <하위 명령> --ledger <장부.csv> [--live] [--yes]

지키는 것
- `--live` 가 없으면 진짜 LLM 클라이언트를 만들지 않는다. .env 도 `--live` 일 때만 읽는다(키가 있어도 부르지 않게).
- 한 실행 = 장부(csv) 한 줄. 열 목록은 LEDGER_COLUMNS. 비용은 단가를 아는 모델만 적고 모르면 비운다(추정하지 않는다).
- 묶음을 시작하기 전에 예상 비용·회차가 남은 상한 안인지 본다. 넘으면 시작하지 않는다(반쯤 돈 묶음을 남기지 않게).
  상한은 환경변수 JUDGE_LIVE_BUDGET_USD(달러)·JUDGE_LIVE_MAX_RUNS(회차), 없으면 아래 기본값.
- 시작할 때 첫 줄에 이번 묶음 예상과 지금까지 누계를 찍고, `--yes` 없이는 멈춘다.
- 누계 비용이 $10 단위를 넘을 때마다(단가를 모르는 실행은 20회마다) 멈추고 확인 입력을 받는다.
- dev 모드는 검증 폴더(JUDGE_VALIDATION_DIR)를 읽지 못한다(guard_read).

하위 명령은 단계마다 붙는다.
- status: 장부 누계와 상한
- seal-generate(J2 끝): 검증용 판 4 요청(폴더의 *.request.json)마다 실제 앱 작업 경로(POST·GET /api/narrative/jobs, 진짜 Claude)로
  서술을 한 번 만들고, 요청·작업 결과·내부 기록을 봉인 폴더에 쓴 뒤 SHA256SUMS·봉인 시각을 적고 읽기 전용으로 바꾼다. 서술 글은
  화면에 찍지 않는다(오염 방지 — 개수·출처·비용·해시만). 작업 DB 는 임시 sqlite. 이미 봉인된 폴더면 거부(다시 만들지 않는다).
- pilot(J4 끝): 심판 개발 입력(랜덤 예측, tests/data/judge_dev)으로 채점 실측. --narrative template 이면 키 없는 규칙 문장
  서술(무료)을, ai 면 진짜 Claude 로 서술을 만들어 judge_once 로 채점한다. 검증 폴더는 읽지 않는다(dev 입력만). 결과 전체를
  --out 폴더에 쓴다(개발 데이터라 봉인하지 않는다).
- smoke(J1): 합성 문장으로 ① GPT-6.1 Sol 이 응답 보관을 끈 채 계산 도구를 여러 번 부르는지(암호화 추론을 다음 턴에 되돌림)
  ② GPT-6 Luna 가 Responses 구조화 출력을 받는지 ③ 같은 앞부분을 두 번 보내면 캐시 입력 토큰이 잡히는지. 검증 데이터는 쓰지 않는다.

상호작용
- 의존(import): csv, os, argparse. `--live` 일 때만 dotenv·anthropic·openai, app.narrative.llm(서술 클라이언트 설정)
- 사용처: 사람이 직접 실행. 시험은 tests/test_judge_live_runner.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# 레포 루트를 sys.path 에 추가 (app.* 절대 import 를 위해)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:  # 같은 폴더의 judge_paths(경로 비교 한 곳)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LEDGER_COLUMNS: tuple[str, ...] = (
    "run_id", "at", "stage", "report", "variant", "repeat", "mode",
    "criteria_version", "manifest_sha256", "git_commit",
    "model_requested", "model_served", "effort",
    "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens",
    "cost_usd", "price_table", "seconds", "gate", "result_file", "result_sha256", "error",
)

MODES = ("dev", "live")

# 라운드 상한 기본값(사용자 답 — 2026-10-28 실호출 상한 $4, 2026-10-29 V1 결정 1 로 $4.5·40회, 장부 누계 포함).
# 회차 상한은 J1 스모크·J2 생성·J4 파일럿·J5b 한 작업·V2 10회의 합보다 넉넉히. 미달 재검증은 그때 다시 묻고 환경변수로 올린다.
DEFAULT_BUDGET_USD = 4.5
DEFAULT_MAX_RUNS = 40

# 누계 알림 — 비용은 이 단위를 넘을 때마다, 단가를 모르는 실행은 이 회차마다 멈춘다.
CHECKPOINT_USD = 10.0
CHECKPOINT_UNKNOWN_RUNS = 20


class RunnerError(RuntimeError):
    """실행기가 시작을 거부하거나 멈출 때."""


# ── 장부 ─────────────────────────────────────────────────────────────


class Ledger:
    """장부 csv. 없으면 머리줄만 만들고, 있으면 머리줄이 LEDGER_COLUMNS 와 같은지부터 본다."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(LEDGER_COLUMNS)
        with self.path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if tuple(header or ()) != LEDGER_COLUMNS:
            raise RunnerError(f"장부 머리줄이 실행기 열 목록과 다르다: {self.path}")

    def rows(self) -> list[dict[str, str]]:
        with self.path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def totals(self) -> tuple[float, int, int]:
        """(아는 비용 합, 전체 실행 수, 비용을 모르는 실행 수)."""
        cost, unknown = 0.0, 0
        rows = self.rows()
        for row in rows:
            if row["cost_usd"].strip():
                cost += float(row["cost_usd"])
            else:
                unknown += 1
        return cost, len(rows), unknown

    def next_run_id(self) -> str:
        return f"R-{len(self.rows()) + 1:04d}"

    def append(self, row: dict) -> dict:
        extra = set(row) - set(LEDGER_COLUMNS)
        if extra:
            raise RunnerError(f"장부에 없는 열: {sorted(extra)}")
        full = {col: "" for col in LEDGER_COLUMNS}
        full.update({k: "" if v is None else str(v) for k, v in row.items()})
        if not full["run_id"]:
            full["run_id"] = self.next_run_id()
        if not full["at"]:
            full["at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([full[col] for col in LEDGER_COLUMNS])
        return full


# ── 상한·알림 ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Budget:
    usd: float
    max_runs: int

    @classmethod
    def from_env(cls) -> "Budget":
        usd = os.getenv("JUDGE_LIVE_BUDGET_USD", "").strip()
        runs = os.getenv("JUDGE_LIVE_MAX_RUNS", "").strip()
        return cls(usd=float(usd) if usd else DEFAULT_BUDGET_USD, max_runs=int(runs) if runs else DEFAULT_MAX_RUNS)


@dataclass(frozen=True)
class BatchPlan:
    """시작 전에 알리는 묶음 예상. 단가를 모르면 est_cost_usd 는 None(회차 상한으로만 막는다)."""

    stage: str
    runs: int
    est_cost_usd: float | None


def crossed_checkpoint(before_usd: float, after_usd: float) -> bool:
    """누계가 $10 단위를 새로 넘었나."""
    return math.floor(after_usd / CHECKPOINT_USD) > math.floor(before_usd / CHECKPOINT_USD)


class Runner:
    """묶음 하나를 감싼다: start(예상·상한·확인) → record(한 줄 쓰고 누계 알림)."""

    def __init__(
        self,
        ledger: Ledger,
        budget: Budget,
        mode: str,
        *,
        confirm: Callable[[str], str] = input,
        out: Callable[[str], None] = print,
    ):
        if mode not in MODES:
            raise RunnerError(f"모르는 모드: {mode}")
        self.ledger, self.budget, self.mode = ledger, budget, mode
        self._confirm, self._out = confirm, out
        self._unknown_since_checkpoint = 0

    def start(self, plan: BatchPlan, *, yes: bool) -> None:
        spent, runs, unknown = self.ledger.totals()
        est = "모름(단가 없음)" if plan.est_cost_usd is None else f"${plan.est_cost_usd:.2f}"
        self._out(
            f"[{self.mode}] 묶음 {plan.stage}: 예상 {plan.runs}회·{est} | 누계 ${spent:.2f}·{runs}회"
            f"(비용 모름 {unknown}회) | 상한 ${self.budget.usd:.2f}·{self.budget.max_runs}회"
        )
        if runs + plan.runs > self.budget.max_runs:
            raise RunnerError(f"회차 상한을 넘는다: {runs} + {plan.runs} > {self.budget.max_runs}")
        if plan.est_cost_usd is not None and spent + plan.est_cost_usd > self.budget.usd:
            raise RunnerError(f"비용 상한을 넘는다: ${spent:.2f} + ${plan.est_cost_usd:.2f} > ${self.budget.usd:.2f}")
        if not yes:
            raise RunnerError("확인 인자 --yes 가 없어 시작하지 않는다(위 예상을 보고 다시 실행).")

    def record(self, row: dict) -> dict:
        before, _, _ = self.ledger.totals()
        written = self.ledger.append({**row, "mode": self.mode})
        after = before + (float(written["cost_usd"]) if written["cost_usd"] else 0.0)
        if not written["cost_usd"]:
            self._unknown_since_checkpoint += 1
        self._out(f"  {written['run_id']} {written['stage']} {written['report']} — 누계 ${after:.2f}")
        reason = None
        if crossed_checkpoint(before, after):
            reason = f"누계가 ${math.floor(after / CHECKPOINT_USD) * CHECKPOINT_USD:.0f} 를 넘었다"
        elif self._unknown_since_checkpoint >= CHECKPOINT_UNKNOWN_RUNS:
            reason = f"비용을 모르는 실행이 {CHECKPOINT_UNKNOWN_RUNS}회 쌓였다"
        if reason:
            self._unknown_since_checkpoint = 0
            answer = self._confirm(f"{reason}. 이어 하려면 '계속'을 입력: ").strip()
            if answer != "계속":
                raise RunnerError(f"{reason} — 확인이 없어 멈춘다.")
        return written


# ── 읽기 경계·클라이언트 ─────────────────────────────────────────────────


def validation_dir() -> Path | None:
    value = os.getenv("JUDGE_VALIDATION_DIR", "").strip()
    return Path(value).resolve() if value else None


def guard_read(path: Path, mode: str) -> Path:
    """dev 모드에서 검증 폴더 안을 읽으려 하면 거부한다(오염 방지 규칙 2). dev 인데 검증 폴더를 모르면 그것도 거부."""
    resolved = Path(path).resolve()
    if mode == "dev":
        vdir = validation_dir()
        if vdir is None:
            raise RunnerError("dev 모드는 JUDGE_VALIDATION_DIR 를 알아야 한다(검증 폴더를 읽지 않음을 확인하려고).")
        import judge_paths  # 같은 폴더(scripts) — 철자·대소문자·자모 분해 우회까지 같은 비교

        if judge_paths.inside(resolved, vdir):
            raise RunnerError(f"dev 모드는 검증 폴더를 읽지 않는다: {resolved}")
    return resolved


@dataclass
class Clients:
    anthropic: object | None = None
    openai: object | None = None


def make_clients(live: bool) -> Clients:
    """`--live` 일 때만 .env 를 읽고 진짜 클라이언트를 만든다. 아니면 둘 다 None — 키가 있어도 만들지 않는다."""
    if not live:
        return Clients()
    from dotenv import load_dotenv

    load_dotenv()
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI

    from app.narrative import llm as narrative_llm

    clients = Clients()
    if os.getenv("ANTHROPIC_API_KEY"):
        clients.anthropic = AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            timeout=narrative_llm.CALL_TIMEOUT_SECONDS,
            max_retries=0,
        )
    if os.getenv("OPENAI_API_KEY"):
        clients.openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
    if clients.anthropic is None and clients.openai is None:
        raise RunnerError("--live 인데 ANTHROPIC_API_KEY·OPENAI_API_KEY 가 둘 다 없다.")
    return clients


# ── 명령줄 ───────────────────────────────────────────────────────────


def run_meta() -> dict:
    """장부 줄의 기준 버전·지문·커밋."""
    import subprocess

    from app.judge import criteria

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "", ""
    return {"criteria_version": criteria.version(), "manifest_sha256": criteria.fingerprint(),
            "git_commit": commit + ("+dirty" if dirty else "")}


def ledger_row(outcome, *, stage: str, report: str, variant: str = "", repeat: int | str = "", meta: dict | None = None) -> dict:
    """llm.Outcome 하나 → 장부 한 줄."""
    r = outcome.record
    return {
        **(meta or {}), "stage": stage, "report": report, "variant": variant, "repeat": repeat,
        "model_requested": r.requested_model, "model_served": r.response_model or "", "effort": r.effort,
        "input_tokens": r.input_tokens, "cached_input_tokens": r.cached_input_tokens, "output_tokens": r.output_tokens,
        "reasoning_tokens": r.reasoning_tokens, "cost_usd": "" if r.cost_usd is None else f"{r.cost_usd:.6f}",
        "price_table": "openai-2026-10-28" if r.cost_usd is not None else "", "seconds": r.seconds,
        "error": r.failure or "",
    }


SMOKE_TOOL = {
    "type": "function", "name": "calc", "description": "사칙연산 식(숫자, + - * /, 괄호)의 값을 돌려준다.", "strict": True,
    "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"], "additionalProperties": False},
}
SMOKE_RATIO = {"type": "object", "properties": {"ratio": {"type": "number"}}, "required": ["ratio"], "additionalProperties": False}
SMOKE_CLAIMS = {
    "type": "object", "additionalProperties": False, "required": ["claims"],
    "properties": {"claims": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["quote", "kind"],
        "properties": {"quote": {"type": "string"}, "kind": {"type": "string", "enum": ["number", "name", "comparison", "degree", "inference"]}},
    }}},
}
# 캐시는 앞부분이 1024 토큰 이상일 때 걸린다 — 고정 지시문을 그만큼 길게 만든다(내용은 합성 규칙).
SMOKE_EXTRACT_RULES = "\n".join(
    f"규칙 {i}: 문장에서 숫자·이름·비교·정도 표현·추론 주장을 하나씩 뽑고, 인용 구절은 원문 그대로 옮긴다. 주장이 없으면 빈 목록이다."
    for i in range(1, 61)
)
SMOKE_SENTENCE = "합성 문장: 이 모델의 Recall 은 0.912 로 목표 0.9 를 넘었고, FN 이 FP 보다 많아 경계 근처 표본이 많음을 시사한다."


async def smoke(runner: "Runner", clients: Clients, meta: dict) -> dict:
    from app.judge import config
    from app.judge import llm
    from app.judge.criteria.rules_calc import calculate

    summary: dict = {"tool_loop": {}, "luna": [], "meta": meta}
    # ① 도구 반복(Sol)
    convo: list = [{"role": "user", "content": "FN 38건, FP 11건이다. calc 도구로 FN 이 오류 전체에서 차지하는 비율을 계산해 ratio 로 답하라."}]
    turns, tool_calls, encrypted = 0, 0, 0
    final = None
    for turn in range(1, 5):
        out = await llm.create_response(
            clients.openai, purpose="smoke_tool", model=config.judge_model(), instructions="계산은 반드시 calc 도구로 한다.",
            input=convo, timeout_s=180, tools=[SMOKE_TOOL], text_format=llm.json_schema_format("ratio", SMOKE_RATIO),
        )
        runner.record(ledger_row(out, stage="smoke", report="tool_loop", repeat=turn, meta=meta))
        turns = turn
        if out.failure:
            summary["tool_loop"]["failure"] = out.failure
            break
        items = llm.next_turn_items(out.response)
        encrypted += sum(1 for i in items if i.get("type") == "reasoning" and i.get("encrypted_content"))
        calls = llm.function_calls(out.response)
        if not calls:
            final, _ = llm.parse_json(out)
            break
        tool_calls += len(calls)
        convo = convo + items + [
            {"type": "function_call_output", "call_id": c.call_id, "output": json.dumps({"value": calculate(json.loads(c.arguments)["expr"])})}
            for c in calls
        ]
    summary["tool_loop"].update({"turns": turns, "tool_calls": tool_calls, "encrypted_reasoning_items": encrypted,
                                 "final": final, "expected_ratio": round(38 / 49, 4)})
    # ②③ 구조화 출력(Luna) 두 번 — 같은 지시문이면 두 번째에 캐시 입력이 잡힌다
    for repeat in (1, 2):
        out = await llm.create_response(
            clients.openai, purpose="smoke_extract", model=config.extract_model(), instructions=SMOKE_EXTRACT_RULES,
            input=[{"role": "user", "content": SMOKE_SENTENCE}], timeout_s=120,
            text_format=llm.json_schema_format("claims", SMOKE_CLAIMS),
        )
        runner.record(ledger_row(out, stage="smoke", report="luna_extract", repeat=repeat, meta=meta))
        data, failure = llm.parse_json(out)
        summary["luna"].append({"failure": failure, "claims": data, "cached_input_tokens": out.record.cached_input_tokens,
                                "input_tokens": out.record.input_tokens, "response_model": out.record.response_model})
    return summary


def seal_generate(runner: "Runner", req_dir: Path, out_dir: Path, meta: dict, *, anthropic_client=None) -> list[dict]:
    """봉인 생성. anthropic_client 를 주면(시험) 앱 시작 뒤 그것으로 바꿔 끼운다. 돌려주는 것은 건마다 요약(글 없음)."""
    import hashlib
    import os as _os
    import stat
    import tempfile
    from datetime import datetime, timezone

    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.database import Base, configure_sqlite
    from app.main import app
    from app.narrative import jobs
    from app.narrative.models import NarrativeJob

    if (out_dir / "SHA256SUMS").exists():
        raise RunnerError(f"이미 봉인된 폴더다(다시 만들지 않는다): {out_dir}")
    requests = sorted(req_dir.glob("*.request.json"))
    if not requests:
        raise RunnerError(f"요청 파일이 없다: {req_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="judge_seal_")
    engine = create_engine(f"sqlite:///{tmp}/jobs.db", connect_args={"check_same_thread": False})
    configure_sqlite(engine)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    app.dependency_overrides[jobs.get_session_factory] = lambda: factory
    summaries: list[dict] = []
    # 봉인은 생성만 한다 — 채점은 V2 가 봉인본을 따로 돈다(비용·기준 고정 순서). J5b 부터 판 4 작업 경로가 생성 뒤 심판을
    # 부르고, 심판 키가 없으면 생성 AI 도 부르지 않으므로(정3) 봉인 동안은 심판 자리를 '돌지 않음'으로 바꿔 끼운다
    # (작업 결과의 judge 는 보류 judge_error — 봉인본은 발급하지 않는다).
    from app.judge.core import JudgeRun

    async def _no_judge(*_a, **_k):
        return JudgeRun(None, "not_run_in_seal")

    real_judge = jobs.judge_once
    jobs.judge_once = _no_judge
    try:
        with TestClient(app) as client:
            if anthropic_client is not None:
                app.state.anthropic_client = anthropic_client
            if app.state.anthropic_client is None:
                raise RunnerError("서술 클라이언트가 없다(ANTHROPIC_API_KEY) — 봉인 서술을 만들 수 없다")
            if app.state.openai_client is None:
                app.state.openai_client = object()  # 심판 자리 표시 — 위 _no_judge 가 받아 부르지 않는다
            for path in requests:
                tag = path.name[: -len(".request.json")]
                raw = json.loads(path.read_text(encoding="utf-8"))
                created = client.post("/api/narrative/jobs", json=raw)
                if created.status_code != 200 or not created.json().get("job_id"):
                    raise RunnerError(f"{tag}: 작업을 만들지 못했다({created.status_code})")
                job_id = created.json()["job_id"]
                view = client.get(f"/api/narrative/jobs/{job_id}").json()
                with factory() as db:
                    internal = json.loads(db.get(NarrativeJob, job_id).internal_json or "{}")
                record = {"tag": tag, "request": raw, "created": created.json(), "view": view, "internal": internal}
                target = out_dir / f"{tag}.json"
                body = json.dumps(record, ensure_ascii=False, indent=1) + "\n"
                target.write_text(body, encoding="utf-8")
                sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
                calls = [c for c in internal.get("calls", []) if c.get("http")]
                models = sorted({c.get("response_model") for c in calls if c.get("response_model")})
                cost = internal.get("cost_usd")
                runner.record({
                    **meta, "stage": "seal_generate", "report": tag, "variant": "base",
                    "model_requested": "claude-sonnet-5-5", "model_served": ",".join(models), "effort": internal.get("effort", ""),
                    "input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
                    "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
                    "cost_usd": "" if cost is None else f"{cost:.6f}", "price_table": "" if cost is None else "anthropic-llm.py",
                    "seconds": internal.get("seconds_total", ""), "result_file": f"00_봉인/{target.name}", "result_sha256": sha,
                    "error": (view.get("failure") or "") if view.get("status") != "done" else "",
                })
                sources = (view.get("result") or {}).get("section_sources") or {}
                summaries.append({"tag": tag, "status": view.get("status"), "sources": sources, "cost_usd": cost,
                                  "http_calls": len(calls), "sha256": sha})
    finally:
        jobs.judge_once = real_judge
        app.dependency_overrides.pop(jobs.get_session_factory, None)
        engine.dispose()
    files = sorted(out_dir.glob("*.json"))
    sums = "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in files)
    (out_dir / "SHA256SUMS").write_text(sums, encoding="utf-8")
    (out_dir / "봉인시각.txt").write_text(
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + f"  git {meta.get('git_commit', '')}\n", encoding="utf-8")
    for f in [*files, out_dir / "SHA256SUMS", out_dir / "봉인시각.txt"]:
        _os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return summaries


def _cmd_seal(args: argparse.Namespace) -> int:
    runner = Runner(Ledger(Path(args.ledger)), Budget.from_env(), "live" if args.live else "dev")
    n = len(list(Path(args.requests).glob("*.request.json")))
    runner.start(BatchPlan("seal_generate", runs=n, est_cost_usd=round(0.22 * n, 2)), yes=args.yes)
    if not args.live:
        raise RunnerError("seal-generate 는 실호출이다 — --live 를 준다")
    make_clients(live=True)  # .env 를 읽는다(앱 시작이 같은 키로 클라이언트를 만든다)
    summaries = seal_generate(runner, Path(args.requests), Path(args.out), run_meta())
    print(json.dumps(summaries, ensure_ascii=False, indent=1))  # 글은 없다 — 상태·출처·비용·해시만
    return 0


async def pilot(runner: "Runner", clients: Clients, paths: list[Path], out_dir: Path, narrative: str, meta: dict) -> list[dict]:
    from app.judge.core import judge_once
    from app.narrative.baselines import build_benchmark_refs
    from app.narrative.generator import generate_sections
    from app.narrative.schemas import NarrativeRequest

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in paths:
        tag = path.name[: -len(".request.json")]
        req = NarrativeRequest.model_validate(json.loads(path.read_text(encoding="utf-8")))
        gen = await generate_sections(clients.anthropic if narrative == "ai" else None, req)
        if narrative == "ai":
            calls = [c for c in gen.trace.get("calls", []) if c.get("http")]
            cost = sum(c.get("cost_usd") or 0 for c in calls)
            runner.record({**meta, "stage": "pilot_generate", "report": tag, "variant": narrative,
                           "model_requested": "claude-sonnet-5-5",
                           "model_served": ",".join(sorted({c.get("response_model") for c in calls if c.get("response_model")})),
                           "input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
                           "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
                           "cost_usd": f"{cost:.6f}", "price_table": "anthropic-llm.py", "seconds": gen.trace.get("seconds", "")})
        refs = build_benchmark_refs(req.task_type.value, req.fact_sheet.metrics, edition=4)
        run = await judge_once(clients.openai, req, gen.response, gen.section_sources, refs)
        calls = judge_calls(run)
        cost = run.cost_usd()
        body = json.dumps({"tag": tag, "narrative": narrative, "sources": gen.section_sources,
                           "result": gen.response.model_dump(), "gate": run.gate, "failure": run.failure,
                           "flagged_slots": run.flagged_slots, "issuance": run.issuance, "blocking_slots": run.blocking_slots,
                           "repair_slots": run.repair_slots, "scorecard": run.scorecard, "record": run.record},
                          ensure_ascii=False, indent=1) + "\n"
        target = out_dir / f"{tag}.{narrative}.json"
        target.write_text(body, encoding="utf-8")
        runner.record({**meta, **judge_run_row(run), "stage": "pilot_judge", "report": tag, "variant": narrative,
                       "result_file": str(target)})
        summaries.append({"tag": tag, "gate": run.gate, "failure": run.failure, "flagged": run.flagged_slots,
                          "cost_usd": cost, "seconds": run.record.get("seconds"), "calls": len(calls),
                          "counts": run.scorecard and {k: {kk: vv for kk, vv in v.items() if kk != "items"}
                                                       for k, v in run.scorecard.items()}})
    return summaries


JUDGE_MODULES = ("extract", "faithfulness", "consistency", "coverage", "readability")


def judge_calls(run) -> list[dict]:
    """채점 한 번(JudgeRun)의 모든 LLM 호출 기록."""
    return [c for m in JUDGE_MODULES for c in (run.record.get(m) or {}).get("calls", [])]


def judge_run_row(run) -> dict:
    """채점 한 번 → 장부 칸(모델·토큰·비용·시간·게이트·실패). 파일럿·검증 실행기가 같이 쓴다."""
    calls = judge_calls(run)
    cost = run.cost_usd()
    return {"model_requested": "gpt-6.1-sol+gpt-6-luna",
            "model_served": ",".join(sorted({c.get("response_model") for c in calls if c.get("response_model")})),
            "effort": run.record.get("effort", ""),
            "input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
            "cached_input_tokens": sum(c.get("cached_input_tokens") or 0 for c in calls),
            "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
            "reasoning_tokens": sum(c.get("reasoning_tokens") or 0 for c in calls),
            "cost_usd": "" if cost is None else f"{cost:.6f}", "price_table": "openai-2026-10-28",
            "seconds": (run.record.get("seconds") or {}).get("total", ""), "gate": run.gate or "",
            "error": run.failure or ""}


def job_live(runner: "Runner", req_path: Path, out_dir: Path, meta: dict, *, anthropic_client=None,
             openai_client=None) -> dict:
    """J5b 끝 한 작업 — 판 4 요청 하나를 실제 작업 경로(POST → 생성 → 심판 사다리 → 묶음)로 돌린다(계획 §마 J5b 행).

    앱 시작이 .env 의 키로 두 클라이언트를 만든다(--live). 시험은 가짜 클라이언트를 넘긴다. 임시 sqlite 만 쓴다.
    장부에는 생성(Anthropic)과 채점(OpenAI, 사다리 전체)을 따로 한 줄씩 쓴다. 돌려주는 요약에는 서술 글이 없다."""
    import hashlib
    import tempfile

    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.database import Base, configure_sqlite
    from app.main import app
    from app.narrative import jobs
    from app.narrative.models import NarrativeJob

    raw = json.loads(Path(req_path).read_text(encoding="utf-8"))
    if raw.get("report_edition") != 4:
        raise RunnerError("job 은 판 4 요청만 받는다(report_edition 4)")
    tag = Path(req_path).name[: -len(".request.json")] if Path(req_path).name.endswith(".request.json") else Path(req_path).stem
    tmp = tempfile.mkdtemp(prefix="judge_job_")
    engine = create_engine(f"sqlite:///{tmp}/jobs.db", connect_args={"check_same_thread": False})
    configure_sqlite(engine)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    app.dependency_overrides[jobs.get_session_factory] = lambda: factory
    try:
        with TestClient(app) as client:
            if anthropic_client is not None:
                app.state.anthropic_client = anthropic_client
            if openai_client is not None:
                app.state.openai_client = openai_client
            if app.state.anthropic_client is None or app.state.openai_client is None:
                raise RunnerError("서술(ANTHROPIC)·심판(OPENAI) 클라이언트가 둘 다 있어야 한다")
            created = client.post("/api/narrative/jobs", json=raw)
            if created.status_code != 200 or not created.json().get("job_id"):
                raise RunnerError(f"{tag}: 작업을 만들지 못했다({created.status_code})")
            job_id = created.json()["job_id"]
            view = client.get(f"/api/narrative/jobs/{job_id}").json()  # TestClient 는 백그라운드 작업을 응답 전에 끝낸다
            with factory() as db:
                internal = json.loads(db.get(NarrativeJob, job_id).internal_json or "{}")
    finally:
        app.dependency_overrides.pop(jobs.get_session_factory, None)
        engine.dispose()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{tag}.job.json"
    body = json.dumps({"tag": tag, "view": view, "internal": internal}, ensure_ascii=False, indent=1) + "\n"
    target.write_text(body, encoding="utf-8")
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()

    gen_calls = [c for c in internal.get("calls", []) if c.get("http")]
    gen_cost = internal.get("cost_usd")
    judge = internal.get("judge") or {}
    summary = judge.get("summary") or {}
    jcalls = [c for r in judge.get("rounds", []) for m in ("extract", "faithfulness", "consistency", "coverage", "readability")
              for c in ((r.get("record") or {}).get(m) or {}).get("calls", [])]
    common = {**meta, "report": tag, "variant": "job", "result_file": str(target), "result_sha256": sha}
    runner.record({**common, "stage": "job_generate", "model_requested": "claude-sonnet-5-5",
                   "model_served": ",".join(sorted({c.get("response_model") for c in gen_calls if c.get("response_model")})),
                   "input_tokens": sum(c.get("input_tokens") or 0 for c in gen_calls),
                   "output_tokens": sum(c.get("output_tokens") or 0 for c in gen_calls),
                   "cost_usd": "" if gen_cost is None else f"{gen_cost:.6f}", "price_table": "anthropic-llm.py",
                   "seconds": internal.get("seconds", ""), "error": internal.get("failure") or internal.get("reason") or ""})
    jcost = summary.get("cost_usd")
    runner.record({**common, "stage": "job_judge", "model_requested": "gpt-6.1-sol+gpt-6-luna",
                   "model_served": ",".join(sorted({c.get("response_model") for c in jcalls if c.get("response_model")})),
                   "input_tokens": sum(c.get("input_tokens") or 0 for c in jcalls),
                   "cached_input_tokens": sum(c.get("cached_input_tokens") or 0 for c in jcalls),
                   "output_tokens": sum(c.get("output_tokens") or 0 for c in jcalls),
                   "reasoning_tokens": sum(c.get("reasoning_tokens") or 0 for c in jcalls),
                   "cost_usd": "" if jcost is None else f"{jcost:.6f}", "price_table": "openai-2026-10-28",
                   "seconds": summary.get("seconds", ""), "gate": judge.get("gate", ""),
                   "error": judge.get("held_kind") or ""})
    return {"tag": tag, "status": view.get("status"), "judge": (view.get("result") or {}).get("judge"),
            "sources": (view.get("result") or {}).get("section_sources"), "summary": summary,
            "generation_cost_usd": gen_cost, "generation_reason": internal.get("reason"),
            "seconds_total": internal.get("seconds_total"), "stages": internal.get("stages"),
            "rounds": [{k: r.get(k) for k in ("round", "after", "gate", "failure", "flagged", "seconds", "cost_usd", "skipped")}
                       for r in judge.get("rounds", [])], "sha256": sha}


def _cmd_job(args: argparse.Namespace) -> int:
    path = Path(args.request)
    guard_read(path, "dev")  # 검증 폴더(봉인)는 읽지 않는다 — 개발 입력만
    runner = Runner(Ledger(Path(args.ledger)), Budget.from_env(), "live" if args.live else "dev")
    # 최악: 생성 $0.25 + 채점 3번(시간표상 10분 안 최대) × $0.15 = $0.70. 보통(한 번에 통과) 약 $0.40.
    runner.start(BatchPlan("job", runs=2, est_cost_usd=0.70), yes=args.yes)
    if not args.live:
        raise RunnerError("job 은 실호출이다 — --live 를 준다")
    make_clients(live=True)  # .env 를 읽는다(앱 시작이 같은 키로 두 클라이언트를 만든다)
    summary = job_live(runner, path, Path(args.out), run_meta())
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def _cmd_pilot(args: argparse.Namespace) -> int:
    import asyncio

    paths = [Path(p) for p in args.request]
    for p in paths:
        guard_read(p, "dev")  # 검증 폴더(봉인)는 읽지 않는다
    runner = Runner(Ledger(Path(args.ledger)), Budget.from_env(), "live" if args.live else "dev")
    per = 0.25 + (0.2 if args.narrative == "ai" else 0.0)
    runner.start(BatchPlan(f"pilot_{args.narrative}", runs=len(paths) * (2 if args.narrative == "ai" else 1),
                           est_cost_usd=round(per * len(paths), 2)), yes=args.yes)
    if not args.live:
        raise RunnerError("pilot 은 실호출이다 — --live 를 준다")
    clients = make_clients(live=True)
    if clients.openai is None or (args.narrative == "ai" and clients.anthropic is None):
        raise RunnerError("키가 없다")
    summaries = asyncio.run(pilot(runner, clients, paths, Path(args.out), args.narrative, run_meta()))
    print(json.dumps(summaries, ensure_ascii=False, indent=1))
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    import asyncio

    runner = Runner(Ledger(Path(args.ledger)), Budget.from_env(), "live" if args.live else "dev")
    runner.start(BatchPlan("smoke", runs=6, est_cost_usd=0.3), yes=args.yes)
    if not args.live:
        raise RunnerError("smoke 는 실호출이다 — --live 를 준다")
    clients = make_clients(live=True)
    summary = asyncio.run(smoke(runner, clients, run_meta()))
    text = json.dumps(summary, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    ledger = Ledger(Path(args.ledger))
    spent, runs, unknown = ledger.totals()
    budget = Budget.from_env()
    print(f"누계 ${spent:.2f}·{runs}회(비용 모름 {unknown}회) | 상한 ${budget.usd:.2f}·{budget.max_runs}회")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="심판 라운드 실호출 실행기")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="장부 누계와 상한")
    status.add_argument("--ledger", required=True)
    status.set_defaults(func=_cmd_status)
    seal_p = sub.add_parser("seal-generate", help="J2 끝 검증용 판 4 서술 생성·봉인(실호출, 건마다 1번)")
    seal_p.add_argument("--ledger", required=True)
    seal_p.add_argument("--requests", required=True)
    seal_p.add_argument("--out", required=True)
    seal_p.add_argument("--live", action="store_true")
    seal_p.add_argument("--yes", action="store_true")
    seal_p.set_defaults(func=_cmd_seal)
    pilot_p = sub.add_parser("pilot", help="J4 파일럿(개발 입력으로 채점 실측)")
    pilot_p.add_argument("--ledger", required=True)
    pilot_p.add_argument("--request", required=True, action="append")
    pilot_p.add_argument("--out", required=True)
    pilot_p.add_argument("--narrative", choices=["template", "ai"], default="template")
    pilot_p.add_argument("--live", action="store_true")
    pilot_p.add_argument("--yes", action="store_true")
    pilot_p.set_defaults(func=_cmd_pilot)
    job_p = sub.add_parser("job", help="J5b 끝 한 작업(개발 판 4 입력 하나 — 생성 + 심판 사다리, 실호출)")
    job_p.add_argument("--ledger", required=True)
    job_p.add_argument("--request", required=True)
    job_p.add_argument("--out", required=True)
    job_p.add_argument("--live", action="store_true")
    job_p.add_argument("--yes", action="store_true")
    job_p.set_defaults(func=_cmd_job)
    smoke_p = sub.add_parser("smoke", help="J1 실호출 스모크(합성 문장, 약 6회)")
    smoke_p.add_argument("--ledger", required=True)
    smoke_p.add_argument("--out")
    smoke_p.add_argument("--live", action="store_true")
    smoke_p.add_argument("--yes", action="store_true")
    smoke_p.set_defaults(func=_cmd_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RunnerError as exc:
        print(f"멈춤: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
