"""app/narrative/jobs.py — 서술 작업 번호(job) 방식 (서술 라운드 S3a, 계획 설계 요지 7·10).

옛 경로(`POST /api/generate-narrative`)는 요청 하나로 서술을 받는다 — 서버 마감 120초, 프론트는 160초에 끊는다.
품질 우선으로 서술이 몇 분 걸려도 되게 하려면 요청과 생성을 떼어야 한다. 그래서 새 경로 둘을 둔다.

- `POST /api/narrative/jobs` : 사실 시트를 받아 작업 번호와 8.1 벤치마크 표를 **즉시** 돌려준다. 서술은
  BackgroundTasks 로 `run_job` 이 만든다. 8.1 표가 여기서 오므로 서술이 실패해도 표는 남는다(점검 유-2).
- `GET /api/narrative/jobs/{id}` : 단계(대기·생성 중·검사 중·다시 만드는 중·완료·실패)와 결과.

옛 경로는 그대로 둔다 — 운영 중인 옛 프론트가 쓰고, 작업 번호를 옛 주소에 넣으면 옛 프론트가 빈 서술을
완성본으로 굳힌다(점검 유-7).

## 끝나는 방식

- **완료(done)**: 서술이 생겼다. AI 가 썼든(llm) 규칙 문장으로 대체됐든(fallback) 완성본이다 — 칸마다 출처를
  싣는다. 키 없음·지갑 소진·AI 호출 실패·숫자 검문 위반·입력 상한 초과는 모두 규칙 문장으로 **완료**한다.
  입력 상한(사실 시트 칸별 상한) 초과는 422 로 돌려보내지 않는다 — 다시 만들어도 같은 입력이라 풀리지 않는
  막다른 길이 되기 때문이다(점검 유-11). 상한 안으로 줄인 뒤 규칙 문장으로 끝낸다(사유 input_too_large).
- **실패(failed)**: 서술이 없다. 전체 10분 제한을 넘김(timeout), 서버 재시작 등으로 멈춤(lost — 조회할 때
  판정), 작업 중 예기치 않은 오류(error). 프론트는 "서술 다시 만들기"를 준다.

## 기록 둘 (계획 원칙 6)

- 고객용(result_json → 조회 응답): 8.1 표, 서술 본문, 칸별 출처. 번호만 알면 열리므로 이것만 싣는다.
- 내부용(internal_json): 실패 종류·강등 사유·호출 기록(응답 모델·거절 종류·토큰·비용)·단계별 시각·슬롯 대기.
  입력(사실 시트·고객 입력)은 서버에 남기지 않는다 — 작업은 메모리의 요청으로 돌고, 재시작되면 유실이 된다.

## 남용 방어 (계획 10)

- 서술 지갑: 작업을 만들 때 보고서 1건을 예약한다(AI 를 부를 작업만). 소진이면 규칙 문장으로 완료.
- 작업 행 지갑(narrative_jobs): 시간당 작업 행 수 — AI 비용이 아니라 DB 행 수를 막는다. 넘으면 행 없이 규칙 문장
  결과를 작업 생성 응답에 바로 담는다(429 로 막지 않는다).
- 호출: 작업 경로의 클라이언트는 재시도 0(`max_retries=0`)·호출 시간 제한 JOB_CALL_TIMEOUT_SECONDS 로 부른다. 생성은
  장별 호출 + 칸별 검문 사다리다(app.narrative.generator, S5) — 한 칸의 HTTP 호출은 최대 2회, 작업 하나는 많아야
  장 3회 + 칸 다시 만들기 6회다. 작업 전체 제한 안에서 호출마다 남은 시간으로 제한을 줄인다.
- 서술 슬롯 대기 상한: 넘으면 규칙 문장(busy).

상호작용
- 의존(import): app.narrative.generator(generate_sections), app.narrative.baselines, app.narrative.models,
  app.narrative.schemas, app.core.database(SessionLocal), app.core.llm_budget, app.core.concurrency
- 사용처: app.narrative.router, tests
"""
import asyncio
import copy
import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timedelta

from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from sqlalchemy import delete, update

from app.core import database, llm_budget
from app.core.concurrency import run_cpu_bound
from app.narrative import sections
from app.narrative.baselines import build_benchmark_refs
from app.narrative.models import NarrativeJob, _utcnow
from app.narrative.schemas import (
    BenchmarkRefOut,
    LimitExceeded,
    NarrativeJobCreated,
    NarrativeJobOut,
    NarrativeJobResult,
    NarrativeRequest,
    NarrativeResponse,
)
from app.narrative.generator import GenerationResult, generate_sections

logger = logging.getLogger(__name__)

# ── 시간 ──────────────────────────────────────────────────────────────────────
# 전체 제한 — 슬롯 대기·생성·검사를 합쳐 이 시간을 넘으면 실패(timeout). 사용자 결정 "품질 우선(5~10분 허용)".
JOB_DEADLINE_SECONDS = 600.0
# 작업 경로 AI 호출 1회의 시간 제한(재시도 0). 넘으면 그 호출은 실패로 끝나고 규칙 문장(사유 timeout)으로 완료한다.
JOB_CALL_TIMEOUT_SECONDS = 300.0
# 서술 슬롯 대기 상한. 넘으면 그 호출의 칸은 규칙 문장(사유 busy).
JOB_SLOT_WAIT_SECONDS = 120.0
# 생성기에 주는 시간 — 전체 제한에서 끝 상태를 DB 에 쓸 여유를 뺀 만큼. 남은 시간이 모자라면 생성기가 호출하지 않고
# 그 칸을 규칙 문장으로 끝낸다(작업 전체가 timeout 실패로 끝나지 않게).
JOB_GENERATION_SECONDS = JOB_DEADLINE_SECONDS - 20.0
# 유실 판정. 진행 중인 작업은 단계가 바뀔 때마다 updated_at 을 고친다. 가장 긴 무변화 구간은 호출 1회
# (슬롯 대기 상한보다 길다)이므로, 그보다 여유만큼 더 오래 바뀌지 않았으면 작업이 멈춘 것이다.
LOST_IDLE_SECONDS = JOB_CALL_TIMEOUT_SECONDS + 60.0
# 만든 뒤 전체 제한 + 이만큼이 지나도 진행 중이면 유실(run_job 의 제한이 먼저 끝내므로 여유를 둔다).
LOST_GRACE_SECONDS = 30.0

# ── 보관 ──────────────────────────────────────────────────────────────────────


def _env_number(name: str, default: float, maximum: float) -> float:
    """보관 시간 환경변수. 비었거나 숫자가 아니거나 0 이하·무한·상한 초과면 기본값.

    상한을 두는 이유: '지우지 않음' 뜻으로 아주 큰 값을 넣으면 `now - timedelta(...)` 가 OverflowError 를 내
    작업 생성이 매번 500 이 된다.
    """
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) and 0 < value <= maximum else default


def result_ttl() -> timedelta:
    """끝난 작업의 결과 글을 지우기까지(`NARRATIVE_RESULT_TTL_HOURS`, 기본 24시간).

    프론트는 완료를 보면 결과를 평가 기록(run)에 저장하므로, 서버의 결과 글은 이어 조회용이다.
    """
    return timedelta(hours=_env_number("NARRATIVE_RESULT_TTL_HOURS", 24.0, maximum=24.0 * 365))


def record_retention() -> timedelta:
    """작업 행(내부 기록 포함)을 지우기까지(`NARRATIVE_JOB_RETENTION_DAYS`, 기본 90일).

    내부 기록은 결과 글을 지운 뒤에도 남기되, 무인증 경로라 무한히 쌓이면 발급 DB 가 찬다.
    """
    return timedelta(days=_env_number("NARRATIVE_JOB_RETENTION_DAYS", 90.0, maximum=3650.0))


# 칸(소절 번호) — 등록표 한 곳(app.narrative.sections).
SECTION_IDS = sections.SECTION_IDS


def get_session_factory():
    """작업 표에 쓰는 세션 팩토리(FastAPI 의존성). 테스트는 인메모리 DB 의 팩토리로 바꿔 끼운다.

    BackgroundTasks 로 도는 run_job 은 요청 범위의 세션을 쓸 수 없어(응답 뒤 닫힘) 팩토리를 넘겨받는다.
    """
    return database.SessionLocal


# ── 순수 함수 ─────────────────────────────────────────────────────────────────


def iso_utc(moment: datetime) -> str:
    """tz 없는 UTC 시각 → 'YYYY-MM-DDTHH:MM:SSZ'."""
    return moment.replace(microsecond=0).isoformat() + "Z"


def is_lost(*, created_at: datetime, updated_at: datetime, now: datetime) -> bool:
    """진행 중인 작업이 멈췄는가. 마지막 갱신 뒤 LOST_IDLE_SECONDS, 또는 만든 뒤 전체 제한 + 여유를 넘으면(초과)."""
    idle = (now - updated_at).total_seconds()
    age = (now - created_at).total_seconds()
    return idle > LOST_IDLE_SECONDS or age > JOB_DEADLINE_SECONDS + LOST_GRACE_SECONDS


def is_expired(*, finished_at: datetime | None, now: datetime) -> bool:
    """끝난 작업의 결과 글을 지울 때가 됐는가(초과)."""
    return finished_at is not None and now - finished_at > result_ttl()


def customer_result(gen: GenerationResult) -> dict:
    """생성 결과에서 고객용만 남긴다 — 서술 본문과 칸별 출처(생성기가 칸마다 정한 llm·fallback·fixed)."""
    resp = gen.response
    result = NarrativeJobResult(
        interpretation=resp.interpretation,
        conclusion=resp.conclusion,
        recommendation_narrative=resp.recommendation_narrative,
        recommendations=resp.recommendations,
        section_sources=gen.section_sources,
    )
    return result.model_dump(mode="json")


def total_cost(calls: list[dict]) -> float | None:
    """호출 기록의 비용 합. 하나라도 단가를 모르면(None) None — 추정하지 않는다."""
    if any(c.get("cost_usd") is None for c in calls):
        return None
    return round(sum(c["cost_usd"] for c in calls), 6)


def http_calls(calls: list[dict]) -> list[dict]:
    """요청이 실제로 나간 호출(슬롯을 못 얻었거나 남은 시간이 없어 부르지 않은 것은 뺀다)."""
    return [c for c in calls if c.get("http", True)]


# ── 입력 상한 초과 → 줄이기 ─────────────────────────────────────────────────────
# 클래스 단위 칸은 잘라 내면 남은 값으로 쓴 문장이 틀린다(예: 앞 256개 클래스 중에서 고른 "가장 약한 클래스").
# 그래서 클래스 단위 칸은 통째로 비우고, 클래스와 무관한 목록(지표·문제)과 글자는 상한까지 자른다.
# 서술용 파생값(facts)의 클래스 목록도 같다 — 최약 클래스 같은 결론은 프론트가 자르기 전 값으로 이미 정했다.
_CLASS_LEVEL_KEYS = {
    ("fact_sheet", "confusion"): None,
    ("fact_sheet", "distribution"): None,
    ("fact_sheet", "per_class"): [],
    ("fact_sheet", "class_averages"): [],
    ("fact_sheet", "confusion_counts"): [],
    ("fact_sheet", "facts", "low_support_classes"): [],
    ("fact_sheet", "facts", "prediction_bias", "rows"): [],
    ("fact_sheet", "facts", "distribution_shift"): [],
    ("fact_sheet", "facts", "omitted"): {},  # 키가 계약(칸 이름 몇 개)보다 많으면 — 버려도 서술은 된다
}
_FACTS = ("fact_sheet", "facts")
# facts 전체 길이 초과 때 1차로 비우는 목록(계약 size_rules — 프론트 fitFacts 의 마지막 두 단계와 같은 칸).
_FACTS_LISTS = ("low_support_classes", "distribution_shift", "fixed_sentences")


# 줄이기를 되풀이하는 최대 횟수. 겹친 목록은 바깥 → 안쪽 순으로 한 겹씩 드러난다(가장 깊은 것: 문제 목록 → 문제
# 하나의 칸). 몇 겹이면 충분하다 — 넘으면 고칠 수 없는 입력으로 보고 422.
_MAX_TRIM_ROUNDS = 4


def _is_limit_error(err: dict) -> bool:
    if err.get("type") in ("too_long", "string_too_long"):
        return True
    return isinstance((err.get("ctx") or {}).get("error"), LimitExceeded)


def _set_path(raw: dict, path: tuple, value) -> None:
    parent = raw
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value


def _clear_facts(raw: dict) -> str:
    """facts 전체 길이 초과 — 1차로 클래스 목록·고정 문장을 비우고, 이미 비어 있으면 facts 를 없앤다."""
    facts = raw["fact_sheet"]["facts"]
    bias = facts.get("prediction_bias") if isinstance(facts, dict) else None
    has_lists = isinstance(facts, dict) and (
        any(facts.get(key) for key in _FACTS_LISTS) or (isinstance(bias, dict) and bias.get("rows"))
    )
    if has_lists:
        for key in _FACTS_LISTS:
            facts[key] = []
        if isinstance(bias, dict):
            bias["rows"] = []
        return "fact_sheet.facts(lists)"
    raw["fact_sheet"]["facts"] = None
    return "fact_sheet.facts"


def _clear_or_trim(raw: dict, err: dict) -> str:
    """오류 하나를 고친다. 고친 자리(점 표기)를 돌려준다. 고칠 수 없는 모양이면 LookupError·TypeError."""
    loc = tuple(err["loc"])
    if loc[:1] == ("client_inputs",):
        raw["client_inputs"] = None  # 규칙 문장은 고객 입력을 쓰지 않는다
        return "client_inputs"
    for prefix, empty in _CLASS_LEVEL_KEYS.items():
        if loc[: len(prefix)] == prefix:
            _set_path(raw, prefix, copy.copy(empty))
            return ".".join(prefix)
    if loc == _FACTS:
        return _clear_facts(raw)
    parent = raw
    for key in loc[:-1]:
        parent = parent[key]
    last = loc[-1]
    if err["type"] in ("too_long", "string_too_long"):
        parent[last] = parent[last][: err["ctx"]["max_length"]]
    elif last == "threshold":  # 레이블별 임계값 dict
        parent[last] = None
    elif last == "evidence":
        parent[last] = {}
    else:
        raise LookupError(loc)
    return ".".join(str(k) for k in loc)


# 부드러운 오류 — 서술을 돕는 칸의 형식이 계약과 어긋남. 422 막다른 길 대신 그 칸만 버리고 AI 는 그대로 부른다.
# facts 안의 형식 오류(모르는 선언 코드 등) → facts 를 버린다(프론트가 자른 수 omitted 는 남긴다).
FACTS_DROPPED = "fact_sheet.facts(invalid)"
# 표시 문자열(value_text·margin_text·gain_text·imbalance_effect.display)이 숫자 꼴이 아님 → 그 칸만 버린다.
DISPLAY_DROPPED = "display_text(invalid)"
SOFT_REPAIRS = (FACTS_DROPPED, DISPLAY_DROPPED)
_DISPLAY_FIELDS = ("value_text", "margin_text", "gain_text")


def _soft_kind(err: dict) -> str | None:
    if _is_limit_error(err):
        return None
    loc = tuple(err["loc"])
    if loc[:2] == _FACTS:
        return FACTS_DROPPED
    if loc and (loc[-1] in _DISPLAY_FIELDS or "display" in loc):
        return DISPLAY_DROPPED
    return None


def _drop_facts_keep_omitted(raw: dict) -> None:
    """facts 를 버리되, 프론트가 표를 자른 수(omitted)는 남긴다 — 잘린 표를 전체 표로 읽지 않게(S4b 검토 BE-04)."""
    facts = raw["fact_sheet"].get("facts")
    omitted = facts.get("omitted") if isinstance(facts, dict) else None
    keep = (
        isinstance(omitted, dict)
        and set(facts) != {"omitted"}  # 이미 omitted 만 남겼는데 또 틀리면 omitted 가 틀린 것
        and all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) for k, v in omitted.items())
    )
    raw["fact_sheet"]["facts"] = {"omitted": omitted} if keep else None


def _repair_soft(raw, exc: ValidationError):
    """부드러운 오류를 고친 사본과 고친 자리 이름을 돌려준다. 고칠 수 없는 오류(부드럽지도 상한도 아님)가 있거나
    부드러운 오류가 없으면 None — 상한 오류는 호출부의 줄이기가 맡는다."""
    errors = exc.errors()
    kinds = [_soft_kind(e) for e in errors]
    if not any(kinds) or not all(k or _is_limit_error(e) for k, e in zip(kinds, errors)):
        return None
    fixed = copy.deepcopy(raw)
    labels: list[str] = []
    for kind, err in zip(kinds, errors):
        if kind == FACTS_DROPPED:
            if fixed["fact_sheet"].get("facts") is not None and (FACTS_DROPPED not in labels):
                _drop_facts_keep_omitted(fixed)
        elif kind == DISPLAY_DROPPED:
            loc = tuple(err["loc"])
            # display 의 세 칸 중 하나라도 틀리면 display 통째로(엔진 표시값은 셋이 한 묶음이다).
            target = loc[: loc.index("display") + 1] if "display" in loc else loc
            _set_path(fixed, target, None)
        else:
            continue
        if kind not in labels:
            labels.append(kind)
    return fixed, labels


def parse_request(raw) -> tuple[NarrativeRequest, list[str]]:
    """요청 본문을 검증한다. 고칠 수 있는 오류만 있으면 고쳐 (요청, 고친 자리 목록)을 돌려준다.

    - 부드러운 오류(facts 형식·표시 문자열 형식): 그 칸만 버린다. 고친 자리에 FACTS_DROPPED·DISPLAY_DROPPED 를
      적는다(상한 초과가 아니므로 호출부는 AI 를 그대로 부른다).
    - 상한 초과: 상한 안으로 줄인다(호출부는 규칙 문장으로 끝낸다 — input_too_large).
    그 밖의 오류(필수 칸 없음·형식 틀림)가 하나라도 있으면 원래 ValidationError 를 그대로 던진다 — 라우터가 422.

    고친 뒤 다시 검증하면 새 오류가 드러날 수 있다 — pydantic 은 목록 길이·facts 전체 길이가 넘으면 그 안쪽을
    검증하지 않는다(상한 오류 뒤에 형식 오류가 숨는다 — S4b 검토 BE-01). 그래서 고칠 수 있는 오류만 남는 동안
    몇 번 되풀이한다. 고치는 자리는 매번 줄거나 비기만 하므로 몇 번이면 끝난다.
    """
    try:
        return NarrativeRequest.model_validate(raw), []
    except ValidationError as exc:
        first = exc
    current = copy.deepcopy(raw)
    soft: list[str] = []
    trims: set[str] = set()
    last = first
    for _ in range(_MAX_TRIM_ROUNDS * 2):
        repaired = _repair_soft(current, last)
        if repaired is not None:
            current, labels = repaired
            soft.extend(l for l in labels if l not in soft)
        else:
            errors = last.errors()
            if not errors or not all(_is_limit_error(e) for e in errors):
                raise first
            try:
                trims.update(_clear_or_trim(current, e) for e in errors)
            except (LookupError, TypeError, KeyError):
                raise first from None
        try:
            return NarrativeRequest.model_validate(current), [*soft, *sorted(trims)]
        except ValidationError as again:
            last = again
    raise first


# ── DB (동기 — run_in_threadpool 로 부른다) ──────────────────────────────────────


def _db_insert(factory, job_id: str, benchmark_refs: list[dict], internal: dict, now: datetime) -> None:
    with factory() as db:
        # 정리를 먼저 한다 — 별도 스케줄러 없이, 작업이 생길 때마다 시간이 지난 것을 치운다.
        db.execute(
            update(NarrativeJob)
            .where(NarrativeJob.finished_at < now - result_ttl(), NarrativeJob.result_json.is_not(None))
            .values(result_json=None)
        )
        db.execute(delete(NarrativeJob).where(NarrativeJob.created_at < now - record_retention()))
        db.add(NarrativeJob(
            id=job_id, status="running", stage="queued",
            result_json=json.dumps({"benchmark_refs": benchmark_refs}, ensure_ascii=False),
            internal_json=json.dumps(internal, ensure_ascii=False),
            created_at=now, updated_at=now,
        ))
        db.commit()


def _db_set_stage(factory, job_id: str, stage: str, now: datetime) -> bool:
    """진행 중일 때만 단계를 바꾼다(끝난 작업 — 유실로 판정된 것 포함 — 은 그대로)."""
    with factory() as db:
        res = db.execute(
            update(NarrativeJob)
            .where(NarrativeJob.id == job_id, NarrativeJob.status == "running")
            .values(stage=stage, updated_at=now)
        )
        db.commit()
        return res.rowcount == 1


def _db_finish(factory, job_id: str, *, status: str, narrative: dict | None, internal: dict, now: datetime) -> bool:
    """진행 중일 때만 끝 상태를 쓴다. 이미 끝났으면(유실 판정 뒤 늦게 끝남) 아무것도 바꾸지 않고 False."""
    with factory() as db:
        job = db.get(NarrativeJob, job_id)
        if job is None or job.status != "running":
            return False
        result = json.loads(job.result_json) if job.result_json else {}
        if narrative is not None:
            result["narrative"] = narrative
        merged = {**json.loads(job.internal_json or "{}"), **internal}
        merged["stages"] = json.loads(job.internal_json or "{}").get("stages", []) + internal.get("stages", [])
        res = db.execute(
            update(NarrativeJob)
            .where(NarrativeJob.id == job_id, NarrativeJob.status == "running")
            .values(
                status=status, stage=status,
                result_json=json.dumps(result, ensure_ascii=False),
                internal_json=json.dumps(merged, ensure_ascii=False),
                updated_at=now, finished_at=now,
            )
        )
        db.commit()
        return res.rowcount == 1


def _db_read(factory, job_id: str, now: datetime) -> dict | None:
    """작업 하나를 읽는다. 진행 중인데 멈췄으면 여기서 유실로 끝내고, 보관 시간이 지난 결과 글은 지운다."""
    with factory() as db:
        job = db.get(NarrativeJob, job_id)
        if job is None:
            return None
        if job.status == "running" and is_lost(created_at=job.created_at, updated_at=job.updated_at, now=now):
            internal = json.loads(job.internal_json or "{}")
            internal["failure"] = "lost"
            internal["lost_judged_at"] = iso_utc(now)
            internal["last_stage"] = job.stage
            res = db.execute(
                update(NarrativeJob)
                .where(NarrativeJob.id == job_id, NarrativeJob.status == "running")
                .values(status="failed", stage="failed", updated_at=now, finished_at=now,
                        internal_json=json.dumps(internal, ensure_ascii=False))
            )
            db.commit()
            if res.rowcount == 1:
                logger.warning("서술 작업 %s 유실 판정(마지막 단계 %s)", job_id, internal["last_stage"])
            db.refresh(job)
        elif job.result_json is not None and is_expired(finished_at=job.finished_at, now=now):
            job.result_json = None
            db.commit()
            db.refresh(job)
        return {
            "id": job.id, "status": job.status, "stage": job.stage,
            "result": json.loads(job.result_json) if job.result_json else None,
            "failure": json.loads(job.internal_json or "{}").get("failure"),
            "created_at": job.created_at, "updated_at": job.updated_at,
        }


# ── 경로가 부르는 함수 ─────────────────────────────────────────────────────────


async def create_job(raw, *, client, session_factory, background_tasks) -> NarrativeJobCreated:
    """작업을 만들고 즉시 돌려준다. ValidationError(422)는 라우터가 응답으로 바꾼다.

    작업 행 시간당 상한을 넘으면 행을 만들지 않고 규칙 문장 결과를 이 응답에 바로 담는다(job_id 없음, status done).
    429 로 막으면 한 사람이 상한을 다 썼을 때 다른 모든 사용자가 1시간 동안 서술도 8.1 표도 받지 못한다(S3a 검토).
    """
    request, trimmed = await run_cpu_bound(parse_request, raw)
    benchmark_refs = [
        BenchmarkRefOut(**ref).model_dump(mode="json")
        for ref in build_benchmark_refs(request.task_type.value, request.fact_sheet.metrics)
    ]
    if not llm_budget.narrative_jobs.try_consume():
        logger.warning(
            "서술 작업 행 시간당 상한(%d건) 초과 → 행 없이 규칙 문장으로 바로 완료(job_limit)",
            llm_budget.narrative_jobs.limit,
        )
        gen = await generate_sections(None, request, forced_reason="job_limit")
        return NarrativeJobCreated(
            job_id=None, status="done", stage="done", benchmark_refs=benchmark_refs,
            result=NarrativeJobResult(**customer_result(gen)),
        )

    # AI 를 부를 작업만 서술 지갑에서 보고서 1건을 예약한다. 사유가 이미 정해진 작업은 AI 를 부르지 않는다.
    forced_reason = None
    if any(t in SOFT_REPAIRS for t in trimmed):
        logger.warning("서술 작업 입력의 형식이 계약과 어긋나 그 칸을 버리고 진행: %s", [t for t in trimmed if t in SOFT_REPAIRS])
    if [t for t in trimmed if t not in SOFT_REPAIRS]:
        forced_reason, budget = "input_too_large", "not_needed"
        logger.warning("서술 작업 입력이 상한을 넘어 줄임 → 규칙 문장으로 완료(input_too_large): %s", trimmed)
    elif client is None:
        budget = "no_key"
    elif llm_budget.narrative.try_consume():
        budget = "reserved"
    else:
        forced_reason, budget = "budget_exceeded", "exceeded"
        logger.warning("서술 시간당 예산 소진(보고서 %d건) → 규칙 문장으로 완료", llm_budget.narrative.limit)

    now = _utcnow()
    job_id = str(uuid.uuid4())
    internal = {
        "task_type": request.task_type.value,
        "report_purpose": request.report_purpose.value,
        "budget": budget,
        "trimmed": trimmed,
        "stages": [{"stage": "queued", "at": iso_utc(now)}],
    }
    try:
        await run_in_threadpool(_db_insert, session_factory, job_id, benchmark_refs, internal, now)
    except Exception:
        # 행을 못 남겼으면 쓴 지갑을 돌려준다 — DB 장애가 서술 지갑(옛 경로와 같이 씀)까지 비우지 않게.
        llm_budget.narrative_jobs.refund()
        if budget == "reserved":
            llm_budget.narrative.refund()
        raise
    background_tasks.add_task(
        run_job, job_id,
        session_factory=session_factory, client=client, request=request,
        budget_reserved=budget == "reserved", forced_reason=forced_reason,
    )
    return NarrativeJobCreated(job_id=job_id, status="running", stage="queued", benchmark_refs=benchmark_refs)


async def run_job(
    job_id: str,
    *,
    session_factory,
    client,
    request: NarrativeRequest,
    budget_reserved: bool = False,
    forced_reason: str | None = None,
) -> None:
    """작업 하나를 끝까지 돌린다(BackgroundTasks). 예외를 밖으로 던지지 않는다 — 끝 상태를 DB 에 쓴다.

    테스트는 이 함수를 직접 await 해 단계와 기록을 본다.
    """
    started = time.perf_counter()
    stages: list[dict] = []
    gen_trace: dict = {}
    # 장별 호출이 동시에 단계를 쓴다 — 한 번에 하나씩(SQLite 는 연결 하나를 스레드끼리 같이 쓰면 쓰기가 깨진다).
    stage_lock = asyncio.Lock()

    async def on_stage(stage: str) -> None:
        moment = _utcnow()
        stages.append({"stage": stage, "at": iso_utc(moment)})
        try:
            async with stage_lock:
                await run_in_threadpool(_db_set_stage, session_factory, job_id, stage, moment)
        except Exception:
            # 단계 표시는 진행 안내일 뿐이다 — 쓰기에 실패해도 서술은 계속 만든다(못 쓰면 유실 판정이 앞당겨질 뿐).
            logger.exception("서술 작업 %s 단계(%s) 기록 실패", job_id, stage)

    narrative = None
    failure = None
    try:
        async with asyncio.timeout(JOB_DEADLINE_SECONDS):
            # 작업 경로의 호출은 재시도 0·호출 시간 제한 JOB_CALL_TIMEOUT_SECONDS (옛 경로 클라이언트 설정과 따로).
            job_client = (
                client.with_options(timeout=JOB_CALL_TIMEOUT_SECONDS, max_retries=0) if client is not None else None
            )
            gen = await generate_sections(
                job_client, request,
                forced_reason=forced_reason,
                deadline_seconds=JOB_GENERATION_SECONDS, slot_wait=JOB_SLOT_WAIT_SECONDS,
                call_timeout=JOB_CALL_TIMEOUT_SECONDS, on_stage=on_stage,
                trace_into=gen_trace,
            )
            narrative = customer_result(gen)
    except TimeoutError:
        failure = "timeout"
        logger.warning("서술 작업 %s 전체 제한(%.0f초) 초과 → 실패(timeout)", job_id, JOB_DEADLINE_SECONDS)
    except Exception:
        failure = "error"
        logger.exception("서술 작업 %s 예기치 않은 오류 → 실패(error)", job_id)

    moment = _utcnow()
    status = "done" if failure is None else "failed"
    stages.append({"stage": status, "at": iso_utc(moment)})
    calls = gen_trace.get("calls", [])
    sent = http_calls(calls)
    internal = {
        "failure": failure,
        **{k: v for k, v in gen_trace.items() if k != "calls"},
        "calls": calls,
        "http_attempts": len(sent),
        # 응답을 받지 못한 호출(끊김)이 있으면 비용을 모른다(None) — 추정하지 않는다.
        "cost_usd": total_cost(sent),
        "seconds_total": round(time.perf_counter() - started, 3),
        "stages": stages,
    }
    try:
        written = await run_in_threadpool(
            _db_finish, session_factory, job_id, status=status, narrative=narrative, internal=internal, now=moment,
        )
    except Exception:
        logger.exception("서술 작업 %s 끝 상태(%s) 기록 실패 — 조회 때 유실로 판정된다", job_id, status)
        return
    if not written:
        logger.warning("서술 작업 %s 은 이미 끝나 있었다(유실 판정 뒤 늦게 끝남) — 결과를 버린다", job_id)
    logger.info("서술 작업 %s 기록 %s", job_id, json.dumps({"status": status, **internal}, ensure_ascii=False))


async def read_job(job_id: str, *, session_factory) -> NarrativeJobOut | None:
    """작업 조회. 없으면 None(라우터가 404)."""
    row = await run_in_threadpool(_db_read, session_factory, job_id, _utcnow())
    if row is None:
        return None
    result = row["result"]
    status, stage, failure = row["status"], row["stage"], row["failure"]
    narrative = None
    if status == "done":
        if result is None or "narrative" not in result:
            # 결과 글을 보관 시간이 지나 지웠다 — 고객에게는 다시 만들어야 하는 실패로 보인다.
            status, stage, failure = "failed", "failed", "expired"
        else:
            narrative = NarrativeJobResult(**result["narrative"])
    return NarrativeJobOut(
        job_id=row["id"], status=status, stage=stage, failure=failure,
        benchmark_refs=[BenchmarkRefOut(**r) for r in (result or {}).get("benchmark_refs", [])],
        result=narrative,
        created_at=iso_utc(row["created_at"]), updated_at=iso_utc(row["updated_at"]),
    )
