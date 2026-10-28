"""app/judge/core.py — 채점 한 번(judge_once, 설계 5절 구조, 계획 12-3).

코드 오케스트레이션: 공용 전처리(주장 추출)·필수 내용·표기를 같이 시작하고, 주장 목록이 나오면 사실 일치·일관성을 같이 돌린 뒤
코드 합산·게이트·안전장치. 네 모듈은 서로의 결과를 보지 않는다(주장 목록은 사실 일치·일관성에만).
사다리(재생성·규칙 문장)·작업 흐름·검증 실행기·파일럿이 모두 이 함수 하나를 쓴다 — 주어진 서술을 채점만 한다(생성 없음).

실패(보류 사유): no_material(심판 읽기 재료 없음·틀림, 조회 자료 빔 — Q4 f), 모듈 호출 실패 종류(timeout·transient·refusal·
bad_output·uncited·no_claims·agent_turns·deadline·no_client 등), safety(안전장치). 실패면 게이트가 없다(통과로 새지 않는다).

상호작용
- 의존(import): app.judge 의 material·reading·sources·extract·faithfulness·consistency·coverage·readability·aggregate·safety·
  criteria·config, app.narrative.derived(생성기와 같은 파생값 계산 — 데이터)
- 사용처: (J5a) ladder, scripts/judge_live.py(파일럿·검증)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.judge import aggregate, config, criteria, safety
from app.judge.consistency import judge_consistency
from app.judge.coverage import judge_coverage
from app.judge.extract import extract_claims
from app.judge.faithfulness import judge_faithfulness
from app.judge.material import parse_material
from app.judge.readability import judge_readability
from app.judge.reading import build_pieces
from app.judge.sources import Sources
from app.narrative.derived import compute_derived


@dataclass
class JudgeRun:
    gate: str | None  # pass | blocked | None(실패)
    failure: str | None
    flagged_slots: list[str] = field(default_factory=list)
    unfixable: list[dict] = field(default_factory=list)
    scorecard: dict | None = None
    record: dict = field(default_factory=dict)  # 내부 기록(모듈별 결과·호출·시간)

    def cost_usd(self) -> float | None:
        calls = [c for m in ("extract", "faithfulness", "consistency", "coverage", "readability")
                 for c in (self.record.get(m) or {}).get("calls", [])]
        costs = [c.get("cost_usd") for c in calls]
        return None if any(c is None for c in costs) else round(sum(costs), 6)


def lookup_data_empty(fs) -> bool:
    """조회 자료(혼동행렬 건수·클래스별 표)가 비었나 — 프론트가 본문 예산으로 비운 드문 요청(Q4 f)."""
    return not fs.confusion_counts and not fs.per_class


async def judge_once(client, req, result, section_sources: dict, benchmark_refs: list[dict], *,
                     deadline: float | None = None, clock=None) -> JudgeRun:
    clock_fn = clock or time.monotonic
    kw = {"deadline": deadline, **({"clock": clock} if clock else {})}
    started = clock_fn()
    timings: dict[str, float] = {}
    record: dict = {"criteria_version": criteria.version(), "criteria_fingerprint": criteria.fingerprint(),
                    "judge_model": config.judge_model(), "extract_model": config.extract_model(), "effort": config.judge_effort()}
    material, problem = parse_material(req.judge_material)
    record["material_problem"] = problem
    record["material_check_version"] = material.material_check_version if material else None
    fs = req.fact_sheet
    if material is None or lookup_data_empty(fs):
        record["seconds"] = {"total": 0.0}
        return JudgeRun(None, "no_material", record=record)
    pieces = build_pieces(result, section_sources, material, benchmark_refs, fs.metrics)
    task = req.task_type.value
    sources = Sources(fs.model_dump(), benchmark_refs, compute_derived(fs, task_type=task), req.client_inputs)
    record["pieces"] = [{"id": p.id, "slot": p.slot, "place": p.place, "source": p.source} for p in pieces]

    async def timed(name, coro):
        t = clock_fn()
        try:
            return await coro
        finally:
            timings[name] = round(clock_fn() - t, 3)

    ext, cov, rea = await asyncio.gather(
        timed("extract", extract_claims(client, pieces, sources.numbers, **kw)),
        timed("coverage", judge_coverage(client, pieces, task, fs, req.client_inputs, benchmark_refs, **kw)),
        timed("readability", judge_readability(client, pieces, fs, req.client_inputs, **kw)),
    )
    record.update({"extract": ext.as_record(), "coverage": cov.as_record(), "readability": rea.as_record()})
    first_failure = ext.failure or cov.failure or rea.failure
    if first_failure:
        record["seconds"] = {**timings, "total": round(clock_fn() - started, 3)}
        return JudgeRun(None, first_failure, record=record)
    faith, cons = await asyncio.gather(
        timed("faithfulness", judge_faithfulness(client, pieces, ext.claims, sources, fs.metrics, benchmark_refs, **kw)),
        timed("consistency", judge_consistency(client, pieces, ext.claims, fs, **kw)),
    )
    record.update({"faithfulness": faith.as_record(), "consistency": cons.as_record()})
    if faith.failure or cons.failure:
        record["seconds"] = {**timings, "total": round(clock_fn() - started, 3)}
        return JudgeRun(None, faith.failure or cons.failure, record=record)
    t = clock_fn()
    issues = safety.problems(pieces, faith, cons, cov)
    record["safety"] = issues
    if issues:
        record["seconds"] = {**timings, "total": round(clock_fn() - started, 3)}
        return JudgeRun(None, "safety", record=record)
    slots, unfixable = aggregate.flagged_slots(faith, cons)
    card = aggregate.scorecard(faith, cons, cov, rea)
    timings["aggregate"] = round(clock_fn() - t, 3)
    record["seconds"] = {**timings, "total": round(clock_fn() - started, 3)}
    return JudgeRun(aggregate.gate(faith, cons), None, slots, unfixable, card, record)
