"""app/narrative/generator.py — 작업 경로의 서술 생성기: 장별 호출 + 칸별 검문 사다리(서술 라운드 S5, 서술 설계 D36).

흐름(계획 설계 요지 2·5)
1. 첫 생성은 장마다 한 번 — 7장(7.1·7.2·7.3)·8장(8.1·8.2·8.3)·9장(9.1·9.2·권고표)을 동시에 부른다. 문제가 0개면 9장은
   부르지 않는다(9칸 = 코드가 정한 문장, 출처 fixed).
2. 호출이 실패하면: 429(사용 한도)는 차례로 한 번 더, 호출 오류·시간 초과·JSON 오류는 한 번 더(SDK 재시도 0 — 앱에서 1회).
   거절(refusal 세 경우)·잘림(max_tokens)·모델 없음은 같은 요청으로 다시 부르지 않고 그 장의 칸을 규칙 템플릿으로.
   거절된 요청은 약 1시간 대체 모델로 고정 라우팅되므로 같은 요청을 되풀이하지 않는다.
3. 칸마다 검사(app.narrative.checks): 숫자·판정 모순·9칸 예측 수치/빠진 권고·D24 선언 숫자·형식.
4. 걸린 칸만 1회 다시(그 칸만 요청하는 호출 — 같은 사실·결론, 위반 숫자는 넣지 않는다). 또 걸리면 그 칸만 템플릿.
   D24 만 걸리면 다시 만든 뒤 그대로 싣고 기록한다. 숫자와 D24 가 함께 걸려도 다시 만들기는 1회.
   7.3(추가 관찰)은 걸리면 다시 만들지 않고 뺀다(성적서는 고정 문구 15 "해당 없음" — D38).
   한 칸의 HTTP 호출은 최대 2회(첫 장 호출이 재시도됐으면 그 칸은 다시 만들지 않는다).
5. 칸을 섞은 최종본에 판정 모순 검사를 한 번 더 — 걸린 AI 칸은 템플릿으로.
6. 기록: 고객용은 칸별 출처(section_sources)만, 내부용(trace)은 호출마다 응답 모델·거절 종류·토큰·비용·슬롯 대기·
   칸마다 검사 결과·시도 수·7.3 원문(추가 관찰 승격 자료). 입력(사실 시트·고객 입력)은 남기지 않는다.

시간: 작업 전체 제한(jobs.JOB_DEADLINE_SECONDS) 안에서 호출마다 남은 시간으로 제한을 줄인다. 남은 시간이 MIN_CALL_SECONDS
보다 적으면 부르지 않고 그 칸을 템플릿으로(사유 deadline) — 다시 만들기 하나가 길어져 작업 전체가 실패로 끝나지 않게.

상호작용
- 의존(import): app.narrative.llm·prompt·checks·fallback·required_items·sections·grounding·derived·baselines,
  app.core.concurrency(narrative_slot·run_cpu_bound)
- 사용처: app.narrative.jobs(run_job·작업 행 상한), tests
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import asdict, dataclass, field

from app.core.concurrency import narrative_slot, run_cpu_bound
from app.narrative import checks, fallback, prompt
from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.grounding import _add, build_number_whitelist, find_verdict_contradictions
from app.narrative.llm import EFFORT_LEVELS, call_narrative_llm
from app.narrative.required_items import select_items
from app.narrative.schemas import (
    GroundingInfo,
    NarrativeMeta,
    NarrativeRequest,
    NarrativeResponse,
    RecommendationOut,
)
from app.narrative.sections import CHAPTERS, SECTION_IDS, SECTIONS_BY_ID

logger = logging.getLogger(__name__)

# 작업 경로 effort(사용자 결정 "품질 우선 — 5~10분 허용"). 코드 기본값 = 운영값. S5 실측(2026-10-21): high 로 장 셋 동시
# 호출이 약 26초·한 건 약 0.13달러, 장별 출력 2~3천 토큰(상한 16000 에 여유) — high 로 정했다.
DEFAULT_JOB_EFFORT = "high"

# 남은 시간이 이보다 적으면 호출하지 않는다(응답을 받아도 검사·기록할 시간이 없다).
MIN_CALL_SECONDS = 30.0

# 같은 요청을 다시 불러 풀릴 수 있는 실패 — 앱에서 1회 더. 거절·잘림·모델 없음은 다시 부르지 않는다.
RETRYABLE = frozenset({"api_error", "timeout", "invalid_json", "rate_limited"})

GENERATOR_VERSION = "sections-1"


def job_effort() -> str:
    """작업 경로 effort(`NARRATIVE_JOB_EFFORT`). 비었거나 모르는 값이면 코드 기본값(오타가 400 이 되지 않게)."""
    value = os.getenv("NARRATIVE_JOB_EFFORT", "").strip().lower()
    return value if value in EFFORT_LEVELS else DEFAULT_JOB_EFFORT


@dataclass
class SectionState:
    """칸 하나의 진행 — 최종 값·출처·시도·검사 기록."""
    id: str
    parts: dict | None = None
    source: str = "fallback"
    http_calls: int = 0
    checks: list[dict] = field(default_factory=list)
    outcome: str = ""  # first · regenerated · template · dropped · fixed
    reason: str | None = None
    d24_missing: bool = False
    # 첫 시도가 막는 사유 없이 D24 만 빠졌거나(("d24", 값)) 9칸이 권고 하나만 빠졌을 때(("missing_rec", 값)) 그 글을 둔다.
    # 다시 만들기가 실패하거나 또 걸리면 이 글을 싣는다 — D24 빠짐은 템플릿으로 바꾸는 사유가 아니다(S5 검토 ladder-2).
    kept: tuple[str, dict] | None = None


@dataclass
class GenerationResult:
    response: NarrativeResponse
    section_sources: dict[str, str]
    trace: dict


async def _no_stage(_stage: str) -> None:
    return None


def _prepare(req: NarrativeRequest):
    fs = req.fact_sheet
    task = req.task_type.value
    derived = compute_derived(fs, task_type=task)
    refs = build_benchmark_refs(task, fs.metrics)
    whitelist = build_number_whitelist(fs, refs, derived, req.client_inputs)
    conclusions = select_items(fs, req.client_inputs, task)
    ctx = prompt.build_context(
        fs, derived, refs, task_type=task, report_purpose=req.report_purpose.value,
        client_inputs=req.client_inputs, conclusions=conclusions,
    )
    # 지시문에서 덜어 낸 표의 개수('외 k개')도 서술이 쓸 수 있다 — 원본 사실 시트에 없는 수일 수 있어 넣는다.
    for count in (ctx.view.get("덜어 낸 표(일부만 실림)") or {}).values():
        _add(whitelist, count)
    exempt = checks.name_exemptions(fs, req.client_inputs)
    masks = checks.verdict_masks(fs, req.client_inputs)
    return derived, refs, whitelist, conclusions, ctx, exempt, masks


def _extract(section_id: str, data: dict) -> dict | None:
    """AI 응답에서 칸 하나의 값(응답 필드 점 표기 → 값). 필드가 없거나 모양이 틀리면 None."""
    section = SECTIONS_BY_ID[section_id]
    parts: dict = {}
    for name, path in zip(section.ai_fields, section.response_paths):
        value = data.get(name)
        if name == "recommendations":
            if not isinstance(value, list) or not all(isinstance(r, dict) for r in value):
                return None
            parts[path] = [
                {k: str(r.get(k) or "").strip() for k in ("problem_id", "category", "action", "expected_impact")}
                for r in value
            ]
        else:
            if not isinstance(value, str):
                return None
            parts[path] = value.strip()
    return parts


def _align(parts: dict, problems: list, *, fill_missing: bool) -> tuple[dict, list[str]]:
    """권고를 문제 목록에 맞춘다(D22) — 목록 밖 번호는 버리고, 우선순위는 문제의 것, 문제 번호 순서.

    fill_missing 이면 권고가 없는 문제를 종류별 폴백 템플릿으로 채운다. 채운 문제 번호를 돌려준다."""
    by_id: dict[str, list[dict]] = {p.id: [] for p in problems}
    for r in parts.get("recommendations") or []:
        if r["problem_id"] in by_id:
            by_id[r["problem_id"]].append(r)
    aligned: list[RecommendationOut] = []
    filled: list[str] = []
    for p in problems:
        if not by_id[p.id]:
            if fill_missing:
                aligned.append(fallback.problem_recommendation(p))
                filled.append(p.id)
            continue
        for r in by_id[p.id]:
            aligned.append(RecommendationOut(
                problem_id=p.id, priority=p.priority,
                category=r["category"], action=r["action"], expected_impact=r["expected_impact"],
            ))
    return {**parts, "recommendations": aligned}, filled


def _rec_dicts(parts: dict) -> dict:
    """검사용 — 권고를 dict 로(RecommendationOut 이면 바꾼다)."""
    recs = parts.get("recommendations")
    if not recs:
        return parts
    return {**parts, "recommendations": [r.model_dump() if isinstance(r, RecommendationOut) else r for r in recs]}


async def generate_sections(
    client,
    req: NarrativeRequest,
    *,
    forced_reason: str | None = None,
    deadline_seconds: float | None = None,
    slot_wait: float | None = None,
    call_timeout: float | None = None,
    effort: str | None = None,
    on_stage=_no_stage,
    trace_into: dict | None = None,
) -> GenerationResult:
    """칸별 사다리로 서술을 만든다. 호출 실패는 칸 템플릿으로 끝내고, 예기치 않은 오류는 호출부가 받는다.

    trace_into 를 주면 내부 기록을 그 dict 에 바로 쓴다 — 바깥 제한(작업 전체 10분)이 도중에 끊어도 그때까지의 호출
    기록이 남는다(요청은 이미 나가 과금됐을 수 있다)."""
    started = time.perf_counter()
    deadline_at = started + deadline_seconds if deadline_seconds is not None else None
    fs = req.fact_sheet
    derived, refs, whitelist, conclusions, ctx, exempt, masks = await run_cpu_bound(_prepare, req)
    effort = effort or job_effort()
    problem_ids = [p.id for p in fs.problems]
    states = {sid: SectionState(sid) for sid in SECTION_IDS}
    trace: dict = trace_into if trace_into is not None else {}
    trace.update({
        "generator": GENERATOR_VERSION,
        "effort": effort,
        "calls": [],
        "sections": {},
        "rate_limited": 0,
        "prompt_trimmed": ctx.trimmed,
        "client_keys_dropped": ctx.dropped_client_keys,
    })
    if ctx.dropped_client_keys:
        logger.warning("서술 고객 입력의 모르는 묶음을 AI 에 보내지 않음(D34): %s", ctx.dropped_client_keys)

    def template(sid: str, reason: str) -> None:
        st = states[sid]
        st.parts = fallback.section_template(
            sid, fs, refs, derived, task_type=req.task_type.value, client_inputs=req.client_inputs, conclusions=conclusions,
        )
        if SECTIONS_BY_ID[sid].fixed_without_ai or (sid == "9" and not fs.problems):
            st.source, st.outcome = "fixed", "fixed"  # 7.3 을 검문으로 뺐으면 호출부가 dropped 로 덮는다
        else:
            st.source, st.outcome = "fallback", "template"
        st.reason = reason

    # 문제가 0개면 9칸은 코드가 정한 문장 — 9장을 부르지 않는다(화-1).
    if not fs.problems:
        template("9", "no_problems")
        states["9"].outcome = "fixed"
        trace["chapter_9_skipped"] = True

    def finish(reason: str | None) -> GenerationResult:
        for sid, st in states.items():
            if st.parts is None:
                template(sid, reason or "not_generated")
        return _result(fs, states, trace, reason, started, derived=derived, refs=refs)

    if forced_reason is not None:
        return finish(forced_reason)
    if client is None:
        return finish("no_key")
    if not prompt.fits(ctx):
        logger.warning("서술 지시문이 상한(%d자)을 넘음 — 덜어 내도 넘어 규칙 문장(input_too_large)", prompt.MAX_PROMPT_CHARS)
        return finish("input_too_large")

    async def stage(name: str) -> None:
        # 단계가 같아도 매번 알린다 — 작업 행의 마지막 갱신 시각이 유실 판정(jobs.is_lost)의 기준이라, 슬롯 대기·차례
        # 호출이 길어질 때마다 갱신해야 멈춘 작업으로 오인되지 않는다.
        await on_stage(name)

    async def call(section_ids: tuple[str, ...], purpose: str):
        """한 번 부른다. (LLMResult 또는 None, 실패 사유) — 슬롯 대기·남은 시간·호출 제한을 여기서 본다."""
        entry: dict = {"purpose": purpose, "sections": list(section_ids), "http": False}
        trace["calls"].append(entry)
        def left() -> float | None:
            return None if deadline_at is None else deadline_at - time.perf_counter()

        remaining = left()
        if remaining is not None and remaining < MIN_CALL_SECONDS:
            entry["failure"] = "deadline"
            return None, "deadline"
        # 슬롯 대기도 남은 시간 안에서 — 호출할 시간(MIN_CALL_SECONDS)을 남긴다(S5 검토 ladder-1).
        wait_cap = slot_wait
        if remaining is not None:
            room = remaining - MIN_CALL_SECONDS
            wait_cap = room if slot_wait is None else min(slot_wait, room)
        user = prompt.build_user_message(ctx, section_ids, regenerate=purpose == "regenerate")
        schema = prompt.response_schema(section_ids, problem_ids)
        entry["prompt_chars"] = len(prompt.SYSTEM_PROMPT) + len(user)
        waited_from = time.perf_counter()
        slot = {"held": False}
        try:
            async with contextlib.AsyncExitStack() as stack:
                async with asyncio.timeout(wait_cap):
                    await stack.enter_async_context(narrative_slot())
                slot["held"] = True
                entry["slot_wait_seconds"] = round(time.perf_counter() - waited_from, 3)
                # 슬롯을 기다린 만큼 남은 시간이 줄었다 — 호출 제한을 지금 남은 시간으로 정한다. 모자라면 부르지 않는다
                # (요청이 나가지 않아 과금 없음). 그러지 않으면 작업 전체 제한을 넘겨 다 만든 칸까지 잃는다.
                remaining = left()
                if remaining is not None and remaining < MIN_CALL_SECONDS:
                    entry["failure"] = "deadline"
                    return None, "deadline"
                limit = call_timeout if remaining is None else (
                    remaining if call_timeout is None else min(call_timeout, remaining)
                )
                # 첫 장 호출은 '생성 중', 장 재시도·칸 다시 만들기는 '다시 만드는 중'(S5 검토 jobs-3).
                await stage("generating" if purpose == "chapter" else "regenerating")
                for sid in section_ids:
                    states[sid].http_calls += 1
                entry["http"] = True  # 요청이 나간다 — 끊겨도 과금됐을 수 있다(비용 모름)
                entry["started_at_seconds"] = round(time.perf_counter() - started, 3)
                async with asyncio.timeout(limit):
                    result = await call_narrative_llm(
                        client, system=prompt.SYSTEM_PROMPT, user=user, schema=schema, effort=effort,
                    )
        except TimeoutError:
            failure = "timeout" if slot["held"] else "busy"
            entry["failure"] = failure
            return None, failure
        entry.update(asdict(result.record))
        if result.failure is not None:
            entry["failure"] = result.failure
            if result.failure == "rate_limited":
                trace["rate_limited"] += 1
        return result, result.failure

    def accept(sid: str, parts: dict, *, last_chance: bool) -> bool:
        """검사해서 받으면 True. 걸리면 False(호출부가 다시 만들기·템플릿을 정한다). last_chance 면 늘 끝을 낸다."""
        st = states[sid]
        aligned = _align(parts, fs.problems, fill_missing=False)[0] if sid == "9" else parts
        result = checks.check_section(
            sid, _rec_dicts(aligned), whitelist=whitelist, verdict=fs.verdict, conclusions=conclusions,
            problem_ids=problem_ids, exempt=exempt, masks=masks,
        )
        st.checks.append(result.as_record())
        hard = result.hard_failures
        if sid == "7.3":
            raw = parts.get("interpretation.additional_observations", "")
            trace.setdefault("additional_observations_raw", []).append(raw)
            if hard:
                template(sid, "check:" + ",".join(hard))
                st.outcome = "dropped"  # 다시 만들지 않는다 — 성적서는 고정 문구 15(D38)
            elif raw:
                st.parts, st.source, st.outcome = parts, "llm", "first"
            else:
                template(sid, "no_observation")
            return True
        if not last_chance and not hard and result.d24_missing:
            st.kept = ("d24", aligned)
        if not last_chance and sid == "9" and hard == ["missing_recommendation"]:
            st.kept = ("missing_rec", parts)
        if sid == "9" and last_chance and hard == ["missing_recommendation"]:
            # 다시 부른 뒤에도 권고가 빠진 문제만 남았다 — D22 대로 종류별 폴백 템플릿으로 채운다(9칸은 규칙 문장이 섞임).
            filled_parts, filled = _align(parts, fs.problems, fill_missing=True)
            st.parts, st.source, st.outcome, st.reason = filled_parts, "fallback", "filled", "missing_recommendation"
            trace["recommendations_from_template"] = filled
            return True
        if not hard and (not result.d24_missing or last_chance):
            st.parts, st.source = aligned, "llm"
            st.outcome = "first" if st.http_calls <= 1 else "second"
            st.d24_missing = result.d24_missing
            return True
        if last_chance:
            rescue(sid, "check:" + ",".join(hard or ["d24_missing"]))
            return True
        return False

    def rescue(sid: str, reason: str) -> None:
        """다시 만들기가 실패했거나 또 걸렸다. 첫 글을 둬 뒀으면 그것을, 아니면 그 칸의 규칙 템플릿을 싣는다."""
        st = states[sid]
        if st.kept is None:
            template(sid, reason)
            return
        kind, parts = st.kept
        st.checks.append({"kept_first": reason})
        if kind == "d24":
            st.parts, st.source, st.outcome, st.d24_missing, st.reason = parts, "llm", "kept_first", True, None
        else:
            filled_parts, filled = _align(parts, fs.problems, fill_missing=True)
            st.parts, st.source, st.outcome, st.reason = filled_parts, "fallback", "filled", "missing_recommendation"
            trace["recommendations_from_template"] = filled

    async def run_batch(jobs: list[tuple[tuple[str, ...], str]]):
        """호출 묶음 — 동시에 부르고, 429 로 끝난 것은 차례로 한 번 더(나머지 재시도는 동시에)."""
        results = await asyncio.gather(*(call(ids, purpose) for ids, purpose in jobs))
        return list(zip(jobs, results))

    # 1. 첫 생성 — 장별 동시.
    chapters = [CHAPTERS["7"], CHAPTERS["8"]] + ([CHAPTERS["9"]] if fs.problems else [])
    first = await run_batch([(ids, "chapter") for ids in chapters])
    await stage("checking")

    retries: list[tuple[str, ...]] = []
    sequential: list[tuple[str, ...]] = []
    regenerate: list[str] = []
    for (ids, _), (result, failure) in first:
        if failure is None:
            for sid in ids:
                parts = _extract(sid, result.data)
                if parts is None:
                    states[sid].checks.append({"format": ["missing_field"]})
                    if sid == "7.3":
                        template(sid, "missing_field")
                    else:
                        regenerate.append(sid)
                elif not accept(sid, parts, last_chance=False):
                    regenerate.append(sid)
        elif failure == "rate_limited":
            sequential.append(ids)
        elif failure in RETRYABLE:
            retries.append(ids)
        else:  # refusal · max_tokens · model_unavailable · busy · deadline
            for sid in ids:
                template(sid, failure)

    # 2. 다시 부르기(장 재시도)와 다시 만들기(칸) — 한 칸의 HTTP 는 최대 2회.
    if retries or sequential or regenerate:
        await stage("regenerating")
    second: list = []
    batch = [(ids, "retry") for ids in retries] + [((sid,), "regenerate") for sid in regenerate]
    if batch:
        second.extend(await run_batch(batch))
    for ids in sequential:  # 429 는 차례로
        await stage("regenerating")
        second.extend(await run_batch([(ids, "retry")]))
    for (ids, purpose), (result, failure) in second:
        if failure is not None:
            for sid in ids:
                rescue(sid, failure)
            continue
        for sid in ids:
            parts = _extract(sid, result.data)
            if parts is None:
                rescue(sid, "missing_field")
            else:
                accept(sid, parts, last_chance=True)
    if second:
        await stage("checking")

    # 3. 최종본 판정 모순 재검사 — 인쇄될 칸 글(AI 글, 권고를 템플릿으로 채운 9칸 포함)을 한 번 더 본다. 칸 검사와 같은 함수라
    #    보통은 아무것도 잡지 않는다 — 조립·채우기 단계의 결함에 대한 마지막 방어다(칸 검사를 끈 테스트로 따로 시험한다).
    #    걸린 칸은 템플릿으로(다시 부르지 않는다).
    final_flags = []
    for sid, st in states.items():
        if st.parts is None or not (st.source == "llm" or st.outcome == "filled"):
            continue
        if find_verdict_contradictions(checks.mask_names(checks.section_texts(sid, _rec_dicts(st.parts)), masks), fs.verdict):
            final_flags.append(sid)
            template(sid, "final_verdict_contradiction")
    trace["final_recheck"] = final_flags
    reasons = [st.reason for st in states.values() if st.source == "fallback" and st.reason]
    return _result(fs, states, trace, reasons[0] if reasons else None, started, derived=derived, refs=refs)


def _result(fs, states: dict, trace: dict, reason: str | None, started: float, *, derived, refs) -> GenerationResult:
    parts: dict = {}
    for sid in SECTION_IDS:
        parts.update(states[sid].parts)
    sources = {sid: states[sid].source for sid in SECTION_IDS}
    non_fixed = [v for v in sources.values() if v != "fixed"]
    overall = "llm" if non_fixed and all(v == "llm" for v in non_fixed) else "fallback"
    llm_models = [c.get("response_model") for c in trace.get("calls", []) if c.get("response_model")]
    violations = sorted({v for st in states.values() for c in st.checks for v in c.get("numbers", [])})
    response = fallback.assemble_response(
        parts, fs,
        NarrativeMeta(
            source=overall,
            model=llm_models[0] if llm_models else None,
            reason=None if overall == "llm" else reason,
            grounding=GroundingInfo(passed=not violations, violations=violations),
        ),
    )
    trace["sections"] = {
        sid: {
            "source": st.source, "outcome": st.outcome, "reason": st.reason, "http_calls": st.http_calls,
            "d24_missing": st.d24_missing, "checks": st.checks,
        }
        for sid, st in states.items()
    }
    trace["source"] = overall
    trace["reason"] = None if overall == "llm" else reason
    trace["seconds"] = round(time.perf_counter() - started, 3)
    return GenerationResult(response=response, section_sources=sources, trace=trace)
