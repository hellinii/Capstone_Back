"""app/judge/ladder.py — 발급 불가 흐름(사다리, 설계 5.5 6단계, 계획 13-2·정6).

채점 1 → 걸린 칸 재생성 1번 → 채점 2 → 걸린 칸 규칙 문장 → 채점 3 → 서술 칸 전부 규칙 문장 → 채점 4 → 보류.
- 재생성은 작업당 한 바퀴(1차 채점에 걸린 칸만, 정6 ①). 2차 채점에 걸린 칸은 새로 걸렸든 또 걸렸든 바로 규칙 문장.
- 이미 규칙 문장·고정 문구인 칸은 다시 만들지 않는다(정6 ②). 7.3 은 비우고 '해당 없음'(정6 ③).
- 단(rung)이 아무 칸도 바꾸지 못하면(걸린 칸이 모두 규칙 문장·표·고정 문구) 같은 글을 다시 채점하지 않고 다음 단으로 간다.
  마지막 단도 바꿀 것이 없으면 보류.
- 남은 시간이 재생성 + 다음 채점에 모자라면 재생성 단을 건너뛰고 규칙 문장 단으로(정6 ⑥). 채점할 시간도 없으면 보류(timeout).
- 재채점은 늘 서술 전체(같은 judge 함수 — 채점 한 번은 core.judge_once).
- 채점이 실패하면(게이트 없음) 그 자리에서 보류 — 통과로 새지 않는다.

심판 사유는 생성 쪽으로 넘기지 않는다(계획 13-3): 생성 세션에는 칸 id 만 넘긴다(judge_regenerate·to_template).

보류 종류(고객용 심판 묶음 계약의 held_kinds)와 "서술 다시 만들기" 허용(Q6 답, 2026-10-29)은 여기 한 곳(held_kind·
retry_allowed)이 정한다 — 작업 흐름(J5b)의 키 없음·지갑·작업 행 상한·바깥 시간 초과도 이 두 함수를 부른다.

상호작용
- 의존(import): app.judge.core(JudgeRun — 형), app.narrative.generator(GenerationSession — 형)
- 사용처: app.narrative.jobs 의 판 4 심판 갈래, tests
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

# 채점 한 번에 남겨 둘 최소 시간(초). J4 파일럿 실측 채점 1회 70~158초(AI 서술 약 150초) — J5b 끝에 확정(계획 14 "10분 안의 숫자").
JUDGE_MIN_SECONDS = 180.0
# 재생성 단에 줄 최소 시간(초). 생성 1회 실측 27~57초(봉인 6건) — J5b 끝에 확정.
REGENERATE_MIN_SECONDS = 60.0

# 보류 종류 — 고객용 심판 묶음 계약(tests/contract/judge_bundle_ed4.json)의 held_kinds 와 같은 집합(시험이 맞댄다).
HELD_KINDS = ("gate_failed", "timeout", "judge_error", "no_key", "budget", "job_limit", "no_material")

# 심판이 판정을 내지 못하고 끝난 일시적 사유(Q6) — 다시 하면 풀릴 수 있다. 4xx(api_error)·거절·형식 오류·안전장치는 아니다.
# busy = 심판 슬롯을 남은 시간 안에 얻지 못함(작업 흐름).
TRANSIENT_FAILURES = frozenset({"transient", "timeout", "busy"})

# 단 이름(내부 기록) — 채점 뒤에 무엇을 했나.
RUNG_REGENERATE = "regenerate"
RUNG_TEMPLATE_FLAGGED = "template_flagged"
RUNG_TEMPLATE_ALL = "template_all"


def held_kind(failure: str) -> str:
    """채점 실패 사유(core.judge_once·사다리) → 고객용 보류 종류. 호출 하나의 timeout(APITimeoutError)은 작업 시간 초과가
    아니라 심판 호출 실패다(judge_error). 작업 흐름이 직접 정하는 종류(no_key·budget·job_limit·바깥 timeout)는 이 함수를
    거치지 않고 retry_allowed 만 부른다."""
    if failure == "gate_failed":
        return "gate_failed"
    if failure == "no_material":
        return "no_material"
    if failure == "no_client":
        return "no_key"
    if failure == "deadline":
        return "timeout"
    return "judge_error"


def retry_allowed(kind: str, *, failure: str | None, ever_blocked: bool) -> bool:
    """보류에 '서술 다시 만들기'를 줄지(Q6 답) — 심판 지갑 소진, 또는 게이트에 한 번도 걸리지 않고 일시적 사유로 끝난 보류."""
    if kind == "budget":
        return True
    if ever_blocked or kind not in ("timeout", "judge_error"):
        return False
    return kind == "timeout" or failure in TRANSIENT_FAILURES


@dataclass
class LadderOutcome:
    gate: str  # passed | held
    held_kind: str | None
    retry_allowed: bool
    final: object | None  # 통과한 채점(core.JudgeRun) — 고객용 건수는 이것에서. 보류면 None
    result: object  # 생성 결과(generator.GenerationResult) — 사다리가 바꾼 칸까지
    record: dict = field(default_factory=dict)


Judge = Callable[[object, float | None], Awaitable[object]]  # (생성 결과, 마감 시각) → core.JudgeRun


async def run_ladder(session, judge: Judge, *, deadline: float | None, clock: Callable[[], float] = time.monotonic,
                     record: dict | None = None) -> LadderOutcome:
    """사다리를 돈다. record 를 주면 그 dict 에 바로 쓴다 — 바깥 제한이 도중에 끊어도 그때까지의 채점 기록이 남는다."""
    rec = record if record is not None else {}
    rec.update({"rounds": [], "regenerated": {}, "templated": [], "ever_blocked": False})
    rounds: list[dict] = rec["rounds"]

    def left() -> float | None:
        return None if deadline is None else deadline - clock()

    def held(failure: str) -> LadderOutcome:
        kind = held_kind(failure)
        allowed = retry_allowed(kind, failure=failure, ever_blocked=rec["ever_blocked"])
        rec.update({"gate": "held", "held_kind": kind, "failure": failure, "retry_allowed": allowed})
        return LadderOutcome("held", kind, allowed, None, session.result, rec)

    async def judge_round(after: str | None):
        remaining = left()
        if remaining is not None and remaining < JUDGE_MIN_SECONDS:
            rounds.append({"round": len(rounds) + 1, "after": after, "skipped": "deadline"})
            return None
        t = clock()
        run = await judge(session.result, deadline)
        rounds.append({
            "round": len(rounds) + 1, "after": after, "gate": run.gate, "failure": run.failure,
            "flagged": list(run.flagged_slots), "unfixable": list(run.unfixable), "seconds": round(clock() - t, 3),
            "cost_usd": run.cost_usd(), "scorecard": run.scorecard, "record": run.record,
        })
        if run.gate == "blocked":
            rec["ever_blocked"] = True
        return run

    run = await judge_round(None)
    rungs = [RUNG_REGENERATE, RUNG_TEMPLATE_FLAGGED, RUNG_TEMPLATE_ALL]
    while True:
        if run is None:
            return held("deadline")
        if run.failure is not None or run.gate not in ("pass", "blocked"):
            return held(run.failure or "error")
        if run.gate == "pass":
            rec.update({"gate": "passed", "held_kind": None, "failure": None, "retry_allowed": False})
            return LadderOutcome("passed", None, False, run, session.result, rec)
        changed: list[str] = []
        after = None
        while rungs and not changed:
            after = rungs.pop(0)
            changed = await _climb(session, after, run.flagged_slots, left, rec)
        if not changed:
            return held("gate_failed")
        run = await judge_round(after)


async def _climb(session, rung: str, flagged: list[str], left, rec: dict) -> list[str]:
    """단 하나를 오른다. 바꾼 칸을 돌려준다(없으면 다음 단으로)."""
    if rung == RUNG_REGENERATE:
        remaining = left()
        targets = [s for s in flagged if session.has_ai_text(s)]
        if not targets:
            return []
        if remaining is not None and remaining < REGENERATE_MIN_SECONDS + JUDGE_MIN_SECONDS:
            rec["regenerate_skipped"] = "deadline"  # 정6 ⑥ — 규칙 문장 단으로
            return []
        seconds = None if remaining is None else remaining - JUDGE_MIN_SECONDS
        outcome = await session.judge_regenerate(targets, seconds=seconds)
        rec["regenerated"] = outcome
        return [s for s, o in outcome.items() if o != "skipped"]
    if rung == RUNG_TEMPLATE_FLAGGED:
        changed = session.to_template(flagged, "judge_flagged")
    else:
        changed = session.to_template(list(session.states), "judge_all")
    rec["templated"] = rec["templated"] + changed
    return changed


def summary(rec: dict) -> dict:
    """내부 기록의 한 줄 요약(서버 로그·계획 16-1) — 판정·회차·비용·시간·기준 버전·고객 입력 기반 단정형 추론 건수.
    채점 하나라도 비용을 모르면 None(추정하지 않는다)."""
    judged = [r for r in rec.get("rounds", []) if "record" in r]
    costs = [r.get("cost_usd") for r in judged]
    first = judged[0]["record"] if judged else {}
    cards = [r.get("scorecard") for r in judged if r.get("scorecard")]
    return {
        "gate": rec.get("gate"), "held_kind": rec.get("held_kind"), "failure": rec.get("failure"),
        "retry_allowed": rec.get("retry_allowed"), "rounds": len(judged),
        "regenerated": sorted(s for s, o in (rec.get("regenerated") or {}).items() if o != "skipped"),
        "templated": list(rec.get("templated") or []),
        "cost_usd": None if any(c is None for c in costs) else round(sum(costs), 6),
        "seconds": round(sum(r.get("seconds") or 0 for r in judged), 3),
        "criteria_version": first.get("criteria_version"), "criteria_fingerprint": first.get("criteria_fingerprint"),
        "material_check_version": first.get("material_check_version"),
        "client_input_assertive": (cards[-1].get("faithfulness") or {}).get("client_input_assertive") if cards else None,
    }
