"""app/judge/aggregate.py — 결과 합산과 게이트(설계 5.4, 계획 12-1). 순수 함수 — LLM 을 부르지 않고 새 정보를 만들지 않는다.

- 축별 점수표: 분모를 늘 함께("모순 0건 / 검사 23건"). 종합 점수는 만들지 않는다.
- 게이트(gate): 사실 위반(데이터와 어긋난 주장·단정형 추론) ≥ 1 또는 모순 ≥ 1 → blocked. 기준 1.25 판정자의 판정이다 — 검증
  도구(app.judge.validation, scripts/judge_validate·judge_eval)가 이 값을 잰다. 발급은 이 값이 아니라 아래 발급 규칙이 정한다.
- 발급 규칙(issuance, 2026-11-05 평가 단계 뒤 민서 결정): 모순 ≥ 1 → blocked(사다리). 사실 일치는 발급을 막지 않는다 — 숫자
  주장 위반이 난 칸만 고칠 칸(repair_slots)으로 넘겨 규칙 문장으로 바꾸고(ladder), 나머지 사실 위반은 기록에만 남는다. 평가 단계에서
  사실 일치 게이트가 오류 없는 원본·변형 30회를 모두 막았고(오판만으로 17회), 일관성은 결함 9/9·오판 0 이었다
  (JUDGE_ROUND/평가/05_결과/평가_실험기록.md). 판정 기준(기준 묶음·지문)은 그대로다 — 이 파일은 기준 묶음 밖이다. 그래서 검증 기록
  잠금(criteria_lock)이 발급 규칙(issuance·REPAIR_KINDS)을 지키지 않는다 — 바꿀 때는 결정 기록을 먼저 남긴다.
  알려진 위험: 숫자 칸을 규칙 문장으로 바꾼 뒤 다시 채점하지 않으므로, 바뀐 숫자가 다른 칸에 남은(추출이 숫자 주장으로 잡지 못한)
  같은 값과 어긋날 수 있다. 고객 부록의 일관성 건수는 바꾸기 전 글을 잰 값이다.
- 걸린 칸(flagged_slots — 기록·검증용, 정6 ⑤): 사실 위반은 그 주장의 칸. 사다리가 다시 만들 칸은 blocking_slots(모순만).
  모순은 서술 칸(AI·규칙 문장) 쪽 — 둘 다 서술 칸이면 칸 이름 순서(7.1 → 7.2 → 7.3 → 8.1 → 8.2 → 8.3 → 9)의 뒤쪽, 한쪽이
  표·고정 문구면 서술 칸. 둘 다 표·고정 문구면 다시 만들 칸이 없다(unfixable 에 적는다 — 사다리는 전부 규칙 문장 단계까지 간다).
"""
from __future__ import annotations

SLOT_ORDER = ("7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "9")
NARRATIVE_SOURCES = ("ai", "template")


# 발급을 막지 않고 칸을 고치는 사실 위반의 주장 종류 — 숫자 주장만(규칙 문장은 코드가 데이터로 만들어 숫자가 틀릴 수 없다).
REPAIR_KINDS = ("number",)


def _ordered(slots: set[str]) -> list[str]:
    return sorted(slots, key=SLOT_ORDER.index)


def blocking_slots(consistency) -> tuple[list[str], list[dict]]:
    """발급을 막는 모순의 칸(사다리가 다시 만들 칸)과 고칠 서술 칸이 없는 모순(unfixable)."""
    slots: set[str] = set()
    unfixable: list[dict] = []
    for c in consistency.contradictions:
        narrative = [s for s, src in zip(c.slots, c.sources) if src in NARRATIVE_SOURCES and s in SLOT_ORDER]
        if narrative:
            slots.add(max(narrative, key=SLOT_ORDER.index))
        else:
            unfixable.append({"type": c.type, "pieces": c.pieces})
    return _ordered(slots), unfixable


def repair_slots(faith) -> list[str]:
    """숫자 주장 위반이 난 서술 칸 — 발급은 막지 않고 규칙 문장으로 바꿀 칸. 숫자 주장의 위반은 늘 violation 이다
    (inference_violation 은 추론 주장에만 난다)."""
    return _ordered({v.slot for v in faith.violations() if v.kind in REPAIR_KINDS and v.slot in SLOT_ORDER})


def flagged_slots(faith, consistency) -> tuple[list[str], list[dict]]:
    """판정자 게이트(gate)에 걸린 칸 전부 — 기록·검증용(사실 위반 칸 + 모순 칸)."""
    slots, unfixable = blocking_slots(consistency)
    faith_slots = {v.slot for v in faith.violations() if v.slot in SLOT_ORDER}
    return _ordered(set(slots) | faith_slots), unfixable


def gate(faith, consistency) -> str:
    """기준 1.25 판정자의 판정(검증 도구가 재는 값). 발급은 issuance 가 정한다."""
    return "blocked" if faith.violations() or consistency.contradictions else "pass"


def issuance(consistency) -> str:
    """발급 규칙 — 판정 지점은 여기 하나다. 모순이 있으면 blocked(사다리), 없으면 pass(사실 위반은 막지 않는다)."""
    return "blocked" if consistency.contradictions else "pass"


def scorecard(faith, consistency, coverage, readability) -> dict:
    f, c = faith.counts, consistency.counts
    return {
        "faithfulness": {"checked": f["checked"], "matched": f["matched"], "violations": f["violations"],
                         "inference_ok": f["inference_ok"], "inference_violations": f["inference_violation"],
                         "unverifiable": f["unverifiable"], "target_unclear": f["target_unclear"],
                         "client_input_assertive": f["client_input_assertive"],
                         # 1.1: 추론으로 뽑혔지만 뜻풀이라 뺀 수, 추론에서 데이터 대조로 옮겨 checked 에 든 수(1.0 점수와 바로 비교하지 않게)
                         "not_claim": f.get("not_claim", 0), "from_inference": f.get("from_inference", 0),
                         # 1.5 재확인: 다시 본 수, 판정이 바뀐 수, 재확인 호출 실패
                         "rechecked": f.get("rechecked", 0), "recheck_changed": f.get("recheck_changed", 0),
                         "recheck_failed": f.get("recheck_failed", 0)},
        "consistency": {"checked": c["checked"], "contradictions": c["contradictions"], "by_type": c["by_type"]},
        "coverage": {"applicable": coverage.counts["applicable"], "described": coverage.counts["described"],
                     "not_applicable": coverage.counts["not_applicable"],
                     "items": [{"id": i.id, "described": i.described, "place": i.place or None} for i in coverage.items]},
        "readability": {"rules": readability.counts["rules"], "violations": readability.counts["violations"],
                        "llm_flags": readability.counts.get("llm_flags", 0)},
    }
