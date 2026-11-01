"""app/judge/aggregate.py — 결과 합산과 게이트(설계 5.4, 계획 12-1). 순수 함수 — LLM 을 부르지 않고 새 정보를 만들지 않는다.

- 축별 점수표: 분모를 늘 함께("모순 0건 / 검사 23건"). 종합 점수는 만들지 않는다.
- 게이트: 사실 위반(데이터와 어긋난 주장·단정형 추론) ≥ 1 또는 모순 ≥ 1 → 발급 불가(blocked). 필수 내용·표기는 게이트가 아니다.
- 걸린 칸(사다리가 다시 만들 칸, 정6 ⑤): 사실 위반은 그 주장의 칸. 모순은 서술 칸(AI·규칙 문장) 쪽 — 둘 다 서술 칸이면 칸 이름
  순서(7.1 → 7.2 → 7.3 → 8.1 → 8.2 → 8.3 → 9)의 뒤쪽, 한쪽이 표·고정 문구면 서술 칸. 둘 다 표·고정 문구면 다시 만들 칸이 없다
  (unfixable 에 적는다 — 사다리는 전부 규칙 문장 단계까지 간다).
"""
from __future__ import annotations

SLOT_ORDER = ("7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "9")
NARRATIVE_SOURCES = ("ai", "template")


def flagged_slots(faith, consistency) -> tuple[list[str], list[dict]]:
    slots: set[str] = set()
    unfixable: list[dict] = []
    for v in faith.violations():
        if v.slot in SLOT_ORDER:
            slots.add(v.slot)
    for c in consistency.contradictions:
        narrative = [s for s, src in zip(c.slots, c.sources) if src in NARRATIVE_SOURCES and s in SLOT_ORDER]
        if narrative:
            slots.add(max(narrative, key=SLOT_ORDER.index))
        else:
            unfixable.append({"type": c.type, "pieces": c.pieces})
    return sorted(slots, key=SLOT_ORDER.index), unfixable


def gate(faith, consistency) -> str:
    return "blocked" if faith.violations() or consistency.contradictions else "pass"


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
