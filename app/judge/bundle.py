"""app/judge/bundle.py — 고객용 심판 묶음(계획 14-2, 계약 tests/contract/judge_bundle_ed4.json).

판 4 서술 작업 결과의 `judge` 칸을 만든다. 실을 축은 여기 한 곳(PUBLISHED_AXES)이 정한다 — 검증(7.7)에서 축을 빼면 이 값만
바꾼다. 근거 인용·사유·주장 내용·재생성 횟수·토큰·비용·시간·기준 지문은 싣지 않는다(계약 never_in_bundle).

상호작용
- 의존(import): app.judge.config·criteria
- 사용처: app.narrative.jobs(판 4 갈래), tests
"""
from __future__ import annotations

from app.judge import config, criteria

PUBLISHED_AXES = ("faithfulness", "consistency", "coverage", "readability")


def _head(gate: str, held_kind: str | None, retry_allowed: bool) -> dict:
    return {"gate": gate, "held_kind": held_kind, "retry_allowed": retry_allowed,
            "criteria_version": criteria.version(), "model": config.judge_model(), "axes": list(PUBLISHED_AXES)}


def held(kind: str, *, retry_allowed: bool) -> dict:
    return {**_head("held", kind, retry_allowed), "counts": None}


def passed(scorecard: dict) -> dict:
    f, c, v, r = (scorecard[a] for a in ("faithfulness", "consistency", "coverage", "readability"))
    counts = {
        "faithfulness": {"checked": f["checked"], "matched": f["matched"]},
        "consistency": {"checked": c["checked"], "contradictions": c["contradictions"]},
        "coverage": {"applicable": v["applicable"], "described": v["described"], "not_applicable": v["not_applicable"],
                     "items": [{"id": i["id"], "described": i["described"], "place": i["place"]} for i in v["items"]]},
        "readability": {"rules": r["rules"], "violations": r["violations"]},
    }
    return {**_head("passed", None, False), "counts": {a: counts[a] for a in PUBLISHED_AXES}}
