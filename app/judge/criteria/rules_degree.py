"""criteria/rules_degree.py — 정도 표현 허용 판정(설계 6.1 정도 표현 기준, Q4 a — 글 규칙은 degree.json).

LLM 은 극성(positive·negative·neutral)과 대상 지표만 준다. 목표 대비 위치는 여기서 코드로 본다.
결과: "ok"(허용) · "violation"(위반) · "target_unclear"(대상 불명 — 분모에서 뺀다).
"""
from __future__ import annotations

import json
from pathlib import Path

_RULES = json.loads((Path(__file__).with_name("degree.json")).read_text(encoding="utf-8"))
STRONG_POSITIVE = tuple(_RULES["strong_positive"])


def is_strong(quote: str) -> bool:
    return any(w in quote for w in STRONG_POSITIVE)


def _ref_quality(display_name: str, refs: list[dict]) -> str | None:
    r = next((r for r in refs if r.get("metric") == display_name), None)
    return None if r is None else r.get("quality")


def _judged(m) -> bool:
    return m.threshold is not None and m.value is not None and m.status in ("pass", "fail")


MARGIN_WORDS = tuple(_RULES.get("margin_words", []))


def _margin_verdict(polarity: str, metric_id: str, metrics) -> str | None:
    """판정 여유 표현(1.7) — near_target 표시로. 표시가 없으면 None(보통 판정으로)."""
    if metric_id:
        targets = [m for m in metrics if m.metric_id == metric_id]
    else:
        targets = [m for m in metrics if m.is_core and _judged(m)]
    flags = [getattr(m, "near_target", None) for m in targets]
    if not flags or any(f is None for f in flags):
        return None
    near = any(flags)
    return "ok" if (near if polarity == "negative" else not near) else "violation"


def degree_verdict(polarity: str, metric_id: str, quote: str, metrics, refs: list[dict]) -> str:
    """polarity·대상 지표(metric_id, 빈 글이면 성능 전체 = 판정 대상 핵심 지표) → 허용 판정."""
    if polarity == "neutral":
        return "ok"
    if polarity not in ("positive", "negative"):
        return "target_unclear"
    if any(w in quote for w in MARGIN_WORDS):
        margin = _margin_verdict(polarity, metric_id, metrics)
        if margin is not None:
            return margin
    strong = polarity == "positive" and is_strong(quote)
    if metric_id:
        m = next((m for m in metrics if m.metric_id == metric_id), None)
        if m is None or m.value is None:
            return "target_unclear"
        quality = _ref_quality(m.display_name, refs)
        if _judged(m):
            meets = m.status == "pass"
            if polarity == "negative":
                return "ok" if not meets else "violation"
            if not meets:
                return "violation"
            if strong and quality is not None:
                return "ok" if quality == "better" else "violation"
            return "ok"
        # 목표 없는 지표 — 참조 범위로만
        if quality is None:
            return "target_unclear"
        if polarity == "positive":
            return "ok" if quality == "better" else "violation"
        return "ok" if quality == "worse" else "violation"
    core = [m for m in metrics if m.is_core and _judged(m)]
    if not core:
        return "target_unclear"
    all_pass = all(m.status == "pass" for m in core)
    if polarity == "negative":
        return "violation" if all_pass else "ok"
    if not all_pass:
        return "violation"
    if strong:
        qualities = [_ref_quality(m.display_name, refs) for m in core]
        known = [q for q in qualities if q is not None]
        if known and not all(q == "better" for q in known):
            return "violation"
    return "ok"
