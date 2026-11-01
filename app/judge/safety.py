"""app/judge/safety.py — 매 실행 안전장치(설계 7.3, 계획 12-2, Q4 c).

모듈 안에서 이미 한 검사(인용이 그 조각에 있나, 조회한 칸이 있나)를 합산 전에 한 번 더 본다 — 모듈 코드가 바뀌어도 이 문을
지나야 한다. 문제가 있으면 그 판정을 쓰지 않고 보류(safety)로 간다.
- 사실 위반·단정형 추론: 인용 구절이 그 조각 글에 있다.
- 모순: 두 인용 모두(값 불일치는 찍힌 숫자가) 그 조각 글에 있다.
- 필수 내용 '서술됨': 인용 구절이 그 조각 글에 있고 그 조각이 표가 아니다.
"""
from __future__ import annotations

from app.judge.reading import Piece


def _norm(t: str) -> str:
    return " ".join(t.split())


def problems(pieces: list[Piece], faith, consistency, coverage, *, coverage_gates: bool = True) -> list[str]:
    """coverage_gates=False 면 필수 내용 인용 확인은 하지 않는다 — 그 축을 고객 묶음에서 뺐을 때(bundle.PUBLISHED_AXES, 2026-11-01)."""
    by_id = {p.id: p for p in pieces}
    out: list[str] = []

    def has(pid: str, quote: str) -> bool:
        p = by_id.get(pid)
        return p is not None and bool(quote.strip()) and _norm(quote) in _norm(p.text)

    for v in faith.violations():
        if not has(v.piece_id, v.quote):
            out.append(f"faithfulness:{v.claim_id}")
    for c in consistency.contradictions:
        if not all(has(pid, q) for pid, q in zip(c.pieces, c.quotes)):
            out.append(f"consistency:{c.type}")
    for i in (coverage.items if coverage_gates else []):
        if i.described and (not has(i.piece_id, i.quote) or by_id[i.piece_id].source == "table"):
            out.append(f"coverage:{i.id}")
    return out
