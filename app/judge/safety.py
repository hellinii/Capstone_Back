"""app/judge/safety.py — 매 실행 안전장치(설계 7.3, 계획 12-2, Q4 c).

모듈 안에서 이미 한 검사(인용이 그 조각에 있나, 조회한 칸이 있나)를 합산 전에 한 번 더 본다 — 모듈 코드가 바뀌어도 이 문을
지나야 한다. 문제가 있으면 그 판정을 쓰지 않고 보류(safety)로 간다.
- 숫자 위반: 인용 구절이 그 조각 글에 있다. 발급 규칙(2026-11-05, aggregate.issuance)이 쓰는 사실 위반은 숫자 위반뿐이라(그 칸을
  규칙 문장으로 바꾼다) 그것만 본다 — 기록만 하는 사실 위반(이름·비교·정도·단정형 추론)의 인용이 어긋났다고 발급을 보류하지 않는다
  (필수 내용·표기 축을 뺐을 때와 같은 원칙).
- 모순: 두 인용 모두(값 불일치는 찍힌 숫자가) 그 조각 글에 있다.
- 필수 내용 '서술됨': 인용 구절이 그 조각 글에 있고 그 조각이 표가 아니다.
"""
from __future__ import annotations

from app.judge.aggregate import REPAIR_KINDS
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
        if v.kind in REPAIR_KINDS and not has(v.piece_id, v.quote):
            out.append(f"faithfulness:{v.claim_id}")
    for c in consistency.contradictions:
        if not all(has(pid, q) for pid, q in zip(c.pieces, c.quotes)):
            out.append(f"consistency:{c.type}")
    for i in (coverage.items if coverage_gates else []):
        if i.described and (not has(i.piece_id, i.quote) or by_id[i.piece_id].source == "table"):
            out.append(f"coverage:{i.id}")
    return out
