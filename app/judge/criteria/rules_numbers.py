"""criteria/rules_numbers.py — 숫자 대조 규칙(설계 6.1 "숫자 대조 규칙", 6.3 값 묶기).

- 문장에 찍힌 자릿수로 데이터를 반올림해 같으면 일치: 0.7↔0.700 일치, 0.71↔0.700 위반.
- 백분율↔소수 변환 허용: 0.944 ↔ "94%" 일치, "95%" 위반. 이미 백분율인 데이터(33.5)도 "33.5%" 와 일치.
- 반올림 방식(올림·짝수 맞춤·이진 표현)의 차이로 갈리지 않게 "찍힌 수 ± 마지막 자리 반 칸" 안이면 일치로 본다.
- 표기 자릿수 자체는 보지 않는다(설계 6.5). 천 단위 쉼표, 부호(+, −, U+2212), %p 를 읽는다.
기준 묶음에 둔다 — 대조 규칙이 바뀌면 판정이 바뀐다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_NUM = re.compile(r"^([+\-−]?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%p|%|퍼센트포인트|퍼센트)?$")
EPS = Decimal("1e-9")


@dataclass(frozen=True)
class Printed:
    value: Decimal
    decimals: int
    percent: bool  # % 또는 %p


def parse_printed(text: str) -> Printed | None:
    """찍힌 숫자 글자 → Printed. 숫자 꼴이 아니면 None."""
    m = _NUM.match(text.strip())
    if not m:
        return None
    sign, whole, frac, unit = m.groups()
    try:
        value = Decimal(whole.replace(",", "") + ("." + frac if frac else ""))
    except InvalidOperation:
        return None
    if sign in ("-", "−"):
        value = -value
    return Printed(value, len(frac or ""), bool(unit))


def _close(printed: Printed, data: Decimal) -> bool:
    half = Decimal(5) * Decimal(10) ** (-(printed.decimals + 1))
    return abs(data - printed.value) <= half + EPS


def matches(text: str, data: float | int) -> bool | None:
    """찍힌 숫자가 데이터 값과 맞나. 찍힌 글이 숫자가 아니면 None."""
    p = parse_printed(text)
    if p is None:
        return None
    d = Decimal(str(data))
    if p.percent:
        return _close(p, d * 100) or _close(p, d)
    return _close(p, d)


def same_quantity(a: str, b: str) -> bool | None:
    """같은 칸의 두 찍힌 숫자가 같은 양으로 읽히나(일관성 값 묶기). 두 구간(±반 칸)이 실제로 겹치면 같다 — 끝점만 맞닿으면
    (54 와 55, 0.5 와 0.6) 다른 양이다. 숫자가 아니면 None.
    한쪽만 백분율이면 소수 쪽을 ×100 해 본다."""
    pa, pb = parse_printed(a), parse_printed(b)
    if pa is None or pb is None:
        return None

    def interval(p: Printed, scale: int) -> tuple[Decimal, Decimal]:
        half = Decimal(5) * Decimal(10) ** (-(p.decimals + 1))
        return (p.value - half) * scale, (p.value + half) * scale

    scale_a = 100 if (pb.percent and not pa.percent) else 1
    scale_b = 100 if (pa.percent and not pb.percent) else 1
    lo_a, hi_a = interval(pa, scale_a)
    lo_b, hi_b = interval(pb, scale_b)
    return lo_a < hi_b - EPS and lo_b < hi_a - EPS
