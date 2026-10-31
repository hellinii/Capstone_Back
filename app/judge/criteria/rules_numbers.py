"""criteria/rules_numbers.py — 숫자 대조 규칙(설계 6.1 "숫자 대조 규칙", 6.3 값 묶기).

- 문장에 찍힌 자릿수로 데이터를 반올림해 같으면 일치: 0.7↔0.700 일치, 0.71↔0.700 위반.
- 백분율↔소수 변환 허용: 0.944 ↔ "94%" 일치, "95%" 위반. 이미 백분율인 데이터(33.5)도 "33.5%" 와 일치.
- 반올림 방식(올림·짝수 맞춤·이진 표현)의 차이로 갈리지 않게 "찍힌 수 ± 마지막 자리 반 칸" 안이면 일치로 본다.
- 표기 자릿수 자체는 보지 않는다(설계 6.5). 천 단위 쉼표, 부호(+, −, U+2212), %p 를 읽는다.
- 부호 없이 쓴 숫자는 크기로도 본다 — "Recall 이 0.066 높다"처럼 방향을 말로 쓴 차이가 음수 데이터(−0.066)와 맞는다(J4 파일럿).
  부호를 붙여 쓴 숫자(+13, −0.275)는 부호까지 맞아야 한다. 방향이 맞는지는 비교 주장이 따로 본다.
- 분수 말('3분의 1')은 그 몫(1 ÷ 3)으로 읽고 셋째 자리 반 칸(FRACTION_TOLERANCE) 안이면 일치(1.1, V2 1차 — '3분의 1' 의 3 을 읽어
  0.333 과 어긋났다).
- 문장에서 숫자를 찾을 때 이름에 붙은 숫자('클래스 3'·'레이블 14'·'Class 9')는 수량이 아니다(NAME_BEFORE_NUMBER, 1.1 — '클래스 3
  으로 예측된 표본이 … 많이' 의 3 을 예측 건수와 대조했다).
기준 묶음에 둔다 — 대조 규칙이 바뀌면 판정이 바뀐다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_NUM = re.compile(r"^([+\-−]?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%p|%|퍼센트포인트|퍼센트)?$")
EPS = Decimal("1e-9")
_FRACTION = re.compile(r"(\d+)\s*분의\s*(\d+)")
FRACTION_TOLERANCE = 0.0005 + 1e-9
# 문장에서 수량 숫자를 찾는 꼴(낱말에 붙은 숫자 'F1'·'M4'·'Class9' 는 숫자가 아니다 — J4 AI 파일럿)
_NUMBER_IN_TEXT = re.compile(r"(?<![A-Za-z0-9.])[+\-−]?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])\s*(?:%p|%)?")
NAME_BEFORE_NUMBER = ("클래스", "레이블", "라벨", "class", "label")
COUNT_UNITS = ("개", "종", "가지")


@dataclass(frozen=True)
class Printed:
    value: Decimal
    decimals: int
    percent: bool  # % 또는 %p
    signed: bool = False  # 부호를 붙여 썼다


def parse_printed(text: str) -> Printed | None:
    """찍힌 숫자 글자 → Printed. 숫자 꼴이 아니면 None."""
    m = _NUM.match(text.strip())
    if not m:
        return None
    sign, whole, frac, unit = m.groups()
    signed = bool(sign)
    try:
        value = Decimal(whole.replace(",", "") + ("." + frac if frac else ""))
    except InvalidOperation:
        return None
    if sign in ("-", "−"):
        value = -value
    return Printed(value, len(frac or ""), bool(unit), signed)


def fraction_value(text: str) -> float | None:
    """'N분의 M' 꼴이면 M ÷ N, 아니면 None."""
    m = _FRACTION.search(text)
    if not m or int(m.group(1)) == 0:
        return None
    return int(m.group(2)) / int(m.group(1))


def first_quantity(text: str) -> str | None:
    """문장에서 대조할 첫 수량 글자 — 분수 말이 있으면 그것, 아니면 이름에 붙지 않은 첫 숫자. 없으면 None."""
    m = _FRACTION.search(text)
    if m:
        return m.group(0)
    for m in _NUMBER_IN_TEXT.finditer(text):
        before = text[: m.start()].rstrip().lower()
        after = text[m.end():].lstrip()
        counted = after.startswith(COUNT_UNITS)  # '레이블 14개' 는 수량이다(1.1 검토)
        if any(before.endswith(w) for w in NAME_BEFORE_NUMBER) and not counted:
            continue
        if parse_printed(m.group(0).strip()):
            return m.group(0).strip()
    return None


def _close(printed: Printed, data: Decimal) -> bool:
    half = Decimal(5) * Decimal(10) ** (-(printed.decimals + 1))
    return abs(data - printed.value) <= half + EPS


def matches(text: str, data: float | int) -> bool | None:
    """찍힌 숫자가 데이터 값과 맞나. 찍힌 글이 숫자가 아니면 None."""
    frac = fraction_value(text)
    if frac is not None:
        return abs(float(data) - frac) <= FRACTION_TOLERANCE
    p = parse_printed(text)
    if p is None:
        return None
    d = Decimal(str(data))
    values = [d] if p.signed else [d, abs(d)]
    return any(_close(p, v * 100) or _close(p, v) if p.percent else _close(p, v) for v in values)


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

    if pa.signed != pb.signed:  # 한쪽만 부호 — 말로 방향을 쓴 크기(16건 적게)와 표의 부호 붙은 차이(−16)
        pa, pb = Printed(abs(pa.value), pa.decimals, pa.percent, False), Printed(abs(pb.value), pb.decimals, pb.percent, False)
    scale_a = 100 if (pb.percent and not pa.percent) else 1
    scale_b = 100 if (pa.percent and not pb.percent) else 1
    lo_a, hi_a = interval(pa, scale_a)
    lo_b, hi_b = interval(pb, scale_b)
    return lo_a < hi_b - EPS and lo_b < hi_a - EPS
