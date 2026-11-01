"""app/judge/validation/agreement.py — 일치도 계수(계획 17-7, 설계 7.1 "ICC 또는 Krippendorff α", 사람 일치도 참고 값).

- krippendorff_alpha: 평가자 × 단위 표(빈칸 None), 명목·구간 척도. 일치 행렬(coincidence matrix) 식 — Krippendorff(2011)
  "Computing Krippendorff's Alpha-Reliability".
- cohen_kappa: 두 평가자, 명목. κ = (관측 일치 − 우연 일치) / (1 − 우연 일치), 우연 일치는 평가자별 주변 비율의 곱.
- gwet_ac1: 두 평가자, 명목. 우연 일치 = Σ π_k(1 − π_k) / (q − 1), π_k 는 두 평가자 평균 비율, q 는 범주 수.
계산할 수 없으면(변이 없음·범주 1개·짝 단위 없음) None.

상호작용
- 의존(import): 없음(표준 라이브러리)
- 사용처: scripts/judge_validate.py(검증 실행기, 참고 값)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations

LEVELS = ("nominal", "interval")


def _delta2(c, k, level: str) -> float:
    if level == "nominal":
        return 0.0 if c == k else 1.0
    return float(c - k) ** 2


def krippendorff_alpha(data: list[list], level: str = "nominal") -> float | None:
    """data = 평가자마다 한 줄(단위 순서 같게), 빈칸 None. 값이 둘 이상인 단위만 센다."""
    if level not in LEVELS:
        raise ValueError(f"모르는 척도: {level}")
    if not data:
        return None
    width = {len(row) for row in data}
    if len(width) != 1:
        raise ValueError("평가자마다 단위 수가 같아야 한다")
    coincidence: dict[tuple, float] = defaultdict(float)
    for unit in zip(*data):
        values = [v for v in unit if v is not None]
        m = len(values)
        if m < 2:
            continue
        for a, b in permutations(values, 2):
            coincidence[(a, b)] += 1.0 / (m - 1)
    if not coincidence:
        return None
    n_c: dict = defaultdict(float)
    for (a, _), w in coincidence.items():
        n_c[a] += w
    n = sum(n_c.values())
    observed = sum(w * _delta2(a, b, level) for (a, b), w in coincidence.items())
    expected = sum(n_c[a] * n_c[b] * _delta2(a, b, level) for a in n_c for b in n_c)
    if expected == 0:
        return None
    return 1.0 - (n - 1) * observed / expected


def _pairs(a: list, b: list) -> list[tuple]:
    if len(a) != len(b):
        raise ValueError("두 평가자의 항목 수가 다르다")
    return [(x, y) for x, y in zip(a, b) if x is not None and y is not None]


def cohen_kappa(a: list, b: list) -> float | None:
    pairs = _pairs(a, b)
    if not pairs:
        return None
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def gwet_ac1(a: list, b: list, categories: list | None = None) -> float | None:
    """categories 를 주면 그 범주 수로 q 를 센다(관측되지 않은 범주 포함). 없으면 관측된 범주."""
    pairs = _pairs(a, b)
    if not pairs:
        return None
    n = len(pairs)
    cats = list(categories) if categories is not None else sorted({v for p in pairs for v in p}, key=str)
    q = len(cats)
    if q < 2:
        return None
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pi = {k: (ca[k] + cb[k]) / (2 * n) for k in cats}
    pe = sum(p * (1 - p) for p in pi.values()) / (q - 1)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)
