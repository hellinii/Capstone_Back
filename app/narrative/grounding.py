"""app/narrative/grounding.py — 환각 방어(grounding) 순수 로직

LLM 출력의 모든 숫자가 fact_sheet/파생값 화이트리스트 안에 있는지 검증한다. 위반 시
호출부(narrator)가 규칙 폴백으로 대체한다. 숫자 화이트리스트 구성·검증·문자열 수집을 담당.

상호작용
- 의존(import): re, typing.Any, app.core.schemas(FactSheet, GroundingInfo)
- 사용처: app.narrative.narrator(generate_narrative 의 화이트리스트/검증), tests
"""
import re
from typing import Any

from app.narrative.schemas import FactSheet, GroundingInfo


# 문맥상 '수치'가 아닌 고정 토큰(표준 표기·절 번호·오류 유형 서수)은 검증 전에 제거한다.
# 0/1/100/연도 같은 범용 숫자는 더 이상 무조건 면제하지 않는다 — fact_sheet 근거로만 허용(D3a).
_CTX_TOKEN_RE = re.compile(
    r"ISO/?IEC\s+TS\s*4213(?:\s*:\s*2022)?|4213\s*:\s*2022|\d+\s*절|제\s*\d+\s*종"
)

# 숫자 토큰: 앞 글자가 영숫자면(F1·P99 등 식별자) 숫자로 보지 않는다(선행 lookbehind).
# 부호(음수 정답값 -0.1 등)·천단위 콤마·백분율(%)을 함께 포착한다(D7[2]).
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[+-]?\d+(?:[.,]\d+)?%?")


def _canon(num_str: str):
    """숫자 토큰을 정규 표기로: %·콤마 제거 후 float → 'g' 포맷(불필요한 0 제거)."""
    s = num_str.replace("%", "").replace(",", "").strip()
    try:
        f = float(s)
    except ValueError:
        return None
    return f"{f:g}"


def _add(wl: set, v: Any) -> None:
    """허용 숫자 v 를 여러 표기(소수 1~4자리, 0~1 범위면 백분율)로 화이트리스트에 추가."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return
    wl.add(f"{f:g}")
    for k in (1, 2, 3, 4):
        wl.add(f"{round(f, k):g}")
    if 0.0 <= f <= 1.0:
        for k in (0, 1, 2):
            wl.add(f"{round(f * 100, k):g}")


def build_number_whitelist(fs: FactSheet, benchmark_refs: list[dict], derived: dict) -> set:
    """출력 검증용 허용 숫자 집합 (fact_sheet + 파생값의 다중 표기)."""
    wl: set = set()

    _add(wl, fs.n_samples)
    _add(wl, fs.dropped_rows)
    _add(wl, fs.score)

    for m in fs.metrics:
        _add(wl, m.value)
        if m.threshold is not None:
            _add(wl, m.threshold)

    for pc in fs.per_class:
        for val in (pc.precision, pc.recall, pc.f1, pc.support):
            _add(wl, val)
        if pc.label and pc.label.replace(".", "").isdigit():
            _add(wl, pc.label)

    if fs.confusion:
        for row in fs.confusion.matrix:
            for cell in row:
                _add(wl, cell)
        for label in fs.confusion.labels:
            if label and label.replace(".", "").isdigit():
                _add(wl, label)

    if fs.distribution:
        for k, v in fs.distribution.class_distribution.items():
            _add(wl, v)
            if k and k.replace(".", "").isdigit():
                _add(wl, k)
        if fs.distribution.imbalance_ratio is not None:
            _add(wl, fs.distribution.imbalance_ratio)

    if fs.latency and fs.latency.available:
        for val in (fs.latency.mean, fs.latency.p50, fs.latency.p95, fs.latency.p99):
            _add(wl, val)

    for ref in benchmark_refs:
        _add(wl, ref.get("model_value"))
        _add(wl, ref.get("ref_low"))
        _add(wl, ref.get("ref_high"))

    # 파생값
    conf = derived.get("confusion", {})
    for val in conf.values():
        _add(wl, val)
    dist = derived.get("distribution", {})
    if "total" in dist:
        _add(wl, dist["total"])
    for pct in dist.get("percentages", {}).values():
        _add(wl, pct)
    for val in derived.get("counts", {}).values():
        _add(wl, val)

    return wl


def verify_grounding(texts: list[str], whitelist: set) -> GroundingInfo:
    """텍스트들에서 숫자 토큰을 추출해 화이트리스트와 대조. 위반 = 환각 수치."""
    violations: list[str] = []
    checked = 0
    for t in texts:
        if not t:
            continue
        # 표준 표기·절/서수 등 문맥 토큰을 먼저 제거한 뒤 숫자만 검증(오탐 방지, D3a).
        stripped = _CTX_TOKEN_RE.sub(" ", t)
        for tok in _NUMBER_RE.findall(stripped):
            c = _canon(tok)
            if c is None:
                continue
            checked += 1
            if c not in whitelist:
                violations.append(tok)
    uniq = sorted(set(violations))
    return GroundingInfo(checked=checked, violations=uniq, passed=len(uniq) == 0)


def _collect_strings(obj: Any, out: list) -> None:
    """LLM 산출물(dict/list/str)의 모든 문자열 리프를 재귀 수집(grounding 전수 검사용)."""
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_strings(item, out)


def _collect_grounding_texts(data: dict) -> list:
    """LLM 응답 전체에서 검증 대상 문자열을 전수 수집(수동 나열 제거 — D3b).

    conclusion.verdict 만 제외한다(서버가 fact_sheet 값으로 강제 대체하므로 미노출 →
    불필요한 폴백 전환 방지). 이후 스키마에 문자열 필드가 추가돼도 자동 포함된다.
    """
    grounded = dict(data)
    concl = grounded.get("conclusion")
    if isinstance(concl, dict):
        grounded["conclusion"] = {k: v for k, v in concl.items() if k != "verdict"}
    texts: list = []
    _collect_strings(grounded, texts)
    return texts
