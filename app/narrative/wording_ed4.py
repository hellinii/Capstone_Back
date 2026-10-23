"""app/narrative/wording_ed4.py — 판 4 숫자 표기(순수) (판 4 라운드 S2 ④, 계획 2.7).

판 4 성적서의 표·소견 숫자는 셋째 자리이고 끝 0 을 유지한다(0.950, 1.000). 판정과 이어진 값은 판정과 어긋나 보이면
자릿수를 늘린다(여섯째 자리까지). 규칙은 두 저장소 공통 대조 표(tests/contract/number_wording_ed4.json — 같은 바이트 사본이
프론트 src/test/contract/ 에 있다)에 적혀 있고, 시험이 그 표의 모든 줄을 이 함수들로 통과시킨다. 프론트 같은 이름의 함수
(edition4Number.ts 의 ed4Plain 등)와 글자가 같아야 한다.

- 반올림은 이진 값 그대로의 절반에서 0에서 먼 쪽이다(JS toFixed 와 같음 — Decimal ROUND_HALF_UP). 파이썬 round()·'%.3f' 는
  절반에서 짝수 쪽이라 쓰지 않는다(0.0625 → 0.063).
- 판정(목표 충족·포함·기준 비교)은 반올림 전 값으로 하고, 차이는 10자리로 반올림해 뺄셈 끝자리 잡음(1e-10)을 걷는다
  (wording.meets_target·프론트 meetsTarget 과 같다). 이 모듈은 판정을 새로 하지 않는다 — 호출부가 받은 판정(meets)을 넘긴다.
- 사실 시트의 *_text 칸(숫자 꼴)은 plain·judged·signed_judged·ci_bound 의 결과다. 백분율(%)·%p·부호(+, U+2212)는 그릴 때와
  글을 쓸 때만(percent·percent_point·signed_count).
- 조사는 찍힌 글자로 고른다 — wording.josa 를 그대로 쓴다(숫자가 바뀌면 조사도 다시: 0.0046으로 → 0.005로).
판 3 표기(wording.num 의 넷째 자리·metric_num 의 끝 0 빼기)는 판 3 경로 그대로이고, 이 모듈은 판 4 요청에서만 쓴다.

상호작용
- 의존(import): app.narrative.wording(josa — 다시 내보냄)
- 사용처: app.narrative.fact_view(판 4 사실 보기), app.narrative.required_items(판 4 참조 범위 문장), tests
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

from app.narrative.wording import josa, with_josa  # noqa: F401 — 판 4 도 같은 조사 함수(공통 규칙)

# 판 4 에서 판정 쪽으로 늘리는 최대 자릿수 — 표·사실 시트 모두 여섯째 자리(판 3 사실 시트는 아홉째). 프론트와 같다.
MAX_DIGITS = 6
PLAIN_DIGITS = 3
MINUS = "−"  # U+2212 — 그리는 글(%p·건수 차이)에만. 숫자 꼴 칸은 ASCII '-'.


def _fixed(value, digits: int) -> str:
    """이진 값 그대로의 절반에서 0에서 먼 쪽으로 digits 자리(끝 0 유지). JS Number.prototype.toFixed 와 같다."""
    return format(Decimal(float(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP), "f")


def _unneg(text: str) -> str:
    """음의 0('-0.000')은 부호 없이."""
    return text[1:] if text.startswith("-") and set(text[1:]) <= set("0.") else text


def _diff(a: float, b: float) -> float:
    return round(float(a) - float(b), 10)


def _meets(value: float, threshold: float, higher_is_better: bool) -> bool:
    d = _diff(value, threshold)
    return d >= 0 if higher_is_better else d <= 0


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def ed4_plain(value) -> str:
    """판정과 무관한 값 — 셋째 자리, 끝 0 유지(0.950·1.000·1.672). 음수는 ASCII '-', 음의 0 은 부호 없이."""
    return _unneg(_fixed(value, PLAIN_DIGITS))


def ed4_judged(value: float, threshold: float, higher_is_better: bool, meets: bool) -> str:
    """판정 대상 값·기준과 비교되는 파생값 — 셋째 자리부터 여섯째 자리까지 늘려, 찍힌 값을 기준과 비교한 결과(높을수록
    좋음 ≥, 낮을수록 좋음 ≤)가 실제 판정(meets)과 같아지는 첫 표기. 여섯째 자리로도 어긋나면 판정 쪽으로 자른다."""
    value = float(value)
    for digits in range(PLAIN_DIGITS, MAX_DIGITS + 1):
        text = _fixed(value, digits)
        if _meets(float(text), threshold, higher_is_better) == meets:
            return _unneg(text)
    scale = 10**MAX_DIGITS
    toward_pass = higher_is_better == meets
    cut = (math.ceil(value * scale) if toward_pass else math.floor(value * scale)) / scale
    return _unneg(_fixed(cut, MAX_DIGITS))


def ed4_signed_judged(value: float, threshold: float, meets: bool) -> str:
    """부호 있는 차이를 크기로 기준과 비교하는 칸(Precision − Recall·분포 차) — 크기를 judged(높을수록 좋음)로 찍고 음수면
    '-' 를 붙인다. 크기가 0 으로 찍히면 부호 없음."""
    magnitude = ed4_judged(abs(float(value)), threshold, True, meets)
    return f"-{magnitude}" if float(value) < 0 and set(magnitude) - set("0.") else magnitude


def ed4_ci_bound(raw: float, target: float) -> str:
    """판정 대상 지표의 신뢰구간 경계 — 반올림 전 경계를 셋째 자리부터 여섯째 자리까지 늘려, 찍힌 경계와 목표의 대소(<·=·>)가
    반올림 전 경계와 목표의 대소와 같아지는 첫 표기. 여섯째 자리로도 안 되면 원값 쪽으로 자른다."""
    raw = float(raw)
    want = _sign(_diff(raw, target))
    for digits in range(PLAIN_DIGITS, MAX_DIGITS + 1):
        text = _fixed(raw, digits)
        if _sign(_diff(float(text), target)) == want:
            return _unneg(text)
    scale = 10**MAX_DIGITS
    cut = (math.floor(raw * scale) if want < 0 else math.ceil(raw * scale)) / scale
    return _unneg(_fixed(cut, MAX_DIGITS))


def _moved(text: str) -> tuple[str, str]:
    """소수 표기 글자의 소수점을 두 칸 옮긴다(숫자로 곱하지 않는다) — (부호, 옮긴 글자)."""
    sign = "-" if text.startswith("-") else ""
    whole, _, frac = text.lstrip("-").partition(".")
    digits = whole + frac.ljust(2, "0")
    point = len(whole) + 2
    head = digits[:point].lstrip("0") or "0"
    tail = digits[point:] or "0"
    return sign, f"{head}.{tail}"


def ed4_percent(fraction_text: str) -> str:
    """소수 표기 글자 → %(소수 자릿수 = 원래 − 2, 최소 1): 0.374 → 37.4%, 0.000 → 0.0%, 0.04996 → 4.996%."""
    sign, body = _moved(fraction_text)
    return f"{sign if set(body) - set('0.') else ''}{body}%"


def ed4_percent_point(fraction_text: str) -> str:
    """소수 표기 글자 → %p — 양수 '+', 음수 U+2212, 0 은 부호 없음(4.5 차이 열: −0.2%p)."""
    sign, body = _moved(fraction_text)
    if not set(body) - set("0."):
        return f"{body}%p"
    return f"{MINUS if sign else '+'}{body}%p"


def ed4_signed_count(n: int) -> str:
    """정수 차이 — 양수 '+', 음수 U+2212, 0 은 부호 없음(4.4 예측 편향: +1)."""
    return f"+{n}" if n > 0 else f"{MINUS}{-n}" if n < 0 else "0"


def ed4_given(value) -> str:
    """의뢰자가 준 결정 임계값·판정 기준값 — 늘이지도 끝 0 을 채우지도 않는다(0.5·0.1·1.5)."""
    return repr(float(value))


def ed4_ratio_percent(numerator: int, denominator: int) -> str:
    """건수 비율의 백분율 — plain(셋째 자리) 글자를 옮긴다(3/171 → 0.018 → 1.8%). 분모가 0 이면 0.0%."""
    return ed4_percent(ed4_plain(numerator / denominator if denominator else 0.0))


# ── 참조 범위 위치(4.6·5.4) ─────────────────────────────────────────────────────────

POSITION_KO = {"above": "참조 범위보다 높음", "within": "참조 범위 안", "below": "참조 범위보다 낮음"}
# 참조 범위 끝은 기준표 글자 그대로 둘째 자리(예시본 4.6 '0.80 ~ 0.92'·5.4 '참조 범위 0.80~0.92'). 고정 문구 23(셋째 자리)의
# 범위 밖이다(계획 정해진 것 6).
RANGE_DIGITS = 2


def ed4_range_bound(value: float) -> str:
    """참조 범위 끝 — 둘째 자리, 끝 0 유지(0.80)."""
    return _unneg(_fixed(value, RANGE_DIGITS))


def ed4_range_text(low: float, high: float) -> str:
    """참조 범위 '0.80~0.92'(서술·사실 보기의 꼴, 표는 그릴 때 '0.80 ~ 0.92')."""
    return f"{ed4_range_bound(low)}~{ed4_range_bound(high)}"


def _position(value: float, low: float, high: float) -> str:
    # baselines.benchmark_position 과 같은 규칙(범위 끝은 안).
    return "below" if value < low else "above" if value > high else "within"


def ed4_range_value_text(
    value: float, low: float, high: float, base: str | None = None,
    target: tuple[float, bool, bool] | None = None,
) -> str:
    """참조 범위 비교의 모델 값 표기(판 4) — 찍힌 숫자가 위치(반올림 전 값으로 한 below/within/above)와 같은 쪽으로 읽히게.

    판 3 wording.range_value_text 와 같은 생각을 판 4 표기로 한다: 평소 표기(base — 판정 대상이면 value_text, 아니면 plain)
    부터 넷째·다섯째·여섯째 자리(끝 0 유지)로 늘려 위치와 같게 읽히는 첫 표기. 판정 대상(target = (목표값, 높을수록 좋음,
    충족))이면 목표 판정과도 맞아야 한다. 여섯째 자리로도 안 되면 위치 쪽으로 자르고, 자른 값이 목표 판정과 어긋나면 base."""
    value = float(value)
    want = _position(value, low, high)

    def ok(text: str) -> bool:
        n = float(text)
        if _position(n, low, high) != want:
            return False
        return target is None or _meets(n, target[0], target[1]) == target[2]

    candidates = [base or ed4_plain(value)] + [_unneg(_fixed(value, d)) for d in range(PLAIN_DIGITS + 1, MAX_DIGITS + 1)]
    for text in candidates:
        if ok(text):
            return text
    scale = 10**MAX_DIGITS
    cut = math.floor(value * scale) / scale if want == "below" else math.ceil(value * scale) / scale if want == "above" else value
    cut_text = _unneg(_fixed(cut, MAX_DIGITS))
    if target is not None and not ok(cut_text):
        return base or ed4_plain(value)
    return cut_text


# ── 사실 시트 값의 판 4 표기(사실 보기·규칙 문장·숫자 허용 목록이 함께 쓰는 한 곳) ──────────────────────────────────


def metric_value_ed4(m) -> str:
    """지표 값의 판 4 표기 — 판정 대상이면 프론트 value_text(판정과 같은 쪽으로 늘린 표기), 없으면 받은 판정으로 judged,
    판정 대상이 아니면 plain. m 은 사실 시트 지표 줄(MetricFact)."""
    if m.value_text:
        return m.value_text
    if m.threshold is not None and m.higher_is_better is not None:
        return ed4_judged(m.value, m.threshold, m.higher_is_better, m.status == "pass")
    return ed4_plain(m.value)


def same_metric_value_ed4(metrics, metric_id: str, value: float | None) -> str | None:
    """파생값에 실린 값이 판정 대상 지표의 값이면 그 지표의 표기 — 같은 값이 두 자리에 다른 글자로 찍히지 않게(계약 결정 7).
    그 밖은 plain."""
    if value is None:
        return None
    m = next((m for m in metrics if m.metric_id == metric_id), None)
    if m is not None and m.threshold is not None and abs(float(m.value) - float(value)) < 1e-9:
        return metric_value_ed4(m)
    return ed4_plain(value)


def imbalance_texts_ed4(ie) -> dict[str, str | None]:
    """불균형 영향 세 판정 값과 네 값의 판 4 표기 — 프론트 display(판 4 요청은 판 4 표기)가 있으면 그것, 없으면 여기서 찍는다
    (엔진 level 이 판정이다 — 비율은 1.5 와, 두 차이는 0.1 과 비교). ie 는 ImbalanceEffectFact."""
    shown = ie.display
    ratio_meets = ie.level != "mild"
    out = {
        "imbalance_ratio": shown.imbalance_ratio if shown else ed4_judged(ie.imbalance_ratio, 1.5, True, ratio_meets),
        "majority_minus_minority_f1": shown.majority_minus_minority_f1 if shown else ed4_judged(
            ie.majority_minus_minority_f1, 0.1, True, ie.majority_minus_minority_f1 >= 0.1,
        ),
        "accuracy_minus_macro_f1": None,
    }
    if ie.accuracy_minus_macro_f1 is not None:
        out["accuracy_minus_macro_f1"] = (
            shown.accuracy_minus_macro_f1 if shown and shown.accuracy_minus_macro_f1 else ed4_judged(
                ie.accuracy_minus_macro_f1, 0.1, True, ie.accuracy_minus_macro_f1 >= 0.1,
            )
        )
    for key in ("accuracy", "macro_f1", "majority_f1", "minority_f1"):
        value = getattr(ie, key)
        text = getattr(shown, key) if shown else None
        out[key] = text or (ed4_plain(value) if value is not None else None)
    return out
