"""app/evaluation/metrics/display.py — 판정과 맞게 자릿수를 늘린 표시 문자열(S2).

엔진은 반올림 전 값으로 기준을 판정하고 넷째 자리로 반올림한 값을 싣는다. 그런데 반올림하면 기준과 같거나
반대쪽으로 읽히는 값이 있다 — IR 1.49996 은 '경미'(1.5 미만)인데 넷째 자리로는 1.5, 임계값 조정 여지
0.09996 은 '부적합 아님'인데 0.1. 그대로 찍으면 성적서·서술이 "1.5 로 1.5 미만" 같은 자기모순을 쓴다.
`judged_text` 는 찍힌 숫자가 판정과 같은 쪽으로 읽힐 때까지 자릿수를 늘린다(사용자 결정 2026-10-19
"판정 자릿수": 실제 값으로 판정 + 어긋나 보이는 값만 자릿수를 늘려 찍기). 프론트 3절 IR 표기
(buildDatasetDiagnosis.formatImbalanceRatio)와 같은 원칙이다.

상호작용
- 의존(import): math
- 사용처: metrics.counts(imbalance_effect.display), metrics.binary(f1_threshold_gain.gain_text)
"""
import math

# 평소 자릿수 — 엔진이 싣는 값(round(x, 4))과 같다.
DEFAULT_DIGITS = 4
# 이 자릿수까지 늘린다. 판정은 9자리로 잡음을 걷어 낸 값으로 하므로(counts._clean) 그 이상은 의미가 없다.
MAX_DIGITS = 9


def _text(value: float, digits: int) -> str:
    text = f"{round(float(value), digits):.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def judged_text(value: float, threshold: float, meets: bool, digits: int = DEFAULT_DIGITS) -> str:
    """`value` 를 `digits` 자리로 찍되, 찍힌 숫자가 판정(`meets` = 값 ≥ 기준)과 반대로 읽히면 자릿수를 늘린다.

    끝의 0 은 뺀다(grounding 허용 목록의 `:g` 표기와 같은 모양). 9자리에서도 반대로 읽히면 판정 쪽으로 자른다
    (1.4999999999 → '1.499999999') — 숫자를 지어내지 않되 반대쪽으로 읽히지 않게.
    """
    for d in range(digits, MAX_DIGITS + 1):
        text = _text(value, d)
        if (float(text) >= threshold) == meets:
            return text
    scale = 10 ** MAX_DIGITS
    cut = math.ceil(value * scale) / scale if meets else math.floor(value * scale) / scale
    return _text(cut, MAX_DIGITS)
