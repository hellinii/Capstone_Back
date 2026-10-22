"""app/narrative/wording.py — 서술 문장 도구(순수): 조사·숫자 표기·판정 이름·클래스 이름(설명) (서술 라운드 S6).

규칙 템플릿(app.narrative.fallback)과 결론 목록(app.narrative.required_items)이 같은 표기를 쓰도록 한곳에 둔다.
여기서 정한 표기는 숫자 허용 목록(app.narrative.grounding)이 받는 꼴이다 — 지표 값·목표값은 성적서와 같은 셋째 자리,
판정과 비교되는 값은 프론트가 판정과 같은 쪽으로 읽히게 만든 표시 문자열(*_text)을 그대로 쓴다.

- 조사(josa): 받침으로 은/는·이/가·을/를·과/와·으로/로를 고른다. 한글 음절·숫자(읽는 소리)·%·로마자(대략의 읽는
  소리)를 본다. 괄호로 끝나면 괄호 앞 말에 맞춘다("1(불량)은"). 프론트 koreanJosa 와 같은 공통 규칙(아래 '조사' 머리말). 이름 뒤 조사는 틀릴 수 있어 템플릿은 되도록 이름 뒤에
  '의'·괄호를 두고, 조사를 붙이는 자리는 숫자·한글 말 뒤로 한다.
- 작성 규칙(서술 설계 D37): 클래스는 이름과 의뢰자가 준 설명을 함께 쓴다(예: 1(불량)). 설명이 없으면 이름만.

상호작용
- 의존(import): 없음(표준 라이브러리 math·re)
- 사용처: app.narrative.fallback, app.narrative.required_items, app.narrative.prompt, tests
"""
from __future__ import annotations

import math
import re

# 판정 이름 — 성적서 8.2 판정 상자(프론트 ConclusionSection VERDICT_CONFIG)와 같은 이름 한 벌.
# 종전 '조건부 적합'·'부적합(불합격)'은 같은 절의 상자 이름과 달라 한 화면에 두 이름이 찍혔다.
VERDICT_LABEL = {
    "PASS": "최종 합격",
    "CONDITIONAL_PASS": "조건부 합격",
    "FAIL": "최종 불합격",
    "INCONCLUSIVE": "판정 불가",
}

# 지표 값·목표값을 찍는 자릿수 — 성적서(프론트 formatMetricValue·새 성적서 targetDigits)가 셋째 자리다.
METRIC_DIGITS = 3


# ── 조사 ────────────────────────────────────────────────────────────────────────
# 두 저장소 공통 규칙(세션 C 마지막 검토 — 3절은 프론트 koreanJosa.josaWaGwa, 7.2 는 여기라 같은 이름에 다른 조사가 찍혔다).
# 같은 대조 표를 양쪽 테스트(백엔드 test_session_c_final_review·프론트 koreanJosa.test)에 박는다.
#  1. 끝의 따옴표·공백은 건너뛴다.
#  2. '이름(설명)' 꼴이면 괄호 앞 이름의 끝소리로 고른다(괄호 안은 읽지 않는 것으로 보고 앞말에 맞춘다 — "2(정상)와").
#  3. 한글은 받침(ㄹ 받침도 받침 있음 — '으로/로'에서만 '로').
#  4. 숫자는 읽는 소리 — 소수는 끝 숫자 한 자(0 영·3 삼·6 육 받침, 1 일·7 칠·8 팔 ㄹ, 2·4·5·9 없음), 정수는 끝 0 개수로
#     십·백·천·만·억·조·경(프론트 numberFinalSound 와 같다 — 조만 받침 없음).
#  5. 영문 대문자로 끝나면(약어) 알파벳 이름 — L 엘·M 엠·N 엔·R 알만 받침(L·R 은 ㄹ).
#  6. 영문 소문자로 끝나면 -ng·-m·-n·-ck 받침, -l 은 ㄹ 받침, -t 는 바로 앞이 모음(a·e·i·o·u)일 때만 받침(cat·set) —
#     그 밖(-nt·-st·-s·-r·-k·-d·-g·-e 등)은 받침 없음(malignant 멀리그넌트·test 테스트).
#  7. 정할 수 없으면(기호만·빈 말) 두 꼴을 함께 쓴다('와(과)'·'은(는)').
# 숫자 끝자리를 읽는 소리의 받침 — (받침 있음, 받침이 ㄹ).
_DIGIT_FINAL = {
    "0": (True, False), "1": (True, True), "2": (False, False), "3": (True, False), "4": (False, False),
    "5": (False, False), "6": (True, False), "7": (True, True), "8": (True, True), "9": (False, False),
}
# 대문자 약어 끝 글자의 이름 중 받침이 있는 것 — L 엘·M 엠·N 엔·R 알.
_UPPER_FINAL = {"L": (True, True), "M": (True, False), "N": (True, False), "R": (True, True)}
_VOWELS = "aeiou"
_CLOSERS = "'\"‘’“”」』"
_TRAILING_NUMBER = re.compile(r"[\d,]*\d(?:\.\d+)?$")


def _number_final(text: str) -> tuple[bool, bool]:
    """숫자 표기(끝의 숫자 덩어리)를 읽는 소리의 끝 — 프론트 koreanJosa.numberFinalSound 와 같은 규칙."""
    text = text.replace(",", "")
    if "." in text:
        return _DIGIT_FINAL[text[-1]]
    if set(text) == {"0"}:
        return True, False  # 영
    zeros = len(text) - len(text.rstrip("0"))
    if zeros == 0:
        return _DIGIT_FINAL[text[-1]]
    if zeros < 4:
        return True, False  # 십·백·천
    return (False, False) if zeros // 4 == 3 else (True, False)  # 만·억·경은 받침, 조는 없음


def _final(word: str) -> tuple[bool, bool] | None:
    """말의 끝소리 — (받침 있음, 받침이 ㄹ). 정할 수 없으면 None. 규칙은 위 공통 규칙 1~7."""
    w = (word or "").rstrip()
    while w and w[-1] in _CLOSERS:
        w = w[:-1].rstrip()
    if w.endswith(")"):
        depth = 0
        for i in range(len(w) - 1, -1, -1):
            if w[i] == ")":
                depth += 1
            elif w[i] == "(":
                depth -= 1
                if depth == 0:
                    if w[:i].strip():
                        return _final(w[:i])
                    w = w[i + 1:-1].rstrip()
                    break
    if not w:
        return None
    ch = w[-1]
    code = ord(ch) - 0xAC00
    if 0 <= code < 11172:
        final = code % 28
        return final != 0, final == 8
    if ch.isascii() and ch.isdigit():
        return _number_final(_TRAILING_NUMBER.search(w).group(0))
    if ch == "%":  # 퍼센트
        return False, False
    if "A" <= ch <= "Z":
        return _UPPER_FINAL.get(ch, (False, False))
    if "a" <= ch <= "z":
        if w.endswith("ng") or w.endswith("ck") or ch in "mn":
            return True, False
        if ch == "l":
            return True, True
        if ch == "t" and len(w) >= 2 and w[-2].lower() in _VOWELS:
            return True, False
        return False, False
    return None


# 끝소리를 정할 수 없을 때 쓰는 두 꼴(프론트 josaWaGwa 의 '와(과)'·josaEunNeun 의 '은(는)' 과 같은 꼴).
_UNDECIDED = {"과/와": "와(과)", "으로/로": "(으)로", "이다/다": "(이)다"}


def josa(word: str, pair: str) -> str:
    """말 뒤에 붙일 조사 — pair 는 '받침 있을 때/없을 때'(예: '은/는', '이/가', '을/를', '과/와', '으로/로', '이다/다').

    '으로/로' 는 ㄹ 받침 뒤에도 '로'다(예: 1(일)로, 7(칠)로). 끝소리를 정할 수 없으면 '은(는)'·'와(과)' 꼴.
    """
    with_final, without_final = pair.split("/")
    sound = _final(word)
    if sound is None:
        return _UNDECIDED.get(pair, f"{with_final}({without_final})")
    has_final, rieul = sound
    if pair == "으로/로":
        return "으로" if has_final and not rieul else "로"
    return with_final if has_final else without_final


def with_josa(word: str, pair: str) -> str:
    return f"{word}{josa(word, pair)}"


# ── 숫자 표기 ──────────────────────────────────────────────────────────────────


def num(value) -> str:
    """판정과 무관한 실수는 소수 넷째 자리까지(허용 목록이 받는 표기 — grounding._add). 정수·글은 그대로.

    판정과 비교되는 값은 표시 문자열(*_text·display·value_text)을 쓴다 — 이 함수로 줄이지 않는다."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        # 고정 소수(지수 표기 없음) — ':g' 는 유효 숫자 6개라 1234567.5 를 '1.23457e+06' 으로 찍고 숫자 검문에 걸렸다(세션 A 검토).
        text = f"{round(value, 4):.4f}".rstrip("0").rstrip(".")
        if text in ("0", "-0") and value != 0:
            text = f"{value:.10f}".rstrip("0").rstrip(".")  # 0.00001 을 '0' 으로 찍지 않는다
        return "0" if text in ("-0", "") else text
    return str(value)


def metric_num(value) -> str:
    """지표 값·목표값·차이·구간 끝을 성적서와 같은 셋째 자리로(끝의 0 은 뺀다).

    반올림한 값(round(v, 3))은 허용 목록(_add)이 넣는 표기라 grounding 을 통과한다.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    text = f"{round(float(value), METRIC_DIGITS):.{METRIC_DIGITS}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def value_target(value: float, threshold: float) -> tuple[str, str]:
    """미달 지표의 값과 목표를 찍는다. 셋째 자리로 같아 보이면 넷째 자리까지 늘린다.

    0.7999 를 셋째 자리로 찍으면 '0.8, 목표 0.8' 인데 미달이라 읽는 사람에게 모순이다. 넷째 자리까지가
    허용 목록(grounding._add)의 표기다. 넷째 자리로도 같으면 값 대신 '목표에 근소하게 못 미침'을 쓴다
    (빈 문자열을 돌려 호출부가 고른다).
    """
    v, t = metric_num(value), metric_num(threshold)
    if v != t or value == threshold:
        return v, t

    def four(x: float) -> str:
        text = f"{round(float(x), 4):.4f}".rstrip("0").rstrip(".")
        return "0" if text == "-0" else text

    v4, t4 = four(value), four(threshold)
    if v4 != t4:
        return v4, t4
    return "", t


def target_text(threshold: float) -> str:
    """목표값 — 반올림하지 않는다(소수 넷째 자리 목표 0.8125 를 0.813 으로 찍으면 값 0.812 와 같아 보여 미달이 충족처럼
    읽힌다 — S4b 검토 FB-01). 끝의 0 은 뺀다. 목표값 그대로라 허용 목록 안이다."""
    text = f"{float(threshold):.9f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def shown_value(m) -> str:
    """지표 값 한 개의 인쇄 표기 — 판정 대상이면 프론트 표시 문자열(value_text, 성적서 6.1 과 같음), 없으면(옛 요청)
    목표와 같아 보이지 않게 자릿수를 늘린 값, 판정 대상이 아니면 넷째 자리."""
    if m.value_text:
        return m.value_text
    if m.threshold is None:
        return num(float(m.value))
    v, _ = value_target(m.value, m.threshold)
    return v or metric_num(m.value)


def _position(value: float, low: float, high: float) -> str:
    # baselines.benchmark_position 과 같은 규칙(범위 끝은 안) — wording 은 다른 모듈을 import 하지 않는다.
    return "below" if value < low else "above" if value > high else "within"


# 8.1 모델 값을 늘리는 최대 자릿수 — 프론트 judgedNumber.MAX_VALUE_DIGITS 와 같다(세션 C 마지막 검토: 백엔드 9·프론트 6 이라
# 범위 끝 근처에서 8.1 표와 서술 숫자가 달랐다).
RANGE_MAX_DIGITS = 6


def meets_target(value: float, threshold: float, higher_is_better: bool) -> bool:
    """목표 충족인가 — 프론트 judgedNumber.meetsTarget 과 같다(차이를 10자리로 반올림해 뺄셈 끝자리 잡음을 걷는다)."""
    d = round(float(value) - float(threshold), 10)
    return d >= 0 if higher_is_better else d <= 0


def range_value_text(
    value: float, low: float, high: float, base: str | None = None,
    target: tuple[float, bool, bool] | None = None,
) -> str:
    """8.1 참조 범위 비교의 모델 값 표기 — 찍힌 숫자가 위치 판정(반올림 전 값으로 한 below/within/above)과 같은 쪽으로 읽히게.

    프론트 judgedNumber.formatBenchmarkValue 와 같은 규칙(두 저장소가 같은 대조 표로 시험한다):
    - 평소 표기(base — 판정 대상의 6.1 표기, 없으면 넷째 자리)부터 여섯째 자리까지 늘려, 찍힌 숫자가 위치와 같은 쪽으로 읽히는
      첫 표기를 쓴다(라운드 검토 P2 — Precision 0.79996 이 '0.8, 참조 범위 0.8~0.92' 로 찍히는데 미흡).
    - 판정 대상 지표(target = (목표값, 높을수록 좋음, 충족))면 같은 표기가 목표 판정(충족/미달)과도 맞아야 한다.
    - 여섯째 자리로도 안 되면 위치 쪽으로 자른다(below 는 내림, above 는 올림). 판정 대상인데 자른 값이 목표 판정과 어긋나면
      목표 판정이 우선 — 평소 표기(base, 성적서 6.1 과 같은 엔진·프론트 판정 표기)를 쓴다.
    응답의 벤치마크 표(model_value)는 바꾸지 않는다(계약)."""
    value = float(value)
    want = _position(value, low, high)

    def ok(text: str) -> bool:
        n = float(text)
        if _position(n, low, high) != want:
            return False
        return target is None or meets_target(n, target[0], target[1]) == target[2]

    candidates = ([base] if base else []) + [num(value)] + [
        (f"{round(value, d):.{d}f}".rstrip("0").rstrip(".") or "0") for d in range(5, RANGE_MAX_DIGITS + 1)
    ]
    for text in candidates:
        if ok(text):
            return text
    scale = 10 ** RANGE_MAX_DIGITS
    cut = math.floor(value * scale) / scale if want == "below" else math.ceil(value * scale) / scale if want == "above" else value
    cut_text = f"{cut:.{RANGE_MAX_DIGITS}f}".rstrip("0").rstrip(".") or "0"
    if target is not None and not ok(cut_text):
        return base or num(value)
    return cut_text


def metric_vs_target(m) -> str:
    """'지표명(값, 목표 t)' — 값은 프론트가 판정과 같은 쪽으로 읽히게 만든 인쇄 표기(value_text, 성적서 6.1 과 같음).

    value_text 가 없는 옛 요청만 여기서 자릿수를 늘리거나 '목표 t에 근소하게 못 미침'으로 쓴다.
    """
    if m.value_text:
        return f"{m.display_name}({m.value_text}, 목표 {target_text(m.threshold)})"
    v, t = value_target(m.value, m.threshold)
    if not v:
        return f"{m.display_name}(목표 {t}에 근소하게 못 미침)"
    return f"{m.display_name}({v}, 목표 {t})"


def pct(ratio: float) -> str:
    """비율(0~1)을 백분율 소수 1자리로 — 허용 목록(_add)의 백분율 표기와 같은 반올림."""
    return f"{round(ratio * 100, 1):g}%"


def pct_text(fraction_text: str) -> str:
    """판정 쪽으로 자릿수를 늘린 비율 표기(0~1, 예 '0.04996')를 같은 정밀도의 백분율로('4.996%').

    pct(소수 첫째 자리)로 바꾸면 0.04996 이 '5%' 가 되어 '5% 미만' 판정과 어긋난다(S4b 검토 RI-01).
    """
    digits = max(len(fraction_text.split(".", 1)[1]) - 2, 1) if "." in fraction_text else 1
    return f"{round(float(fraction_text) * 100, digits):g}%"


def unsigned(number_text: str) -> str:
    """부호 있는 표기의 크기('-0.0659' → '0.0659'). 허용 목록은 부호 있는 차이의 절댓값도 받는다."""
    return number_text.lstrip("-−+")


def join_and(items: list[str]) -> str:
    """'A, B, C' — 나열(조사 없이 쉼표)."""
    return ", ".join(items)


# 열 개까지의 개수를 숫자 없이 — 숫자 검문에 걸리지 않게 '상위 3개' 대신 '세'를 쓴다(그보다 많으면 호출부가 '여러').
KOREAN_COUNT = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열"}


# ── 고객 입력의 '제공되지 않음' ──────────────────────────────────────────────────
# 프론트 reportMaterials.NOT_PROVIDED — 답이 없을 때 client_inputs 에 싣는 말(재료 선택 점검 4.2).
NOT_PROVIDED = "제공되지 않음"


def is_not_provided(value) -> bool:
    """고객 입력 한 칸이 제공되지 않았는가 — 비었거나(글이 아님 포함) '제공되지 않음'으로 시작한다.

    프론트 materialRowText 는 '모름 + 글'·'기타 빈칸 + 글'을 "제공되지 않음 (의뢰자 제공: …)" 으로 보낸다. 값은 제공되지
    않은 것이고 괄호 글은 의뢰자 참고 글이다(점검 표 2.0·1.4 도 빠짐으로 센다). 그래서 '같은가'가 아니라 '시작하는가'로
    본다 — 고객 입력이 제공되었는지 보는 자리는 모두 이 함수 하나를 쓴다(지시문 규칙 17 과 같은 기준)."""
    return not isinstance(value, str) or not value.strip() or value.strip().startswith(NOT_PROVIDED)


# ── 클래스 이름(설명) — 작성 규칙 D37 ──────────────────────────────────────────


def class_descriptions(client_inputs: dict | None) -> dict[str, str]:
    """고객 입력의 클래스 설명(class_label_descriptions: {클래스: 설명}) — 빈 설명·'제공되지 않음'(으로 시작하는 글)은 뺀다."""
    raw = (client_inputs or {}).get("class_label_descriptions") if isinstance(client_inputs, dict) else None
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        text = str(v).strip() if v is not None else ""
        if not is_not_provided(text) and text != str(k).strip():
            out[str(k).strip()] = text
    return out


def class_name(label, descriptions: dict[str, str]) -> str:
    """'이름(설명)' — 설명이 없으면 이름만(D37)."""
    name = str(label)
    desc = descriptions.get(name.strip())
    return f"{name}({desc})" if desc else name
