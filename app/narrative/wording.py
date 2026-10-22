"""app/narrative/wording.py — 서술 문장 도구(순수): 조사·숫자 표기·판정 이름·클래스 이름(설명) (서술 라운드 S6).

규칙 템플릿(app.narrative.fallback)과 결론 목록(app.narrative.required_items)이 같은 표기를 쓰도록 한곳에 둔다.
여기서 정한 표기는 숫자 허용 목록(app.narrative.grounding)이 받는 꼴이다 — 지표 값·목표값은 성적서와 같은 셋째 자리,
판정과 비교되는 값은 프론트가 판정과 같은 쪽으로 읽히게 만든 표시 문자열(*_text)을 그대로 쓴다.

- 조사(josa): 받침으로 은/는·이/가·을/를·과/와·으로/로를 고른다. 한글 음절·숫자(읽는 소리)·%·로마자(대략의 읽는
  소리)를 본다. 괄호로 끝나면 괄호 앞 말에 맞춘다("1(불량)은"). 이름 뒤 조사는 틀릴 수 있어 템플릿은 되도록 이름 뒤에
  '의'·괄호를 두고, 조사를 붙이는 자리는 숫자·한글 말 뒤로 한다.
- 작성 규칙(서술 설계 D37): 클래스는 이름과 의뢰자가 준 설명을 함께 쓴다(예: 1(불량)). 설명이 없으면 이름만.

상호작용
- 의존(import): 없음(표준 라이브러리)
- 사용처: app.narrative.fallback, app.narrative.required_items, app.narrative.prompt, tests
"""
from __future__ import annotations

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
# 숫자 끝자리를 읽는 소리의 받침 — (받침 있음, 받침이 ㄹ). 0(영)·3(삼)·6(육)은 받침, 1(일)·7(칠)·8(팔)은 ㄹ 받침.
# 10·100·1000 처럼 0 으로 끝나는 수(십·백·천·만)도 모두 ㄹ 이 아닌 받침이라 0 과 같다.
_DIGIT_FINAL = {
    "0": (True, False), "1": (True, True), "2": (False, False), "3": (True, False), "4": (False, False),
    "5": (False, False), "6": (True, False), "7": (True, True), "8": (True, True), "9": (False, False),
}
# 로마자 끝 글자 — 대문자 약어는 글자 이름(엘·엠·엔·알), 소문자 낱말은 흔한 소리(-m·-n·-ng·-l 은 받침). 대략값이다.
_UPPER_FINAL = {"L": (True, True), "M": (True, False), "N": (True, False), "R": (True, True)}
_LOWER_FINAL = {"l": (True, True), "m": (True, False), "n": (True, False), "g": None, "k": (False, False)}
_CLOSERS = "'\"‘’“”」』"


def _final(word: str) -> tuple[bool, bool]:
    """말의 끝소리 — (받침 있음, 받침이 ㄹ). 괄호·따옴표로 끝나면 그 앞 말을 본다."""
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
                    w = w[i + 1:-1]
                    break
    if not w:
        return False, False
    ch = w[-1]
    code = ord(ch) - 0xAC00
    if 0 <= code < 11172:
        final = code % 28
        return final != 0, final == 8
    if ch in _DIGIT_FINAL:
        return _DIGIT_FINAL[ch]
    if ch == "%":  # 퍼센트
        return False, False
    if ch in _UPPER_FINAL and (len(w) == 1 or not w[-2].islower()):
        return _UPPER_FINAL[ch]
    if ch.islower():
        if ch == "g":
            return (True, False) if w.endswith("ng") else (False, False)
        hit = _LOWER_FINAL.get(ch)
        if hit is not None:
            return hit
    return False, False


def josa(word: str, pair: str) -> str:
    """말 뒤에 붙일 조사 — pair 는 '받침 있을 때/없을 때'(예: '은/는', '이/가', '을/를', '과/와', '으로/로', '이다/다').

    '으로/로' 는 ㄹ 받침 뒤에도 '로'다(예: 1(일)로, 7(칠)로).
    """
    with_final, without_final = pair.split("/")
    has_final, rieul = _final(word)
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
