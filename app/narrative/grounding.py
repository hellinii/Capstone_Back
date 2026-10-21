"""app/narrative/grounding.py — 환각 방어(grounding) 순수 로직

LLM 출력의 모든 숫자가 fact_sheet/파생값 화이트리스트 안에 있는지 검증한다. 위반 시
호출부(narrator)가 규칙 폴백으로 대체한다. 숫자 화이트리스트 구성·검증·문자열 수집을 담당.

상호작용
- 의존(import): re, typing.Any, app.narrative.schemas(FactSheet, GroundingInfo),
  app.narrative.derived(THRESHOLD_GAIN_NOTABLE), app.evaluation.metrics.counts(불균형 기준값)
- 사용처: app.narrative.service(generate_narrative 의 화이트리스트/검증), tests
"""
import re
from typing import Any

# 불균형 영향 기준(D20)은 평가 엔진이 판정에 쓴 값을 그대로 가져온다(값마다 한 곳).
from app.evaluation.metrics.counts import IMBALANCE_F1_GAP, IMBALANCE_MILD_BELOW
from app.narrative.derived import THRESHOLD_GAIN_NOTABLE
from app.narrative.schemas import FactSheet, GroundingInfo


# 문맥상 '수치'가 아닌 고정 토큰(표준 표기·절 번호·오류 유형 서수)은 검증 전에 제거한다.
# 0/1/100/연도 같은 범용 숫자는 더 이상 무조건 면제하지 않는다 — fact_sheet 근거로만 허용(D3a).
#
# 절 번호: "7절"뿐 아니라 "8.3절에서"·"7.1 절"처럼 소수점 꼴도 절 번호다. 소수점 꼴은 첫 자리가 1~9 인
# 한 자리 + 소수 한 자리(d.d)만 받는다 — "0.82절"처럼 0 으로 시작하는 지표값 꼴은 절 번호로 보지 않는다.
# '절' 뒤가 절대·절반·절감·절차 같은 다른 낱말이면 절 번호가 아니다("0.3 절감").
# 괄호 안("(8.2)")이나 "절 " 뒤에 홀로 오는 소수점 꼴은 성적서의 실제 절 번호(6.0~6.4·7.1~7.3·8.1~8.3·9.1·9.2)일
# 때만 지운다 — 아무 소수나 지우면 "(0.82)"·"8.2%" 같은 가짜 수치가 검증을 빠져나간다.
# 절 번호 나열("7.1·7.2절", "8.2와 8.3절에서", "(8.2, 8.3)")도 실제 절 번호끼리 ·/,/및/와/과 로 이은 꼴이
# '절' 앞에 오거나 괄호로 감싸였을 때만 지운다 — "(0.82, 0.91)" 같은 구간 값은 그대로 검증한다.
# "절 8.3." 처럼 문장 끝 마침표가 붙어도 절 번호다(뒤에 숫자가 이어지는 "8.3.5"·"8.3,000" 은 아니다).
_SECTION_IDS = r"(?:6\.[0-4]|7\.[1-3]|8\.[1-3]|9\.[12])"
_SECTION_SEP = r"\s*(?:[·,/]|및|와|과)\s*"
_NOT_SECTION_WORD = r"(?!대|반|감|차|약|단|편|충)"
_CTX_TOKEN_RE = re.compile(
    r"ISO/?IEC\s+TS\s*4213(?:\s*:\s*2022)?|4213\s*:\s*2022"
    rf"|(?<![\d.]){_SECTION_IDS}(?:{_SECTION_SEP}{_SECTION_IDS})+\s*절{_NOT_SECTION_WORD}"
    rf"|(?<![\d.])(?:[1-9]\.\d|\d+)\s*절{_NOT_SECTION_WORD}"
    rf"|\(\s*{_SECTION_IDS}(?:{_SECTION_SEP}{_SECTION_IDS})*\s*\)"
    rf"|(?<=절)\s+{_SECTION_IDS}(?!\d|[.,]\d|%)"
    # 글 맨 앞의 소절 머리표("9.1 데이터 측면에서는…"). 2026-10-19 Claude 실호출에서 9절 두 칸 앞에 붙여 써
    # 숫자 검문 폴백을 냈다("9.1 데이터 측면에서", "9.1에서 지목된"). 글 맨 앞에서, 실제 절 번호 뒤에
    # 띄어쓰기 + 한글이나 절을 가리키는 조사(에서·에는·에·의·은·는)가 올 때만 지운다 — "9.1로 측정"은 값이다.
    rf"|^\s*{_SECTION_IDS}(?=\s+[가-힣]|에서|에는|에|의|은|는)"
    r"|제\s*\d+\s*종"
)

# 신뢰구간의 신뢰수준(서술 설계 D29). 평가 응답 confidence_intervals 의 level 과 같다.
CI_LEVEL = 0.95

# 서술용 파생값 판정 기준(계약 constants — 프론트 narrativeFacts.ts 가 판정에 쓴 값). 켜진 판정이 있을 때만 허용한다.
LOW_SUPPORT_MIN = 30
LOW_SHARE_MIN = 0.05
WEAKEST_F1_GAP = 0.1
PR_GAP_MIN = 0.1
DISTRIBUTION_SHIFT_MIN = 0.1

# 아슬아슬 기준(서술 설계 D12) — 목표값과의 차이가 목표값의 3% 미만. 판정은 프론트가 하고(near_target),
# 서술이 "3%"라고 쓸 수 있게 near_target 지표가 있을 때만 허용한다.
NEAR_TARGET_SHARE = 0.03

# 숫자 토큰: 앞 글자가 영숫자면(F1·P99 등 식별자) 숫자로 보지 않는다(선행 lookbehind).
# 부호(음수 정답값 -0.1 등)·천단위 콤마·백분율(%)을 함께 포착한다(D7[2]).
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[+-]?\d+(?:[.,]\d+)?%?")


# 유니코드 마이너스(U+2212)와 대시는 숫자 앞에서 빼기 부호다 — 검증 전에 ASCII '-' 로 맞춘다. 맞추지 않으면
# 정규식이 부호를 놓쳐 "−0.05" 를 "0.05" 로 읽는다(점검 D-V4).
_MINUS_RE = re.compile(r"[\u2212\u2013\u2012](?=\d)")


def _normalize_minus(text: str) -> str:
    return _MINUS_RE.sub("-", text)


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


def _add_signed(wl: set, v: Any) -> None:
    """부호 있는 차이 — 값과 크기(절댓값)를 함께. "목표보다 0.0288 낮다"처럼 부호 없이 크기만 적어도 같은 사실이다."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return
    _add(wl, f)
    _add(wl, abs(f))


def _decimals(text: str) -> int:
    return len(text.split(".", 1)[1]) if "." in text else 0


def _add_judged(wl: set, value: Any, text: str | None, default_digits: int) -> None:
    """기준과 비교되는 값 — 판정과 같은 쪽으로 읽히게 자릿수를 늘린 표시 문자열(text)이 있으면 그 표기를 넣고,
    text 가 평소보다 길면(반올림하면 판정과 반대로 읽혀 늘린 값) 그보다 짧은 반올림 표기는 넣지 않는다(계획 S4b —
    '판정과 어긋나 보이는 짧은 반올림 표기는 뺀다'). 기준값 자체는 따로 허용되므로 이것으로 모든 모순을 막지는 못한다.
    """
    if not text:
        _add(wl, value)
        return
    c = _canon(text)
    if c is not None:
        wl.add(c)
    d = _decimals(text)
    if d <= default_digits:
        _add(wl, value)
        return
    try:
        f = float(value)
    except (TypeError, ValueError):
        return
    wl.add(f"{f:g}")
    for k in range(d, 10):
        wl.add(f"{round(f, k):g}")
    if 0.0 <= f <= 1.0:
        for k in range(max(d - 2, 0), 8):
            wl.add(f"{round(f * 100, k):g}")


def _add_fact_numbers(wl: set, obj: Any) -> None:
    """서술용 파생값(facts) 안의 모든 숫자 — 값, 음수면 크기, 글(이름·표기·사유) 속 숫자. bool 은 건너뛴다."""
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        _add_signed(wl, obj)
    elif isinstance(obj, str):
        for tok in _NUMBER_RE.findall(_CTX_TOKEN_RE.sub(" ", _normalize_minus(obj))):
            c = _canon(tok)
            if c is not None:
                _add_signed(wl, c)
    elif isinstance(obj, dict):
        for v in obj.values():
            _add_fact_numbers(wl, v)
    elif isinstance(obj, list):
        for item in obj:
            _add_fact_numbers(wl, item)


def _add_client_numbers(wl: set, obj: Any) -> None:
    """고객 입력 안의 숫자(학습 표본 수, 라벨러 수, 일치도, 수집 기간 등)를 허용 목록에 넣는다.

    고객 입력은 계산값이 아니라 의뢰자가 적은 사실이다. 서술이 그 숫자를 옮겨 적는 것은 환각이
    아니므로 허용한다. 글 속의 숫자는 검증 쪽과 같은 정규식으로 뽑는다.
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        _add(wl, obj)
    elif isinstance(obj, str):
        for tok in _NUMBER_RE.findall(_CTX_TOKEN_RE.sub(" ", _normalize_minus(obj))):
            c = _canon(tok)
            if c is not None:
                _add(wl, c)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _add_client_numbers(wl, k)
            _add_client_numbers(wl, v)
    elif isinstance(obj, list):
        for item in obj:
            _add_client_numbers(wl, item)


def build_number_whitelist(
    fs: FactSheet,
    benchmark_refs: list[dict],
    derived: dict,
    client_inputs: dict | None = None,
) -> set:
    """출력 검증용 허용 숫자 집합 (fact_sheet + 파생값 + 고객 입력의 다중 표기)."""
    wl: set = set()

    _add(wl, fs.n_samples)
    _add(wl, fs.dropped_rows)
    _add(wl, fs.score)

    for m in fs.metrics:
        # 값은 판정과 같은 쪽으로 읽히는 인쇄 표기(value_text, 평소 셋째 자리)가 있으면 그것을 따른다.
        _add_judged(wl, m.value, m.value_text, 3)
        _add(wl, m.beta)  # F-beta 의 β — 서술이 "β=2" 라고 쓸 수 있다(None 이면 건너뛴다)
        _add_client_numbers(wl, m.display_name)  # 이름 속 숫자(F1·"Top-5" 등 — 식별자 뒤 숫자는 원래 안 뽑힌다)
        if m.margin_text:
            _add_signed(wl, _canon(m.margin_text))
        if m.threshold is not None:
            _add(wl, m.threshold)
        # 95% 신뢰구간(D29) — None 이면 _add 가 건너뛴다.
        _add(wl, m.ci_low)
        _add(wl, m.ci_high)
        _add(wl, m.margin)  # 목표값과의 차이(D12) — None 이면 _add 가 건너뛴다
        if m.margin is not None:
            # "목표값보다 0.0288 낮다"처럼 부호 없이 크기만 적는 것도 같은 사실이다.
            _add(wl, abs(m.margin))
    # 구간이 실렸을 때만 신뢰수준(95%)을 허용한다 — 평가 응답의 level 과 같은 값이다.
    if any(m.ci_low is not None and m.ci_high is not None for m in fs.metrics):
        _add(wl, CI_LEVEL)
    if any(m.near_target for m in fs.metrics):
        _add(wl, NEAR_TARGET_SHARE)
    # 6.0 신뢰구간 방식 한 줄(D29) — "재표집(2,000회)"처럼 글 속 숫자를 서술이 옮겨 적을 수 있다.
    # 프론트가 만든 고정 문안이라 고객 입력과 같이 글 속 숫자를 뽑는다(None 이면 건너뛴다).
    _add_client_numbers(wl, fs.ci_method_note)

    # 클래스 이름 속 숫자("2"·"class_3"·"v2 결함")도 허용한다 — 이름을 옮겨 적는 것은 수치 주장이 아니다.
    for pc in fs.per_class:
        for val in (pc.precision, pc.recall, pc.f1, pc.support):
            _add(wl, val)
        _add_client_numbers(wl, pc.label)

    # 건수와, 서술이 흔히 더해 쓰는 합(클래스마다 오류 fp+fn·예측 수 tp+fp·실제 수 tp+fn). S1 실호출에서 모델이
    # 직접 더한 "합계 57건"이 허용 목록 밖이라 서술 전체가 규칙 문장이 됐다 — 자주 쓰는 합은 미리 넣는다.
    for row in fs.confusion_counts:
        for val in (row.tp, row.fp, row.fn, row.tn, row.fp + row.fn, row.tp + row.fp, row.tp + row.fn):
            _add(wl, val)
        _add_client_numbers(wl, row.label)

    for avg in fs.class_averages:
        for val in (avg.precision, avg.recall, avg.f1, avg.support):
            _add(wl, val)

    if fs.confusion:
        for row in fs.confusion.matrix:
            for cell in row:
                _add(wl, cell)
            _add(wl, sum(row))  # 그 클래스의 실제 수
        width = max((len(row) for row in fs.confusion.matrix), default=0)
        for j in range(width):  # 그 클래스로 예측한 수
            _add(wl, sum(row[j] for row in fs.confusion.matrix if j < len(row)))
        for label in fs.confusion.labels:
            _add_client_numbers(wl, label)

    if fs.distribution:
        for k, v in fs.distribution.class_distribution.items():
            _add(wl, v)
            _add_client_numbers(wl, k)
        if fs.distribution.imbalance_ratio is not None:
            _add(wl, fs.distribution.imbalance_ratio)

    if fs.derived_prediction:
        threshold = fs.derived_prediction.threshold
        for val in threshold.values() if isinstance(threshold, dict) else (threshold,):
            _add(wl, val)

    # 경계·확신 오류 비율(D30) — 건수·비율(백분율 표기 포함)과 두 기준(0.1·0.9), 경계 판정 임계값.
    # 키가 없으면 0.1·0.9 는 허용하지 않는다(다른 사실에서 나오지 않는 한).
    er = fs.error_ratios
    if er:
        _add(wl, er.misclassified)
        for part in (er.boundary, er.confident):
            if part is not None:
                _add(wl, part.count)
                _add(wl, part.ratio)
        _add(wl, er.boundary_margin)
        _add(wl, er.confident_level)
        for val in er.threshold.values() if isinstance(er.threshold, dict) else (er.threshold,):
            _add(wl, val)

    # 임계값 조정 여지(D40) — 네 값과 판정 기준 0.10. 키가 없으면 0.1 은 다른 사실에서만 나온다.
    gain = fs.f1_threshold_gain
    if gain:
        for val in (gain.max_f1, gain.current_f1, gain.best_threshold):
            _add(wl, val)
        _add_judged(wl, gain.gain, gain.gain_text, 4)
        _add(wl, THRESHOLD_GAIN_NOTABLE)

    # 불균형 영향(D20) — 비율·두 차이(음수일 수 있다)와 기준 1.5·0.10. 숫자 꼴 클래스 이름("2")도 허용한다.
    ie = fs.imbalance_effect
    if ie:
        shown = ie.display
        _add_judged(wl, ie.imbalance_ratio, shown.imbalance_ratio if shown else None, 4)
        for val, text in (
            (ie.accuracy_minus_macro_f1, shown.accuracy_minus_macro_f1 if shown else None),
            (ie.majority_minus_minority_f1, shown.majority_minus_minority_f1 if shown else None),
        ):
            if val is None:
                continue
            _add_judged(wl, val, text, 4)
            _add_judged(wl, abs(val), text.lstrip("-") if text else None, 4)
        _add_client_numbers(wl, ie.majority_class)
        _add_client_numbers(wl, ie.minority_class)
        _add(wl, IMBALANCE_MILD_BELOW)
        _add(wl, IMBALANCE_F1_GAP)

    # 문제 목록(D22) — 근거 수치(evidence)와 이름·덧붙임 속 숫자(숫자 꼴 클래스 이름 등).
    # evidence 는 자유 dict 라 고객 입력과 같이 재귀로 뽑는다.
    for problem in fs.problems:
        _add_client_numbers(wl, problem.evidence)
        _add_client_numbers(wl, problem.title)
        _add_client_numbers(wl, problem.note)

    if fs.latency and fs.latency.available:
        for val in (fs.latency.mean, fs.latency.p50, fs.latency.p95, fs.latency.p99):
            _add(wl, val)
        # 백분위 이름("95번째 백분위수"·"상위 99%") — 지연시간이 있을 때만.
        for pct in (50, 95, 99):
            _add(wl, pct)

    # 서술용 파생값(S4a facts) — 모든 숫자(음수는 크기도)와, 켜진 판정의 기준값만(계획 S4b).
    facts = fs.facts
    if facts is not None:
        _add_fact_numbers(wl, facts.model_dump(mode="json"))
        if facts.weakest_class is not None or facts.low_support_classes:
            _add(wl, LOW_SUPPORT_MIN)
            _add(wl, LOW_SHARE_MIN)
        if facts.weakest_class is not None and facts.weakest_class.gap_from_macro is not None:
            _add(wl, WEAKEST_F1_GAP)
        if facts.precision_recall_gap is not None:
            _add(wl, PR_GAP_MIN)
        if facts.distribution_shift:
            _add(wl, DISTRIBUTION_SHIFT_MIN)

    for ref in benchmark_refs:
        _add(wl, ref.get("model_value"))
        _add(wl, ref.get("ref_low"))
        _add(wl, ref.get("ref_high"))
        _add(wl, ref.get("target_value"))  # 목표값 수준(D31) — None 이면 _add 가 건너뛴다

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

    if client_inputs:
        _add_client_numbers(wl, client_inputs)

    return wl


def verify_grounding(texts: list[str], whitelist: set) -> GroundingInfo:
    """텍스트들에서 숫자 토큰을 추출해 화이트리스트와 대조. 위반 = 환각 수치."""
    violations: list[str] = []
    checked = 0
    for t in texts:
        if not t:
            continue
        # 표준 표기·절/서수 등 문맥 토큰을 먼저 제거한 뒤 숫자만 검증(오탐 방지, D3a). 유니코드 마이너스는 '-' 로.
        stripped = _CTX_TOKEN_RE.sub(" ", _normalize_minus(t))
        for tok in _NUMBER_RE.findall(stripped):
            c = _canon(tok)
            if c is None:
                continue
            checked += 1
            if c not in whitelist:
                violations.append(tok)
    uniq = sorted(set(violations))
    return GroundingInfo(checked=checked, violations=uniq, passed=len(uniq) == 0)


def has_number_token(text: str) -> bool:
    """글에 숫자 토큰이 있는지 — verify_grounding 과 같은 기준(문맥 토큰 제거 후 같은 정규식).

    권고의 기대 효과(expected_impact)에 쓴다(서술 설계 D22). 기대 효과의 숫자는 허용 목록 안이어도
    "F1 0.8 달성"처럼 이 평가가 보장하지 않는 예측값이 되므로 아예 받지 않는다.
    """
    if not isinstance(text, str):
        return False
    return any(_canon(tok) is not None for tok in _NUMBER_RE.findall(_CTX_TOKEN_RE.sub(" ", text)))


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


# ── 판정 모순 검사 (ISSUES.md G-05) ────────────────────────────────────────
#
# grounding 은 **숫자 토큰만** 대조한다. 그래서 "모든 시험항목이 합격 기준을 충족하였다"
# 처럼 수치가 없는 문장은 검증 대상이 아니었고 **무조건 통과**했다. 판정이 FAIL 인
# 성적서에 그 문장이 실리면 독자는 정반대 결론을 읽는다 — 숫자가 없다는 이유로 가장
# 위험한 문장이 무검증으로 통과한 것이다.
#
# **범위를 좁게 잡는다.** 일반적인 '금지 표현 사전'은 문체 취향의 문제라 여기서 다루지
# 않는다. 이 검사는 오직 **강제된 판정(verdict)과 직접 모순되는 주장**만 본다 — 판정은
# fact_sheet 값으로 이미 강제되므로(LLM echo 무시) 그것과 어긋나는 문장은 정의상 근거가
# 없다. 문체가 아니라 사실 관계다.
#
# 한계: 완곡한 표현("아쉬움이 남는다")이나 영어 서술은 잡지 못한다. 서술의 정성적
# 품질 전반을 보증하지 않는다.

# "전부 통과했다"는 주장. FAIL·CONDITIONAL_PASS 판정과 모순된다.
_CLAIMS_ALL_PASSED = re.compile(
    r"(모든|전\s*항목|전체)[^.]{0,20}(충족|합격|통과)(하|되|였|했|함|됨)"
    r"|목표\s*성능을\s*달성"
    r"|기준을\s*모두\s*(충족|만족)"
)

# "미달했다"는 주장. PASS 판정과 모순된다.
# 단, "미달 가능성·미달 위험·미달할 우려"는 미달했다는 주장이 아니다. 아슬아슬 합격(near_target, D12)을
# 8.3 에 쓰면 합격 성적서에도 자연히 나오는 말이라, 이것까지 잡으면 맞는 서술이 폴백으로 강등된다.
# 미달·불합격·미충족 바로 뒤에 다음이 오면 건너뛴다. 같은 글에 진짜 미달 주장이 따로 있으면 그쪽에서 여전히 걸린다.
# - (할·될) 가능성·위험·우려·리스크 / 할·될 수(도) 있다
# - (을·를) 예방·방지 — 권고의 이름("기준 미달을 예방", "미달 방지")
# - (으)로 + 앞으로 일어날 일을 말하는 꼴(받침이 ㄹ 인 관형형: 될·바뀔·떨어질) + 수(도) 있다 / 가능성·위험·우려·리스크
#   — "불합격으로 판정될 수 있다", "미달로 바뀔 수 있다". "미달로 판정되어 위험이 크다"·"미달로 확인되어 우려된다"
#   처럼 이미 일어난 일(되어·돼·나타나·이어져)을 말한 뒤 위험을 덧붙인 꼴은 진짜 미달 주장이라 그대로 잡는다.
# "충족하지 못·이르지 못" 뒤에 "할 수(도) 있다·할 가능성"이 와도 가정이다("충족하지 못할 수도 있다").
_MAYBE = r"(?:수\s*(?:도\s*)?있|가능성|위험|우려|리스크)"


def _has_rieul_final(ch: str) -> bool:
    """한글 음절의 받침이 ㄹ 인지 — 유니코드 음절은 (코드 − 0xAC00) % 28 이 받침 번호이고 ㄹ 은 8 이다."""
    code = ord(ch) - 0xAC00
    return 0 <= code < 11172 and code % 28 == 8


# 받침이 ㄹ 인 음절 전부(될·할·바뀔·떨어질의 '질'·내려갈의 '갈' …)를 문자 집합으로 만든다.
_RIEUL_FINAL = "[" + "".join(chr(c) for c in range(0xAC00, 0xD7A4) if _has_rieul_final(chr(c))) + "]"
_RIEUL_FINAL_WORD = rf"[가-힣]{{0,4}}{_RIEUL_FINAL}"
# 받침이 ㄹ 이어도 "미달로 볼·판단할·평가할 수 있다"는 앞일이 아니라 지금 결과를 미달로 본다는 주장이다.
# 능동 판단 동사는 가정에서 뺀다("판정될 수 있다"처럼 피동은 앞일이라 그대로 건너뛴다).
_JUDGING_VERB = r"(?:볼|판단할|평가할|간주할|해석할|판정할|분류할|확인할|보아야\s*할)"
_HYPOTHETICAL = (
    r"(?!\s*(?:할|될)?\s*(?:가능성|위험|우려|리스크)"
    r"|\s*(?:할|될)\s*수\s*(?:도\s*)?있"
    r"|\s*(?:을|를)?\s*(?:예방|방지)"
    rf"|\s*(?:으로|로)\s*(?!{_JUDGING_VERB}\s*수){_RIEUL_FINAL_WORD}\s*{_MAYBE})"
)
_CLAIMS_SOME_FAILED = re.compile(
    rf"(미달|불합격|미충족){_HYPOTHETICAL}"
    rf"|충족하지\s*못(?!할\s*{_MAYBE})"
    rf"|기준에\s*이르지\s*못(?!할\s*{_MAYBE})"
)


_PARTIAL_COUNT = re.compile(r"(\d+)\s*개[^.\d]{0,10}?\s중\s*(\d+)\s*개")


def _is_partial_count(claim: str) -> bool:
    """'전체 5개 지표 중 3개가 통과'처럼 일부만 통과했다는 말인지. 'N개 중 N개'(전부)는 그대로 주장으로 본다.

    2026-10-19 Claude 실호출에서 조건부 합격 서술의 '전체 5개 지표 중 3개가 통과하였으며'를 '전체 … 통과'
    (전부 통과 주장)로 읽어 판정 모순 폴백을 냈다.
    """
    m = _PARTIAL_COUNT.search(claim)
    return bool(m) and m.group(1) != m.group(2)


def find_verdict_contradictions(texts: list[str], verdict: str) -> list[str]:
    """강제된 판정과 직접 모순되는 문장을 돌려준다(없으면 빈 목록).

    - FAIL / CONDITIONAL_PASS / INCONCLUSIVE 인데 "모두 충족했다" → 모순
      (판정 불가는 확인하지 못했다는 뜻이다. 다 충족했다고 쓰면 확인한 것처럼 읽힌다.)
    - PASS 인데 "미달했다" → 모순
    """
    contradictions: list[str] = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        if verdict in ("FAIL", "CONDITIONAL_PASS", "INCONCLUSIVE") and any(
            not _is_partial_count(m.group(0)) for m in _CLAIMS_ALL_PASSED.finditer(text)
        ):
            contradictions.append(text)
        elif verdict == "PASS" and _CLAIMS_SOME_FAILED.search(text):
            contradictions.append(text)
    return contradictions
