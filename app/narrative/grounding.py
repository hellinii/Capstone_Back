"""app/narrative/grounding.py — 환각 방어(grounding) 순수 로직

LLM 출력의 모든 숫자가 fact_sheet/파생값 화이트리스트 안에 있는지 검증한다. 위반 시
호출부(narrator)가 규칙 폴백으로 대체한다. 숫자 화이트리스트 구성·검증·문자열 수집을 담당.

상호작용
- 의존(import): re, typing.Any, app.narrative.schemas(FactSheet, GroundingInfo),
  app.narrative.derived(THRESHOLD_GAIN_NOTABLE), app.narrative.required_items(benchmark_value_text — 8.1 모델 값 표기),
  app.evaluation.metrics.counts(불균형 기준값)
- 사용처: app.narrative.legacy·generator·checks(허용 목록·판정 모순 검사), tests
"""
import re
from typing import Any

# 불균형 영향 기준(D20)은 평가 엔진이 판정에 쓴 값을 그대로 가져온다(값마다 한 곳).
from app.evaluation.metrics.counts import IMBALANCE_F1_GAP, IMBALANCE_MILD_BELOW
from app.narrative.derived import THRESHOLD_GAIN_NOTABLE
from app.narrative.required_items import benchmark_value_text
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
# 천 단위 쉼표는 세 자리 묶음만 받는다 — "1,234.5" 는 숫자 하나, 붙여 쓴 목록 "43,52" 는 숫자 둘(S5 검토 checks-12).
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[+-]?(?:\d{1,3}(?:,\d{3})+(?![\d])|\d+)(?:\.\d+)?%?")


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


# facts 안에서 숫자를 뽑지 않는 칸 — 계약 이름("facts-1")·지표 번호·소절·고정 문구 번호 같은 구조용 글자다.
# 이것을 훑으면 '1'·'17'·'7.1' 같은 숫자가 언제나 허용돼 숫자 검문에 구멍이 난다(S4b 검토 G-1). 고정 문장
# (fixed_sentences)은 코드가 찍는 글이라 그 숫자는 재료 칸에 이미 있다(계약 decisions — 따로 넣지 않는다).
_FACT_SKIP_KEYS = frozenset({
    "contract", "fixed_sentences", "selected_metrics", "metric_id", "report_purpose_label",
    "core_metrics_source", "class_share_basis", "distribution_shift_status", "inconclusive_cause",
    "unit", "kind", "side", "class_priority", "average", "risk_direction", "usage_mode",
})
# 값과 그 서술용 표기의 짝(판정과 같은 쪽으로 읽히게 자릿수를 늘린 표기) — 표기가 길면 짧은 반올림은 넣지 않는다.
_FACT_TEXT_PAIRS = {"share": "share_text", "gap_from_macro": "gap_text", "difference": "difference_text"}


def _add_fact_numbers(wl: set, obj: Any) -> None:
    """서술용 파생값(facts) 안의 숫자 — 값(음수면 크기도), 이름·사유 글 속 숫자. bool·구조용 칸은 건너뛴다."""
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
        paired_texts = {t for k, t in _FACT_TEXT_PAIRS.items() if k in obj and isinstance(obj.get(t), str)}
        for key, value in obj.items():
            # 짝으로 넣은 표기 칸은 다시 훑지 않는다 — 글로 훑으면 짧은 반올림('−0.1')이 다시 들어간다.
            if key in _FACT_SKIP_KEYS or key in paired_texts:
                continue
            text_key = _FACT_TEXT_PAIRS.get(key)
            text = obj.get(text_key) if text_key else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(text, str):
                _add_judged(wl, value, text, 4)
                _add_judged(wl, abs(value), text.lstrip("-"), 4)
                continue
            _add_fact_numbers(wl, value)
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
    *,
    edition: int,
) -> set:
    """출력 검증용 허용 숫자 집합 (fact_sheet + 파생값 + 고객 입력의 다중 표기).

    edition 은 성적서 양식 판(3 | 4 — 요청의 report_edition). 판 4 숫자 규칙은 판 4 라운드 S2 ⑥ 이 이 함수에 더한다."""
    wl: set = set()

    _add(wl, fs.n_samples)
    _add(wl, fs.dropped_rows)
    _add(wl, fs.score)

    for m in fs.metrics:
        # 값은 판정과 같은 쪽으로 읽히는 인쇄 표기(value_text, 평소 셋째 자리)가 있으면 그것을 따른다.
        _add_judged(wl, m.value, m.value_text, 3)
        _add(wl, m.beta)  # F-beta 의 β — 서술이 "β=2" 라고 쓸 수 있다(None 이면 건너뛴다)
        _add_client_numbers(wl, m.display_name)  # 이름 속 숫자(F1·"Top-5" 등 — 식별자 뒤 숫자는 원래 안 뽑힌다)
        if m.threshold is not None:
            _add(wl, m.threshold)
        # 95% 신뢰구간(D29) — None 이면 _add 가 건너뛴다.
        _add(wl, m.ci_low)
        _add(wl, m.ci_high)
        # 목표값과의 차이(D12). "목표값보다 0.0288 낮다"처럼 부호 없이 크기만 적는 것도 같은 사실이다. 판정과 같은
        # 쪽으로 읽히는 표기(margin_text)가 있으면 그 규칙으로 — 미달 −0.00004 가 '0' 으로 통과하지 않게(S4b 검토 G-3).
        if m.margin is not None:
            _add_judged(wl, m.margin, m.margin_text, 4)
            _add_judged(wl, abs(m.margin), m.margin_text.lstrip("-") if m.margin_text else None, 4)
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
        # 차이를 만든 네 값(정확도·매크로 F1·다수/소수 F1) — 사실 시트에 실리면 인용할 수 있다. 판정과 무관한 평소 표기.
        for val, text in (
            (ie.accuracy, shown.accuracy if shown else None),
            (ie.macro_f1, shown.macro_f1 if shown else None),
            (ie.majority_f1, shown.majority_f1 if shown else None),
            (ie.minority_f1, shown.minority_f1 if shown else None),
        ):
            if val is not None:
                _add_judged(wl, val, text, 4)
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
        # 규칙 문장·사실 시트 보기의 모델 값 — 범위 끝과 같아 보이면 자릿수를 늘린 표기(0.7999612 → '0.79996'). :g·1~4자리
        # 반올림에 없는 꼴이라 따로 넣는다(늘린 표기만 — 세션 C 마지막 검토 #1).
        if "metric" in ref and "ref_low" in ref and "ref_high" in ref:
            shown = _canon(benchmark_value_text(fs.metrics, ref))
            if shown is not None:
                wl.add(shown)
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


# ── 판정 모순 검사 (G-05) ────────────────────────────────────────
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

# "전부 통과했다"는 주장. FAIL·CONDITIONAL_PASS·INCONCLUSIVE 판정과 모순된다.
# 주어: '모든 …'·'전 항목'·'전체 (…) 지표/항목'·'N개 지표 (모두)'·'지표는 모두'. '전체'는 주어가 지표 전체일 때만 본다 —
# "전체 정확도는 기준을 통과했으나"처럼 흔한 말이라(점검 BE-04).
# 동사: 충족·합격·통과·만족·달성·도달·상회 + 하/되/였/했/함/됨, 넘었다·웃돌았다(S5 검토 checks-4).
_PASS_VERB = r"(?:(?:충족|합격|통과|만족|달성|도달|상회)(?:하|되|였|했|함|됨)|넘(?:었|는|어|겼)|웃돌(?:았|아|고))"
_CLAIMS_ALL_PASSED = re.compile(
    rf"(모든|전\s*항목|전체\s*(?:\d+\s*개|(?:\S+\s*)?(?:지표|항목|판정\s*대상)))[^.]{{0,20}}{_PASS_VERB}"
    r"|(모든|전\s*항목|전부)[^.]{0,20}목표값?\s*(이상|이하)(이다|이었|입니다|으로|이며|이고)"
    # '모두'가 주어 뒤에 오는 꼴 — "판정 대상 지표는 모두 목표를 충족하였다", "5개 지표 모두 목표값을 넘었다".
    rf"|(지표|항목)(?:은|는|이|가|들은|들이)?\s*모두\s*[^.]{{0,12}}{_PASS_VERB}"
    # "목표 성능을 충족한 지표도 있으나"처럼 관형절이면 주장이 아니다(검토 checks-8).
    r"|목표\s*성능을\s*(달성|충족)(?!한\s*(?:지표|항목))"
    r"|기준을\s*모두\s*(충족|만족)"
)
# '모두 충족' 주장의 주어를 좁히는 말 — 주어 자리에 바로 붙은 꼴만 받는다(검토 checks-3: "나머지 문제는 없으며 모든 지표가
# 충족"·"핵심 지표를 포함한 모든 지표가 충족"까지 빠지던 것).
# 조건부 합격: "핵심 지표는 기준을 모두 충족하였으나 보조 지표가 미달" 은 판정 규칙(D19 ③) 그대로다.
# 최종 불합격: "그 외 모든 지표는 기준을 충족했다" 는 핵심 미달과 함께 참일 수 있다.
_CORE_SUBJECT = re.compile(r"핵심\s*지표(?:는|은|가|이|들은|들이)\s")
_NONCORE_SUBJECT = re.compile(
    r"(?:그\s*외|나머지|비핵심|핵심이\s*아닌|핵심\s*지표가\s*아닌|그\s*밖의)(?:의)?\s*(?:모든\s*|다른\s*)?(?:보조\s*)?(?:지표|항목)"
    r"|보조\s*지표(?:는|은|가|이|들은|들이|도)\s"
)
# 하나를 뺀 나머지 — "Recall 을 제외한 모든 지표가 충족"(검토 checks-8). 판정 셋 모두에서 전부 충족 주장이 아니다.
_EXCEPT_ONE = re.compile(r"(?:을|를|만)\s*(?:제외한|제외하고|제외하면|빼고|뺀|말고)")
# 판정 불가를 설명하는 문장 — "목표값이 설정된 지표는 모두 충족했으나 핵심 지표가 산출되지 않아 판정할 수 없다" 는 참이다.
# '판정 불가 사유가 없다'·'판정 불가가 아니라'처럼 부정으로 쓰인 말은 사유가 아니다(검토 checks-3).
_INCONCLUSIVE_REASON = re.compile(
    r"(?:산출되지\s*않|산출하지\s*못|산출에\s*실패|판정할\s*수\s*없|목표값이\s*(?:설정되지\s*않|없))"
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
# "충족하지 못·이르지 못" 뒤에 "할 수(도) 있다·할 가능성·할 경우"가 와도 가정이다("충족하지 못할 수도 있다").
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
    rf"|\s*(?:으로|로)\s*(?!{_JUDGING_VERB}\s*수){_RIEUL_FINAL_WORD}\s*{_MAYBE}"
    # 점검 BE-04·S5 검토 checks-6 의 오탐 — 조건("미달하면"·"미달 시"·"미달할 경우"·"미달로 바뀌는 경우")·
    # 물음("미달 여부"·"미달로 바뀌는지"·"미달인지")·근접("미달에 가까운")·규칙 설명("불합격으로 판정하는 규칙").
    # '판정한다'만 붙은 꼴("기준 미달로 판정한다")은 지금 결과의 단정이라 잡는다(검토 checks-2).
    r"|\s*(?:하|되|이)?면(?![가-힣])"
    r"|\s*시(?![가-힣])"
    r"|\s*(?:할|될|하는|되는)\s*경우"
    r"|\s*(?:으로|로)\s*바뀌는\s*(?:경우|지)"
    r"|\s*여부"
    r"|\s*(?:하는지|되는지|인지)"
    r"|\s*에\s*(?:가까|근접)"
    r"|\s*(?:으로|로)\s*판정하는\s*(?:규칙|기준|방식|경우))"
)
# 부정 — "미달 지표는 없다"·"미달하지 않았다"·"충족하지 못한 지표는 없다"(점검 BE-04)와 S5 검토 checks-5 의 꼴
# ("미달되지는 않았다"·"미달은 아니다"·"충족하지 못하는 핵심 지표는 없다"·"불합격 사유는 없다"·"미달 지표: 없음"·
# "미달한 지표가 0개"). 바로 뒤에 붙은 꼴만 본다 — "미달하였으며 개선 여지가 없다"처럼 멀리 떨어진 '없다'는 부정이 아니다.
_NEGATED_FAILURE = re.compile(
    r"(?:미달|미충족|불합격)(?:한|된|인)?\s*(?:지표|항목|시험\s*항목|핵심\s*지표|사유|사례)?\s*[:：]?\s*(?:은|는|이|가|도)?\s*"
    r"(?:하나도\s*|전혀\s*)?(?:없|0\s*(?:개|건))"
    r"|(?:미달|미충족)(?:하|되)지\s*(?:는|도)?\s*(?:않|아니)"
    r"|(?:미달|불합격|미충족)(?:은|이)\s*아니"
    r"|충족하지\s*못(?:한|하는)\s*(?:핵심\s*)?(?:지표|항목|시험\s*항목)\s*(?:은|는|이|가|도)?\s*(?:하나도\s*|전혀\s*)?없"
)
_CLAIMS_SOME_FAILED = re.compile(
    rf"(미달|불합격|미충족){_HYPOTHETICAL}"
    rf"|충족하지\s*못(?!할\s*{_MAYBE}|(?:할|하는)\s*경우)"
    rf"|기준에\s*이르지\s*못(?!할\s*{_MAYBE}|(?:할|하는)\s*경우)"
)
# 미달의 주어가 신뢰구간 끝이면 지표의 미달 주장이 아니다 — "신뢰구간 하한 0.78 은 목표값 0.8 에 미달한다"는 합격 성적서에서도
# 참이다(검토 checks-6, 8.2 는 신뢰구간을 쓰는 칸이다).
_CI_SUBJECT_BEFORE = re.compile(r"(?:신뢰\s*구간|하한|상한)[^,]{0,30}$")  # 문장 안에서만 본다(값의 소수점은 넘는다)


_PARTIAL_COUNT = re.compile(r"(\d+)\s*개[^.\d]{0,10}?\s중\s*(\d+)\s*개")


def _is_partial_count(claim: str) -> bool:
    """'전체 5개 지표 중 3개가 통과'처럼 일부만 통과했다는 말인지. 'N개 중 N개'(전부)는 그대로 주장으로 본다.

    2026-10-19 Claude 실호출에서 조건부 합격 서술의 '전체 5개 지표 중 3개가 통과하였으며'를 '전체 … 통과'
    (전부 통과 주장)로 읽어 판정 모순 폴백을 냈다.
    """
    m = _PARTIAL_COUNT.search(claim)
    return bool(m) and m.group(1) != m.group(2)


# 판정 이름을 직접 말한 주장 — "종합 판정은 X이다", "판정: X", "판정 결과 X이다", "X으로 판정되었다·결정되었다",
# "X 판정을 받았다", 그리고 종합 판정에만 쓰는 이름의 단정("최종 합격이다"·"조건부 합격이다"·"판정 불가이다")·모델이
# 주어인 단정("본 모델은 합격이다")(점검 BE-04·S5 검토 checks-1). X 가 사실 시트 판정과 다르면 모순이다.
# 주장이 아닌 것(검토 checks-7): 조건절 뒤("…미달하면 판정은 불합격이다"), 판정 종류의 나열("합격, 조건부 합격, …"),
# 부정("합격으로 판정되었다고 보기 어렵다"). "판정이 불합격으로 바뀔 수 있다"처럼 단정 꼴이 아닌 것도 아니다.
# 긴 이름부터 맞춘다(조건부 합격 ⊃ 합격, 불합격 ⊃ 합격).
_VERDICT_WORD = (
    r"(?:CONDITIONAL_PASS|INCONCLUSIVE|PASS|FAIL"
    r"|(?:최종\s*)?(?:조건부\s*(?:합격|적합)|판정\s*불가|불합격|부적합|합격|적합))"
)
_OVERALL_ONLY_WORD = r"(?:최종\s*(?:합격|불합격)|조건부\s*(?:합격|적합)|판정\s*불가)"
_Q = r"['\"‘’“”「」『』]?"
_ASSERT_END = r"(?=이다|이며|이고|이었|입니다|임(?![가-힣])|으로\s*(?:결정|확정|나왔|나타났)|로\s*(?:결정|확정|나왔|나타났)|에\s*해당|[.,;]|$)"
_EXPLICIT_VERDICT = re.compile(
    rf"판정\s*(?:은|이|결과는|결과가|결과|[:：])\s*{_Q}\s*(?P<a>{_VERDICT_WORD}){_Q}\s*{_ASSERT_END}"
    rf"|{_Q}(?P<b>{_VERDICT_WORD}){_Q}\s*(?:으로|로)\s*(?:최종\s*|종합\s*)?(?:판정|결정|확정)(?:되었|됐|되어|되며|하였|했|받았|을\s*받)"
    rf"|{_Q}(?P<c>{_VERDICT_WORD}){_Q}\s*판정을\s*받(?:았|은|아)"
    rf"|(?<![가-힣]){_Q}(?P<d>{_OVERALL_ONLY_WORD}){_Q}\s*(?=이다|이며|이고|이었|입니다)"
    rf"|(?:모델(?:은|이)|성적서(?:는|의\s*판정은)|결과(?:는|가))\s*{_Q}(?P<e>{_VERDICT_WORD}){_Q}\s*(?=이다|이며|이고|이었|입니다)"
)
_CONDITION_BEFORE = re.compile(r"(?:면|다면|경우|시)\s*,?\s*(?:판정\s*(?:은|이)\s*)?(?:최종\s*)?$")
_NEGATION_AFTER = re.compile(r"\s*(?:이다|이었다)?(?:다)?고\s*(?:보기|하기|단정하기)\s*어렵|\s*(?:이|가)\s*아니|\s*(?:은|는)\s*아니|지\s*않")
_LIST_AFTER = re.compile(rf"\s*{_Q}\s*[,·]\s*{_Q}\s*{_VERDICT_WORD}")


def _verdict_code(word: str) -> str:
    w = re.sub(r"\s+", "", word).removeprefix("최종")
    if w in ("CONDITIONAL_PASS", "조건부합격", "조건부적합"):
        return "CONDITIONAL_PASS"
    if w in ("INCONCLUSIVE", "판정불가"):
        return "INCONCLUSIVE"
    if w in ("FAIL", "불합격", "부적합"):
        return "FAIL"
    return "PASS"


def claimed_verdicts(text: str) -> list[str]:
    """글이 직접 단정한 판정(코드값) 목록 — 조건절·나열·부정은 뺀다."""
    out = []
    for m in _EXPLICIT_VERDICT.finditer(text):
        name = next(g for g in ("a", "b", "c", "d", "e") if m.group(g))
        if _CONDITION_BEFORE.search(text[:m.start()]) or _CONDITION_BEFORE.search(text[:m.start(name)]):
            continue
        tail = text[m.end(name):]
        if _NEGATION_AFTER.match(tail) or _LIST_AFTER.match(tail) or re.match(r"[^.]{0,6}다고\s*보기\s*어렵", text[m.end():]):
            continue
        if re.search(rf"{_VERDICT_WORD}\s*{_Q}\s*[,·]\s*{_Q}\s*$", text[:m.start(name)]):
            continue  # 나열의 뒤쪽 이름("합격, 조건부 합격, …")
        out.append(_verdict_code(m.group(name)))
    return out


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


def _all_passed_claim(sentence: str, verdict: str) -> bool:
    for m in _CLAIMS_ALL_PASSED.finditer(sentence):
        if _is_partial_count(m.group(0)):
            continue
        window = sentence[max(0, m.start() - 25):m.end()]
        if _EXCEPT_ONE.search(window):
            continue
        if verdict == "CONDITIONAL_PASS" and _CORE_SUBJECT.search(window) and not _NONCORE_SUBJECT.search(window) \
                and not re.search(r"핵심\s*지표(?:를\s*포함|뿐|와|과|및)", window):
            continue
        if verdict == "FAIL" and _NONCORE_SUBJECT.search(window):
            continue
        if verdict == "INCONCLUSIVE":
            reason = _INCONCLUSIVE_REASON.search(sentence)
            if reason and not re.match(r"[^.]{0,6}(?:사유가\s*없|것이\s*아니|가\s*아니)", sentence[reason.end():]):
                continue
        if re.match(r"[^.]{0,12}(?:것은\s*아니|것이\s*아니|지\s*않|지\s*못)", sentence[m.end():]):
            continue  # "모든 지표가 충족된 것은 아니다"
        if re.match(r"\s*(?:었|았)?(?:다)?면(?![가-힣])|[^.]{0,3}(?:경우|시(?![가-힣]))", sentence[m.end():]):
            continue  # 조건 — "핵심 지표가 모두 충족되면 판정은 합격이다"(규칙 설명)
        return True
    return False


# 규칙을 말하는 조건문 — "핵심 지표가 하나라도 미달하면 최종 불합격"·"미달할 경우 불합격으로 판정된다" 의 '불합격'은 이
# 성적서의 판정 주장이 아니다.
_RULE_CONDITION_BEFORE = re.compile(r"(?:면|경우|시)\s*,?\s*(?:최종\s*)?$")


def _some_failed_claim(sentence: str) -> bool:
    negated = [m.span() for m in _NEGATED_FAILURE.finditer(sentence)]
    for m in _CLAIMS_SOME_FAILED.finditer(sentence):
        if any(a <= m.start() < b for a, b in negated):
            continue
        before = sentence[:m.start()]
        if _RULE_CONDITION_BEFORE.search(before) or _CI_SUBJECT_BEFORE.search(before):
            continue
        return True
    return False


def find_verdict_contradictions(texts: list[str], verdict: str) -> list[str]:
    """강제된 판정과 직접 모순되는 글을 돌려준다(없으면 빈 목록). 문장마다 본다.

    - 판정 이름을 직접 단정했는데(종합 판정은 X이다·X으로 판정되었다·본 모델은 X이다) X 가 판정과 다르다 → 모순.
    - FAIL / CONDITIONAL_PASS / INCONCLUSIVE 인데 "모두 충족했다" → 모순. 단 주어가 좁혀진 주장은 뺀다 — 조건부 합격의
      "핵심 지표는 모두 충족", 불합격의 "그 외 지표는 모두 충족", 하나를 뺀 "X 를 제외한 모든 지표", 판정 불가의
      "…충족했으나 핵심 지표가 산출되지 않아". (판정 불가는 확인하지 못했다는 뜻이다. 다 충족했다고 쓰면 확인한 것처럼 읽힌다.)
    - PASS 인데 "미달했다" → 모순. 부정("미달 지표는 없다")·가정·조건("미달하면"·"미달 시")·물음("미달 여부")·
      신뢰구간 끝의 미달은 뺀다.
    회귀 문장 표: tests/test_s5_checks.py(점검 BE-04 의 19문장, S5 검토 문장 포함).
    """
    contradictions: list[str] = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        hit = False
        for sentence in _SENTENCE_SPLIT.split(text):
            if not sentence.strip():
                continue
            if any(code != verdict for code in claimed_verdicts(sentence)):
                hit = True
            elif verdict in ("FAIL", "CONDITIONAL_PASS", "INCONCLUSIVE") and _all_passed_claim(sentence, verdict):
                hit = True
            elif verdict == "PASS" and _some_failed_claim(sentence):
                hit = True
            if hit:
                break
        if hit:
            contradictions.append(text)
    return contradictions
