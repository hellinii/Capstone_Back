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
_CTX_TOKEN_RE = re.compile(
    r"ISO/?IEC\s+TS\s*4213(?:\s*:\s*2022)?|4213\s*:\s*2022|\d+\s*절|제\s*\d+\s*종"
)

# 신뢰구간의 신뢰수준(서술 설계 D29). 평가 응답 confidence_intervals 의 level 과 같다.
CI_LEVEL = 0.95

# 아슬아슬 기준(서술 설계 D12) — 목표값과의 차이가 목표값의 3% 미만. 판정은 프론트가 하고(near_target),
# 서술이 "3%"라고 쓸 수 있게 near_target 지표가 있을 때만 허용한다.
NEAR_TARGET_SHARE = 0.03

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
        for tok in _NUMBER_RE.findall(_CTX_TOKEN_RE.sub(" ", obj)):
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
        _add(wl, m.value)
        if m.threshold is not None:
            _add(wl, m.threshold)
        # 95% 신뢰구간(D29) — None 이면 _add 가 건너뛴다.
        _add(wl, m.ci_low)
        _add(wl, m.ci_high)
        _add(wl, m.margin)  # 목표값과의 차이(D12) — None 이면 _add 가 건너뛴다
    # 구간이 실렸을 때만 신뢰수준(95%)을 허용한다 — 평가 응답의 level 과 같은 값이다.
    if any(m.ci_low is not None and m.ci_high is not None for m in fs.metrics):
        _add(wl, CI_LEVEL)
    if any(m.near_target for m in fs.metrics):
        _add(wl, NEAR_TARGET_SHARE)

    for pc in fs.per_class:
        for val in (pc.precision, pc.recall, pc.f1, pc.support):
            _add(wl, val)
        if pc.label and pc.label.replace(".", "").isdigit():
            _add(wl, pc.label)

    for row in fs.confusion_counts:
        for val in (row.tp, row.fp, row.fn, row.tn):
            _add(wl, val)

    for avg in fs.class_averages:
        for val in (avg.precision, avg.recall, avg.f1, avg.support):
            _add(wl, val)

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
        for val in (gain.max_f1, gain.current_f1, gain.gain, gain.best_threshold):
            _add(wl, val)
        _add(wl, THRESHOLD_GAIN_NOTABLE)

    # 불균형 영향(D20) — 비율·두 차이(음수일 수 있다)와 기준 1.5·0.10. 숫자 꼴 클래스 이름("2")도 허용한다.
    ie = fs.imbalance_effect
    if ie:
        for val in (ie.imbalance_ratio, ie.accuracy_minus_macro_f1, ie.majority_minus_minority_f1):
            _add(wl, val)
        _add_client_numbers(wl, ie.majority_class)
        _add_client_numbers(wl, ie.minority_class)
        _add(wl, IMBALANCE_MILD_BELOW)
        _add(wl, IMBALANCE_F1_GAP)

    if fs.latency and fs.latency.available:
        for val in (fs.latency.mean, fs.latency.p50, fs.latency.p95, fs.latency.p99):
            _add(wl, val)

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
# 미달·불합격·미충족 바로 뒤(사이에 '할·될'과 공백만)에 가능성·위험·우려가 오거나 "할 수 있다"가
# 오면 건너뛴다. 같은 글에 진짜 미달 주장이 따로 있으면 그쪽에서 여전히 걸린다.
_HYPOTHETICAL = r"(?!\s*(?:할|될)?\s*(?:가능성|위험|우려)|\s*(?:할|될)\s*수\s*(?:도\s*)?있)"
_CLAIMS_SOME_FAILED = re.compile(
    rf"(미달|불합격|미충족){_HYPOTHETICAL}"
    r"|충족하지\s*못"
    r"|기준에\s*이르지\s*못"
)


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
        if verdict in ("FAIL", "CONDITIONAL_PASS", "INCONCLUSIVE") and _CLAIMS_ALL_PASSED.search(text):
            contradictions.append(text)
        elif verdict == "PASS" and _CLAIMS_SOME_FAILED.search(text):
            contradictions.append(text)
    return contradictions
