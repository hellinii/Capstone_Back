"""
narrator.py — LLM 서술 생성 오케스트레이션 + 환각 방어(grounding).

핵심 불변식:
  1. LLM 은 fact_sheet 안의 숫자만 사용한다. 출력의 모든 숫자는 화이트리스트로 검증한다.
  2. 위반(화이트리스트 밖 숫자) 발생 시 LLM 산문을 폐기하고 규칙 폴백으로 대체한다.
  3. 파생 계산(오분류 합계, 분포 백분율 등)은 서버가 미리 수행한다(LLM 은 계산하지 않음).

이 파일의 compute_derived/build_number_whitelist/verify_grounding 는 순수 함수로,
LLM·API 키 없이 단위 테스트 가능하다. (LLM 호출부는 generate_narrative)
"""
import json
import logging
import re
from typing import Any

from schemas import (
    FactSheet,
    GroundingInfo,
    NarrativeRequest,
    NarrativeResponse,
    InterpretationOut,
    ConclusionOut,
    RecommendationNarrativeOut,
    RecommendationOut,
    NarrativeMeta,
)
from benchmark_baselines import build_benchmark_refs
from narrative_fallback import build_fallback_narrative
from narrative_prompt import build_system_prompt, build_user_prompt, build_response_schema

_MODEL = "gpt-4.1-nano"

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


def _find_pos_idx(labels: list, positive_class) -> int:
    """2x2 혼동행렬에서 양성 클래스의 행/열 index(0|1)를 찾는다.

    labels 는 정렬된 클래스 문자열, positive_class 는 metadata 의 양성값.
    직접 → 정규화(strip/casefold) → float 비교 순으로 매칭하고("1.0" vs "1" 등),
    못 찾으면 index 1 로 폴백한다(평가기 'sorted-last=positive' 규칙 = 기존 동작 하위호환).
    """
    if not positive_class or not labels or len(labels) != 2:
        return 1
    norm = [str(l).strip() for l in labels]
    p = str(positive_class).strip()
    if p in norm:
        return norm.index(p)
    low = [x.casefold() for x in norm]
    if p.casefold() in low:
        return low.index(p.casefold())
    try:
        pf = float(p)
        floats = [float(x) for x in norm]
        if pf in floats:
            return floats.index(pf)
    except ValueError:
        pass
    return 1


def compute_derived(fs: FactSheet) -> dict:
    """LLM·폴백이 공통으로 인용할 파생 사실(서버 계산). 환각 방지를 위해 덧셈은 여기서만."""
    derived: dict = {}

    if fs.confusion and fs.confusion.matrix:
        m = fs.confusion.matrix
        total = sum(sum(row) for row in m)
        correct = sum(m[i][i] for i in range(len(m)) if i < len(m[i]))
        misclassified = total - correct
        conf = {
            "total": total,
            "correct": correct,
            "misclassified": misclassified,
        }
        # 흔히 인용되는 파생 백분율도 서버가 미리 계산(정확값)해 화이트리스트에 포함시킨다.
        # → LLM 이 오분류율 등을 인용해도 grounding 통과(틀린 비율은 여전히 차단됨).
        if total > 0:
            conf["correct_pct"] = round(correct / total * 100, 1)
            conf["misclassified_pct"] = round(misclassified / total * 100, 1)
        # 2x2(binary)면 FN/FP 및 그 비율도 제공 (행=실제, 열=예측).
        # positive 클래스의 실제 index 를 근거로 매핑한다(index 1 하드코딩 금지).
        if len(m) == 2 and len(m[0]) == 2:
            pos_idx = _find_pos_idx(fs.confusion.labels, fs.confusion.positive_class)
            neg_idx = 1 - pos_idx
            tp = m[pos_idx][pos_idx]
            tn = m[neg_idx][neg_idx]
            fp = m[neg_idx][pos_idx]  # 실제=음성, 예측=양성
            fn = m[pos_idx][neg_idx]  # 실제=양성, 예측=음성
            conf.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
            if total > 0:
                conf.update({
                    "tn_pct": round(tn / total * 100, 1),
                    "fp_pct": round(fp / total * 100, 1),
                    "fn_pct": round(fn / total * 100, 1),
                    "tp_pct": round(tp / total * 100, 1),
                })
        derived["confusion"] = conf

    if fs.distribution and fs.distribution.class_distribution:
        dist = fs.distribution.class_distribution
        total = sum(dist.values())
        derived["distribution"] = {
            "total": total,
            "percentages": {
                k: round((v / total) * 100, 1) if total > 0 else 0.0
                for k, v in dist.items()
            },
        }

    # 판정 카운트(통과율 근거) — 폴백/LLM 이 "N개 중 M개 통과"를 인용할 때 사용
    target = sum(1 for m in fs.metrics if m.threshold)
    passed = sum(1 for m in fs.metrics if m.status == "pass" and m.threshold)
    derived["counts"] = {
        "target": target,
        "passed": passed,
        "n_metrics": len(fs.metrics),
    }

    return derived


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


async def generate_narrative(client, req: NarrativeRequest) -> NarrativeResponse:
    """
    LLM 서술 생성 진입점.
    - client is None(무키) / LLM 호출 실패 / grounding 위반 → 규칙 폴백.
    - verdict 는 항상 fact_sheet 값으로 강제(LLM echo 무시).
    """
    fs = req.fact_sheet
    derived = compute_derived(fs)
    benchmark_refs = build_benchmark_refs(req.task_type.value, fs.metrics)
    whitelist = build_number_whitelist(fs, benchmark_refs, derived)

    # 1. 무키 → 폴백
    if client is None:
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="no_key")

    # 2. LLM 호출
    try:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt(req.report_purpose.value)},
                {"role": "user", "content": build_user_prompt(fs.model_dump(), benchmark_refs, derived)},
            ],
            response_format=build_response_schema(),
            temperature=0,
            seed=4213,
        )
        data = json.loads(response.choices[0].message.content)
    except Exception:
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="api_error")

    # 3~5. grounding 검증 + 응답 조립. 스키마 불일치/누락으로 조립이 실패해도
    #      500 대신 규칙 폴백(assembly_error)으로 강등한다(D7[1]).
    try:
        interp = data.get("interpretation", {})
        concl = data.get("conclusion", {})
        rec_narr = data.get("recommendation_narrative", {})
        recs = data.get("recommendations", [])

        # LLM 이 생성한 모든 문자열 필드를 전수 수집해 검증(수동 나열 제거 — D3b).
        grounding = verify_grounding(_collect_grounding_texts(data), whitelist)

        # 위반 시 report_purpose 무관하게 폴백(internal fail-open 제거 — D3c).
        if not grounding.passed:
            fb = build_fallback_narrative(fs, benchmark_refs, derived, reason="grounding_failed")
            fb.meta.grounding = grounding  # 어떤 환각이 잡혔는지 추적성 보존
            return fb

        # 정상 — verdict 는 서버값으로 강제
        return NarrativeResponse(
            interpretation=InterpretationOut(**interp),
            conclusion=ConclusionOut(
                verdict=fs.verdict,
                benchmark=concl.get("benchmark", ""),
                narrative=concl.get("narrative", ""),
                risks=concl.get("risks", ""),
            ),
            recommendation_narrative=RecommendationNarrativeOut(**rec_narr),
            recommendations=[RecommendationOut(**r) for r in recs][:5],
            meta=NarrativeMeta(source="llm", model=_MODEL, grounding=grounding),
        )
    except Exception:
        logging.exception("narrative 조립/검증 실패 → 규칙 폴백(assembly_error)")
        return build_fallback_narrative(fs, benchmark_refs, derived, reason="assembly_error")
