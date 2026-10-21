"""app/narrative/required_items.py — 필수 서술 항목 목록(서술 설계 5절)과 결론 목록(D7).

목록은 하나다 — AI 지시문(S5)과 규칙 템플릿(S6)이 같은 목록을 읽는다(계획 원칙 2). 심판은 이 함수를 쓰지 않는다
(사실 시트·파생값만 공유 — 서술 설계 8절).

- REQUIRED_ITEMS: 5절 41행을 코드 표로 — 항목 ID · 칸 · 작업 유형 · 재료 조건 · 필요한 값(결론을 만드는 함수) ·
  작성 주체 · 생성 뒤 확인(D24 숫자).
- select_items(사실 시트, 고객 입력, 작업 유형): 이 성적서에 해당하는 항목만 남기고, 숫자를 채운 결론을 돌려준다.
  숫자는 받은 사실(facts·지표·엔진 값)과 서버 파생값(혼동행렬 합계·백분율)뿐이다 — 여기서 새로 판정하지 않는다.

작성 주체(Author):
- ai: AI 가 쓴다(규칙 템플릿은 결론을 문장으로).
- ai_fact: AI 가 해석을 쓰고, 규칙 템플릿은 사실 문장만(해석하지 않는다 — 모델 용도·최약 클래스 위험).
- fixed_interpret: 코드가 고정 문구(17·18)로 숫자를 찍고, AI·템플릿은 그 숫자를 되풀이하지 않고 해석만.
- fixed: 코드가 고정 문구로 찍는다(서술 칸에 쓰지 않는다).
- form: 양식이 의뢰자 선언·선택으로 조립한다(5절·6.1 — 서술 칸에 쓰지 않는다).

계획 결정 반영(00_계획.md 사용자 결정·설계 요지 3):
- 최약 클래스·주요 오류(7.1)는 M21·M22 선택과 무관하게 '항상'(결정 A).
- 8.2 신뢰구간 해석은 판정 대상 지표에 구간이 하나라도 있으면 — 목표 포함 / 모두 떨어짐 두 갈래. 판정 대상 전부가
  생략이면 생략 사유를 쓴다.
- 7.1 오류 성격·8.3 근소 차이는 고정 문구 17·18 + 해석. 8.2 평균 방식·지표 선택 기준은 양식.

상호작용
- 의존(import): app.narrative.schemas, app.narrative.derived(compute_derived), app.narrative.baselines, app.narrative.sections
- 사용처: (S5) 지시문의 칸별 "반드시 다룰 결론", (S6) 규칙 템플릿, (S5) D24 생성 뒤 확인, tests
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.schemas import FactSheet
from app.narrative.sections import SECTION_IDS

Author = Literal["ai", "ai_fact", "fixed_interpret", "fixed", "form"]
TaskType = Literal["binary", "multiclass", "multilabel"]
ALL: frozenset[str] = frozenset({"binary", "multiclass", "multilabel"})
BINARY: frozenset[str] = frozenset({"binary"})
MULTICLASS: frozenset[str] = frozenset({"multiclass"})
MULTILABEL: frozenset[str] = frozenset({"multilabel"})
MULTI: frozenset[str] = frozenset({"multiclass", "multilabel"})


@dataclass(frozen=True)
class Conclusion:
    """항목 하나의 결론 — 숫자를 채운 사실 문장(한국어). 지시문은 '반드시 다룰 결론'으로, 템플릿은 문장으로 쓴다."""
    item_id: str
    section: str
    author: Author
    text: str
    # D24 생성 뒤 확인: 그 칸의 글에 이 값 중 하나 이상(허용 목록과 같은 표기 변형)이 있어야 한다. D24 항목만.
    d24_values: tuple[float, ...] = ()
    # 7.3 처럼 결론 목록 밖에서 AI 가 찾는 칸(정해진 결론이 없다).
    open_ended: bool = False


@dataclass
class Ctx:
    fs: FactSheet
    task_type: str
    client_inputs: dict[str, Any]
    derived: dict
    benchmark_refs: list[dict]
    selected: set[str] = field(default_factory=set)

    @property
    def facts(self):
        return self.fs.facts

    def fixed(self, sentence_id: str):
        if self.facts is None:
            return None
        return next((s for s in self.facts.fixed_sentences if s.id == sentence_id), None)


@dataclass(frozen=True)
class RequiredItem:
    id: str
    section: str
    title: str
    task_types: frozenset[str]
    author: Author
    material: str  # 재료 조건(사람이 읽는 설명 — 5절의 '재료 조건' 칸)
    build: Callable[[Ctx], "str | None"]  # 결론 문장. None 이면 이 성적서에 해당 없음
    d24: Callable[[Ctx], tuple[float, ...]] | None = None
    open_ended: bool = False


# ── 숫자 표기 ──────────────────────────────────────────────────────────────────


def _n(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _pct(ratio: float) -> str:
    return f"{round(ratio * 100, 1):g}%"


def _pct_text(fraction_text: str) -> str:
    """판정 쪽으로 자릿수를 늘린 비율 표기(0~1, 예 '0.04996')를 같은 정밀도의 백분율로('4.996%').

    _pct(소수 첫째 자리)로 바꾸면 0.04996 이 '5%' 가 되어 '5% 미만' 판정과 어긋난다(S4b 검토 RI-01).
    """
    digits = max(len(fraction_text.split(".", 1)[1]) - 2, 1) if "." in fraction_text else 1
    return f"{round(float(fraction_text) * 100, digits):g}%"


def _ro(number_text: str) -> str:
    """숫자 뒤 조사 '로/으로' — 끝자리 0·3·6 은 받침(ㅇ·ㅁ·ㄱ)이라 '으로', 1·7·8 은 ㄹ받침이라 '로', 나머지 '로'."""
    last = number_text.rstrip("%")[-1:] if number_text else ""
    return "으로" if last in "036" and last else "로"


def _model_purpose(ctx: Ctx) -> str | None:
    model = ctx.client_inputs.get("model") if isinstance(ctx.client_inputs, dict) else None
    purpose = model.get("모델 용도") if isinstance(model, dict) else None
    return purpose.strip() if isinstance(purpose, str) and purpose.strip() else None


# ── 7.1 혼동행렬 해석 ──────────────────────────────────────────────────────────


def _misclassified(ctx: Ctx) -> str | None:
    conf = ctx.derived.get("confusion")
    if not conf:
        return None
    unit = "칸" if conf.get("unit") == "label_cell" else "건"
    what = "틀린 레이블 칸은" if unit == "칸" else "오분류는"
    pct = f"({conf['misclassified_pct']:g}%)" if "misclassified_pct" in conf else ""
    return f"{what} 전체 {conf['total']}{unit} 중 {conf['misclassified']}{unit}{pct}이다."


def _binary_row(ctx: Ctx):
    rows = ctx.fs.confusion_counts
    return rows[0] if ctx.task_type == "binary" and rows else None


def _fp_fn(ctx: Ctx) -> str | None:
    row = _binary_row(ctx)
    if row is None:
        return None
    return f"양성 클래스({row.label}) 기준 거짓 양성(FP)은 {row.fp}건, 거짓 음성(FN)은 {row.fn}건이다."


def _declared_risk(ctx: Ctx) -> str | None:
    dr = ctx.facts.declared_risk_errors if ctx.facts else None
    if dr is None:
        return None
    side = "FN(놓침)은" if dr.side == "fn" else "FP(잘못 잡음)은"
    if dr.share is None:
        return f"의뢰자가 더 위험하다고 밝힌 {side} {dr.count}건이다(오분류가 없다)."
    return f"의뢰자가 더 위험하다고 밝힌 {side} {dr.count}건으로 전체 오류 {dr.total_errors}건의 {_pct(dr.share)}이다."


def _declared_risk_values(ctx: Ctx) -> tuple[float, ...]:
    dr = ctx.facts.declared_risk_errors if ctx.facts else None
    if dr is None:
        return ()
    return (float(dr.count),) + ((dr.share,) if dr.share is not None else ())


def _per_class_errors(ctx: Ctx, noun: str) -> str | None:
    rows = ctx.fs.confusion_counts
    if not rows:
        return None
    top = sorted(rows, key=lambda r: -(r.fp + r.fn))[:3]
    items = ", ".join(f"{r.label}(FP {r.fp}건, FN {r.fn}건)" for r in top)
    return f"오류(FP+FN)가 많은 {noun} {items}이다."


def _top_pairs(ctx: Ctx) -> str | None:
    pairs = ctx.facts.top_confusion_pairs if ctx.facts else []
    if not pairs:
        return None
    items = ", ".join(f"{p.actual}→{p.predicted} {p.count}건({_pct(p.share_of_errors)})" for p in pairs)
    return f"가장 많이 헷갈린 쌍(실제→예측)은 {items}이다(괄호는 전체 오분류 중 비율)."


def _bias(ctx: Ctx, unit: str) -> str | None:
    bias = ctx.facts.prediction_bias if ctx.facts else None
    if bias is None or bias.unit != unit or not bias.rows:
        return None
    if unit == "positive_class":
        r = bias.rows[0]
        lean = "많이" if r.difference > 0 else "적게" if r.difference < 0 else "같게"
        return (
            f"양성({r.label})으로 예측한 것은 {r.predicted}건, 실제 양성은 {r.actual}건으로 양성을 실제보다 "
            f"{abs(r.difference)}건 {lean} 예측했다(FP − FN)."
            if r.difference
            else f"양성({r.label})으로 예측한 것과 실제 양성이 {r.actual}건으로 같다(FP = FN)."
        )
    if unit == "label_total":
        r = bias.rows[0]
        return f"예측한 레이블은 모두 {r.predicted}개, 실제 레이블은 {r.actual}개다(차이 {r.difference})."
    top = sorted(bias.rows, key=lambda r: -abs(r.difference))[:3]
    if all(r.difference == 0 for r in top):
        return "클래스마다 예측한 수와 실제 수가 같다."
    items = ", ".join(f"{r.label}(예측 {r.predicted}건, 실제 {r.actual}건)" for r in top if r.difference)
    return f"예측 수와 실제 수의 차이가 큰 클래스는 {items}이다."


def _weakest(ctx: Ctx) -> str | None:
    w = ctx.facts.weakest_class if ctx.facts else None
    if w is None:
        return None
    noun = "레이블" if ctx.task_type == "multilabel" else "클래스"
    subject = "레이블은" if noun == "레이블" else "클래스는"
    base = "레이블 등장" if ctx.facts.class_share_basis == "label_occurrences" else "전체 표본"
    text = f"F1 이 가장 낮은 {subject} {w.label}(F1 {_n(w.f1)}, {w.support}건, {base}의 {_pct_text(w.share_text)})이다."
    if w.gap_text is not None:
        text += f" 매크로 F1 과의 차이는 {w.gap_text}이다."
    flags = []
    # 사유별로 따로 쓴다 — 'A 또는 B' 로 쓰면 걸리지 않은 쪽까지 주장하게 된다.
    if w.support < 30:
        flags.append("표본이 30건 미만이다")
    if float(w.share_text) < 0.05:  # share_text 는 프론트 판정과 같은 쪽으로 읽히게 만든 표기다
        flags.append(f"{base}의 5% 미만이다")
    if w.far_below_macro:
        flags.append("매크로 F1 보다 0.1 이상 낮다")
    if flags:
        text += " " + " ".join(flags[:1]) + "".join(f" 또한 {f}" for f in flags[1:]) + "."
    small = [r for r in ctx.facts.low_support_classes if r.label != w.label]
    if small:
        text += f" 그 밖에 표본이 작은 {noun}: {', '.join(f'{r.label}({r.support}건)' for r in small[:5])}."
    return text


def _error_nature(ctx: Ctx) -> str | None:
    if ctx.fs.error_ratios is None:
        return None
    return "경계·확신 오류 비율은 고정 문구 17 이 숫자로 찍는다 — 숫자를 되풀이하지 말고 오류의 성격만 해석한다."


def _derivation(ctx: Ctx) -> str | None:
    if ctx.fs.derived_prediction is None:
        return None
    sentence = ctx.fixed("9")
    return sentence.text if sentence else "예측값은 제출된 확률에서 파생했다(고정 문구 9)."


# ── 7.2 분포·불균형 ───────────────────────────────────────────────────────────


def _imbalance_ratio(ctx: Ctx) -> str | None:
    ie = ctx.fs.imbalance_effect
    if ie is not None:
        shown = ie.display.imbalance_ratio if ie.display else _n(ie.imbalance_ratio)
        text = f"불균형 비율은 {shown}이다."
    elif ctx.fs.distribution is not None and ctx.fs.distribution.imbalance_ratio is not None:
        text = f"불균형 비율은 {_n(ctx.fs.distribution.imbalance_ratio)}이다."
    else:
        return None
    if ctx.task_type == "multilabel":
        # AI·템플릿이 쓰는 결론에는 절·문구 번호를 넣지 않는다(숫자 검문 대상이다). 정의 문장은 성적서가 찍는다.
        text += " 멀티레이블 불균형 비율의 정의는 성적서가 고정 문구로 찍는다."
    return text


# 엔진 판정 규칙(D20)의 설명 — 이진·다중은 '정확도 − 매크로 F1' 과 '다수 − 소수 F1' 둘을, 멀티레이블은 다수·소수
# 레이블 F1 차 하나만 본다(S4b 검토 RI-08).
_IMBALANCE_LEVEL = {
    "affected": "불균형이 성능 수치에 영향을 주었다(비율 1.5 이상이고 {gaps} 0.1 이상)",
    "no_distortion": "불균형은 있으나 성능 왜곡은 관찰되지 않았다(비율 1.5 이상이고 {gaps_all} 0.1 미만)",
    "mild": "불균형이 경미하다(비율 1.5 미만)",
}


def _imbalance_effect(ctx: Ctx) -> str | None:
    ie = ctx.fs.imbalance_effect
    if ie is None:
        return None
    shown = ie.display
    gap = shown.majority_minus_minority_f1 if shown else _n(ie.majority_minus_minority_f1)
    if ie.accuracy_minus_macro_f1 is None:
        gaps = gaps_all = "다수·소수 레이블 F1 차이가"
    else:
        gaps, gaps_all = "정확도 − 매크로 F1 과 다수 − 소수 F1 차이 중 하나가", "정확도 − 매크로 F1 과 다수 − 소수 F1 차이가 모두"
    rule = _IMBALANCE_LEVEL[ie.level].format(gaps=gaps, gaps_all=gaps_all)
    text = f"엔진 판단: {rule}. 다수({ie.majority_class})−소수({ie.minority_class}) F1 차이는 {gap}"
    if ie.accuracy_minus_macro_f1 is not None:
        acc = shown.accuracy_minus_macro_f1 if shown and shown.accuracy_minus_macro_f1 else _n(ie.accuracy_minus_macro_f1)
        text += f", 정확도 − 매크로 F1 은 {acc}"
    return text + "이다."


def _distribution_shift(ctx: Ctx) -> str | None:
    facts = ctx.facts
    if facts is None or facts.distribution_shift_status != "compared" or not facts.distribution_shift:
        return None
    flagged = [r for r in facts.distribution_shift if r.flagged]
    if not flagged:
        return "학습 분포와 평가 분포의 클래스 비율 차는 모두 10%p 미만이다."
    items = ", ".join(
        f"{r.label}(학습 {_pct(r.training_ratio)}, 평가 {_pct(r.evaluation_ratio)})" for r in flagged
    )
    return f"학습과 평가의 비율 차가 10%p 이상인 클래스는 {items}이다."


def _label_caution(ctx: Ctx) -> str | None:
    sentence = ctx.fixed("12")
    return sentence.text if sentence else None


def _additional(ctx: Ctx) -> str | None:
    return "결론 목록 밖에서 사실 시트로 찾은 관찰(있으면 허용 숫자를 하나 이상 인용, 없으면 고정 문구 15 '해당 없음')."


# ── 8.1 벤치마크 ──────────────────────────────────────────────────────────────

_QUALITY = {"better": "우수", "within": "기준 범위 내", "worse": "미흡"}
_TARGET_POS = {"below": "참조 범위보다 낮음", "within": "참조 범위 안", "above": "참조 범위보다 높음"}


def _benchmark(ctx: Ctx) -> str | None:
    if not ctx.benchmark_refs:
        return None
    items = ", ".join(
        f"{r['metric']} {_n(r['model_value'])}(참조 범위 {_n(r['ref_low'])}~{_n(r['ref_high'])}, {_QUALITY.get(r['quality'], r['quality'])})"
        for r in ctx.benchmark_refs
    )
    return f"내부 참조 기준 대비 위치: {items}."


def _target_level(ctx: Ctx) -> str | None:
    refs = [r for r in ctx.benchmark_refs if r.get("target_value") is not None]
    if not refs:
        return None
    items = ", ".join(
        f"{r['metric']} 목표 {_n(r['target_value'])}({_TARGET_POS.get(r['target_position'], r['target_position'])})"
        for r in refs
    )
    return f"목표값 수준: {items}."


def _no_benchmark(ctx: Ctx) -> str | None:
    covered = {r["metric"] for r in ctx.benchmark_refs}
    missing = [m.display_name for m in _narrative_metrics(ctx.fs) if m.display_name not in covered]
    if not missing:
        return None
    return f"내부 참조 기준치가 없어 비교하지 않는 지표: {', '.join(missing)}(고정 문구 16)."


# ── 8.2 종합 총평 ─────────────────────────────────────────────────────────────

_VERDICT = {"PASS": "최종 합격", "CONDITIONAL_PASS": "조건부 합격", "FAIL": "최종 불합격", "INCONCLUSIVE": "판정 불가"}
_CAUSE = {
    "core_not_computed": "핵심 지표 산출 실패",
    "no_target": "핵심 지표 목표값 없음",
    "no_judged": "판정 대상 지표 없음",
}


def _verdict_rule(ctx: Ctx) -> str | None:
    return "판정 규칙(핵심 지표 미달 → 불합격, 산출 실패·판정 대상 없음 → 판정 불가 …)은 6.1 표에 고정 문구 10 으로 찍힌다."


def _verdict_reason(ctx: Ctx) -> str | None:
    vd = ctx.facts.verdict_detail if ctx.facts else None
    name = _VERDICT.get(ctx.fs.verdict, ctx.fs.verdict)
    if vd is None:
        return f"종합 판정은 '{name}'이다."
    parts = [f"종합 판정은 '{name}'이다."]
    if vd.failed_core:
        parts.append(f"미달한 핵심 지표: {', '.join(m.display_name for m in vd.failed_core)}.")
    if vd.inconclusive_cause:
        names = ", ".join(m.display_name for m in vd.inconclusive_metrics)
        parts.append(f"판정 불가 원인: {_CAUSE[vd.inconclusive_cause]}" + (f"({names})." if names else "."))
    if vd.unavailable_core and vd.inconclusive_cause != "core_not_computed":
        parts.append(f"산출에 실패한 핵심 지표: {', '.join(m.display_name for m in vd.unavailable_core)}.")
    if vd.failed_noncore:
        parts.append(f"미달한 비핵심 지표: {', '.join(m.display_name for m in vd.failed_noncore)}.")
    if vd.judged_count:
        parts.append(f"참고로 판정 대상 {vd.judged_count}개 중 {vd.passed_count}개가 목표를 충족했다(통과율은 판정 근거가 아니다).")
    return " ".join(parts)


def _ci(ctx: Ctx) -> str | None:
    judged = [m for m in _narrative_metrics(ctx.fs) if m.threshold is not None]
    with_ci = [m for m in judged if m.ci_low is not None and m.ci_high is not None]
    skipped = ctx.facts.ci_skipped if ctx.facts else []
    reasons = ", ".join(f"{s.display_name}({s.reason})" if s.reason else s.display_name for s in skipped)
    if with_ci:
        contains = set(ctx.derived.get("ci_contains_target") or [])
        inside = [m for m in with_ci if m.display_name in contains]
        partial = len(with_ci) < len(judged)  # 구간이 없는 판정 대상이 있다 — '모두'라고 보증하지 않는다(RI-02)
        if inside:
            items = ", ".join(f"{m.display_name}(구간 {_n(m.ci_low)}~{_n(m.ci_high)}, 목표 {_n(m.threshold)})" for m in inside)
            text = f"95% 신뢰구간이 목표값을 포함하는 지표: {items} — 표본이 달라지면 판정이 바뀔 수 있다."
        elif partial:
            names = ", ".join(m.display_name for m in with_ci)
            text = f"신뢰구간을 산출한 지표({names})는 구간이 목표값과 떨어져 있다."
        else:
            text = "판정 대상 지표의 95% 신뢰구간이 모두 목표값과 떨어져 있어 표본 변동으로 판정이 흔들리지 않는다."
        if partial:
            missing = [m.display_name for m in judged if m not in with_ci]
            text += f" 신뢰구간을 산출하지 않은 지표: {reasons or ', '.join(missing)}."
        return text
    if judged and skipped:
        return f"판정 대상 지표의 신뢰구간을 산출하지 않았다: {reasons}."
    return None


def _generalization(ctx: Ctx) -> str | None:
    sentence = ctx.fixed("11a") or ctx.fixed("11b")
    return sentence.text if sentence else None


def _pr_gap(ctx: Ctx) -> str | None:
    g = ctx.facts.precision_recall_gap if ctx.facts else None
    if g is None or not {"M2", "M3"} <= ctx.selected:
        return None
    lean = "Precision 이 높다" if g.difference > 0 else "Recall 이 높다" if g.difference < 0 else "둘이 같다"
    flag = "치우침이 있다(|차이| 0.1 이상)" if g.flagged else "치우침은 크지 않다(|차이| 0.1 미만)"
    return f"양성 클래스의 Precision {_n(g.precision)}, Recall {_n(g.recall)} — {lean}, 차이 {g.difference_text}{_ro(g.difference_text)} {flag}."


def _threshold_gain(ctx: Ctx) -> str | None:
    g = ctx.fs.f1_threshold_gain
    if g is None or not ctx.derived.get("threshold_tuning"):
        return None
    return (
        f"결정 임계값을 {_n(g.best_threshold)}{_ro(_n(g.best_threshold))} 바꾸면 F1 이 {_n(g.current_f1)}에서 {_n(g.max_f1)}까지 오를 수 있다"
        f"(향상폭 {g.gain_text or _n(g.gain)} — 같은 평가 데이터로 고른 값이라 별도 데이터로 확인해야 한다)."
    )


def _average_gap(ctx: Ctx, kind: str) -> str | None:
    g = ctx.facts.average_gap if ctx.facts else None
    if g is None or g.kind != kind or "M22" not in ctx.selected:
        return None
    other = "가중" if kind == "macro_vs_weighted" else "마이크로"
    ref = g.reference_class
    why = (
        f"support 가 가장 큰 클래스는 {ref.label}({ref.support}건, F1 {_n(ref.f1)})"
        if kind == "macro_vs_weighted"
        else f"F1 이 가장 낮은 레이블은 {ref.label}(F1 {_n(ref.f1)}, {ref.support}건)"
    )
    return f"매크로 F1 {_n(g.macro_f1)}, {other} 평균 F1 {_n(g.other_f1)}(차이 {g.difference_text}). {why}."


def _hamming_exact(ctx: Ctx) -> str | None:
    by_id = {m.metric_id: m for m in _narrative_metrics(ctx.fs)}
    if "M15" not in by_id or "M16" not in by_id:
        return None
    return f"Hamming Loss {_n(by_id['M15'].value)}, Exact Match {_n(by_id['M16'].value)} — 칸 단위 오류와 표본 단위 완전 일치를 함께 본다."


def _averaging_form(ctx: Ctx) -> str | None:
    decl = ctx.facts.declarations if ctx.facts else None
    if decl is None or decl.class_priority is None:
        return None
    return "평균 방식 선택 기준은 의뢰자 선언(클래스 중요도)으로 5절에 고정 문구 4 로 찍힌다."


def _declared_average(ctx: Ctx) -> str | None:
    d = ctx.facts.declared_average if ctx.facts else None
    if d is None or d.f1 is None:
        return None
    name = "가중 평균" if d.average == "weighted" else "마이크로 평균"
    why = "'다수 클래스 우선'으로" if d.class_priority == "larger_first" else "'샘플 단위로 중요'로"
    return f"의뢰자가 클래스 중요도를 {why} 밝혀 {name} F1 {_n(d.f1)}{_ro(_n(d.f1))} 본 결과를 쓴다."


def _declared_average_values(ctx: Ctx) -> tuple[float, ...]:
    d = ctx.facts.declared_average if ctx.facts else None
    return (d.f1,) if d is not None and d.f1 is not None else ()


# 프론트 reportMaterials.NOT_PROVIDED — 답이 없을 때 client_inputs 에 싣는 말(재료 선택 점검 4.2).
NOT_PROVIDED = "제공되지 않음"


def _metric_rationale(ctx: Ctx) -> str | None:
    rationale = ctx.client_inputs.get("metric_rationale")
    if not isinstance(rationale, str) or not rationale.strip() or rationale.strip() == NOT_PROVIDED:
        return None
    return "지표 선택 기준은 의뢰자가 고른 이유로 5절에 찍힌다."


def _purpose_reading(ctx: Ctx) -> str | None:
    purpose = _model_purpose(ctx)
    return f"모델 용도('{purpose}')에 비추어 결과를 해석한다." if purpose else None


# ── 8.3 리스크 ───────────────────────────────────────────────────────────────


def _weakest_risk(ctx: Ctx) -> str | None:
    w = ctx.facts.weakest_class if ctx.facts else None
    if w is None:
        return None
    purpose = _model_purpose(ctx)
    noun = "레이블" if ctx.task_type == "multilabel" else "클래스"
    tail = f" 모델 용도('{purpose}')에서 그 {noun}가 가지는 위험을 쓴다." if purpose else ""
    subject = "레이블은" if noun == "레이블" else "클래스는"
    return f"가장 약한 {subject} {w.label}(F1 {_n(w.f1)})이다.{tail}"


def _near_target(ctx: Ctx) -> str | None:
    if not any(m.near_target for m in ctx.fs.metrics):
        return None
    return "목표값과의 차이가 3% 미만인 지표는 고정 문구 18 이 숫자로 찍는다 — 숫자를 되풀이하지 말고 판정이 뒤집힐 위험만 해석한다."


def _threshold_risk(ctx: Ctx) -> str | None:
    if ctx.fs.f1_threshold_gain is None or not ctx.derived.get("threshold_tuning"):
        return None
    return "현재 결정 임계값이 최적이 아닐 수 있다(모델 쪽 임계값 조정 권고의 근거)."


def _latency(ctx: Ctx) -> str | None:
    lat = ctx.fs.latency
    if not lat.available:
        return None
    parts = [f"{name} {_n(v)}{lat.unit}" for name, v in (("평균", lat.mean), ("p50", lat.p50), ("p95", lat.p95), ("p99", lat.p99)) if v is not None]
    return f"지연시간: {', '.join(parts)}."


def _realtime(ctx: Ctx) -> str | None:
    decl = ctx.facts.declarations if ctx.facts else None
    if decl is None or decl.usage_mode != "realtime" or not ctx.fs.latency.available:
        return None
    return "의뢰자가 대화형(실시간) 사용을 밝혔으므로 응답 시간(지연시간 값)을 그 기준으로 해석한다."


def _realtime_values(ctx: Ctx) -> tuple[float, ...]:
    lat = ctx.fs.latency
    return tuple(float(v) for v in (lat.mean, lat.p50, lat.p95, lat.p99) if v is not None)


# ── 9 권고 ───────────────────────────────────────────────────────────────────


def _problem_table(ctx: Ctx) -> str | None:
    return "9절 문제 목록(번호·우선순위·원천 절)은 코드가 표로 찍는다."


def _recommendations(ctx: Ctx) -> str | None:
    if not ctx.fs.problems:
        return None
    items = ", ".join(f"{p.id}({p.title})" for p in ctx.fs.problems)
    return f"문제마다 권고를 하나 이상 쓴다(문제 번호에 묶음): {items}."


def _no_problem(ctx: Ctx) -> str | None:
    if ctx.fs.problems:
        return None
    sentence = ctx.fixed("6")
    return sentence.text if sentence else "지목된 문제가 없다(고정 문구 6)."


# ── 표(5절 41행) ──────────────────────────────────────────────────────────────

REQUIRED_ITEMS: tuple[RequiredItem, ...] = (
    RequiredItem("7.1-01", "7.1", "오분류 총건수·비율", ALL, "ai", "항상(혼동행렬·엔진 건수)", _misclassified),
    RequiredItem("7.1-02", "7.1", "오분류 방향 FP·FN", BINARY, "ai", "항상(엔진 건수)", _fp_fn),
    RequiredItem("7.1-03", "7.1", "선언한 위험 쪽 오류", BINARY, "ai", "위험 방향이 FN·FP", _declared_risk, d24=_declared_risk_values),
    RequiredItem("7.1-04", "7.1", "클래스별 FP·FN", MULTICLASS, "ai", "항상(엔진 건수)", lambda c: _per_class_errors(c, "클래스는")),
    RequiredItem("7.1-05", "7.1", "레이블별 FP·FN", MULTILABEL, "ai", "항상(엔진 건수)", lambda c: _per_class_errors(c, "레이블은")),
    RequiredItem("7.1-06", "7.1", "최다 오분류 쌍 상위 3", MULTICLASS, "ai", "항상(혼동행렬)", _top_pairs),
    RequiredItem("7.1-07", "7.1", "예측 편향(이진)", BINARY, "ai", "항상", lambda c: _bias(c, "positive_class")),
    RequiredItem("7.1-08", "7.1", "예측 편향(다중)", MULTICLASS, "ai", "항상", lambda c: _bias(c, "per_class")),
    RequiredItem("7.1-09", "7.1", "예측 편향(멀티)", MULTILABEL, "ai", "항상", lambda c: _bias(c, "label_total")),
    RequiredItem("7.1-10", "7.1", "최약 클래스·작은 표본", ALL, "ai", "항상(결정 A)", _weakest),
    RequiredItem("7.1-11", "7.1", "오분류의 성격(경계·확신 오류)", ALL, "fixed_interpret", "확률 컬럼", _error_nature),
    RequiredItem("7.1-12", "7.1", "사용된 결정 임계값", ALL, "fixed", "예측을 점수에서 파생", _derivation),
    RequiredItem("7.2-01", "7.2", "불균형 비율", ALL, "ai", "불균형 비율이 있음(멀티 정의는 고정 문구 7)", _imbalance_ratio),
    RequiredItem("7.2-02", "7.2", "불균형 영향 판단", ALL, "ai", "정답 + 예측(엔진 판단)", _imbalance_effect),
    RequiredItem("7.2-03", "7.2", "학습↔평가 분포 차이", ALL, "ai", "학습 분포 숫자 입력", _distribution_shift),
    RequiredItem("7.2-04", "7.2", "정답 라벨 주의", ALL, "fixed", "신뢰성 검토 없음·구축 방법 규칙·운영 로그", _label_caution),
    RequiredItem("7.3-01", "7.3", "추가 관찰", ALL, "ai", "결론 목록 밖의 관찰", _additional, open_ended=True),
    RequiredItem("8.1-01", "8.1", "참조 범위 대비 위치", ALL, "ai", "기준표에 있는 지표", _benchmark),
    RequiredItem("8.1-02", "8.1", "목표값 수준", ALL, "ai", "기준표에 있는 지표 + 목표값", _target_level),
    RequiredItem("8.1-03", "8.1", "비교 기준 없음", ALL, "fixed", "기준표에 없는 지표", _no_benchmark),
    RequiredItem("8.2-01", "8.2", "판정 규칙", ALL, "form", "항상(6.1 표)", _verdict_rule),
    RequiredItem("8.2-02", "8.2", "종합 판정과 근거", ALL, "ai", "항상(판정 사유 D19, 통과율은 참고)", _verdict_reason),
    RequiredItem("8.2-03", "8.2", "신뢰구간 해석", ALL, "ai", "판정 대상 지표에 구간이 하나라도 있음(전부 생략이면 사유)", _ci),
    RequiredItem("8.2-04", "8.2", "일반화 해석", ALL, "fixed", "학습 데이터와의 관계", _generalization),
    RequiredItem("8.2-05", "8.2", "Precision↔Recall 치우침", BINARY, "ai", "M2·M3 선택", _pr_gap),
    RequiredItem("8.2-06", "8.2", "임계값 조정 여지", BINARY, "ai", "점수 + F1, 엔진이 부적합으로 봄", _threshold_gain),
    RequiredItem("8.2-07", "8.2", "macro vs weighted", MULTICLASS, "ai", "M22 선택", lambda c: _average_gap(c, "macro_vs_weighted")),
    RequiredItem("8.2-08", "8.2", "macro vs micro", MULTILABEL, "ai", "M22 선택", lambda c: _average_gap(c, "macro_vs_micro")),
    RequiredItem("8.2-09", "8.2", "Hamming Loss vs Exact Match", MULTILABEL, "ai", "M15·M16 둘 다", _hamming_exact),
    RequiredItem("8.2-10", "8.2", "평균 방식 선택 기준", MULTICLASS, "form", "클래스 중요도", _averaging_form),
    RequiredItem("8.2-11", "8.2", "선언한 평균 방식으로 본 결과", MULTI, "ai", "클래스 중요도가 동등이 아님", _declared_average, d24=_declared_average_values),
    RequiredItem("8.2-12", "8.2", "지표 선택 기준", MULTILABEL, "form", "지표 선택 이유 입력", _metric_rationale),
    RequiredItem("8.2-13", "8.2", "모델 용도에 비춘 해석", ALL, "ai_fact", "모델 용도", _purpose_reading),
    RequiredItem("8.3-01", "8.3", "최약 클래스의 용도상 위험", ALL, "ai_fact", "항상(결정 A) + 모델 용도", _weakest_risk),
    RequiredItem("8.3-02", "8.3", "아슬아슬 지표", ALL, "fixed_interpret", "목표값 지표 중 근소 차이", _near_target),
    RequiredItem("8.3-03", "8.3", "임계값 부적합 위험", BINARY, "ai", "엔진이 부적합으로 봄", _threshold_risk),
    RequiredItem("8.3-04", "8.3", "지연시간", ALL, "ai", "latency 컬럼", _latency),
    RequiredItem("8.3-05", "8.3", "대화형 용도의 응답 시간", ALL, "ai", "사용 방식이 대화형 + latency", _realtime, d24=_realtime_values),
    RequiredItem("9-01", "9", "문제 목록", ALL, "form", "항상", _problem_table),
    RequiredItem("9-02", "9", "권고", ALL, "ai", "문제 1개 이상", _recommendations),
    RequiredItem("9-03", "9", "권고 없음", ALL, "fixed", "문제 0개", _no_problem),
)

assert {i.section for i in REQUIRED_ITEMS} == set(SECTION_IDS), "필수 항목의 칸은 소절 7칸이다"
assert len({i.id for i in REQUIRED_ITEMS}) == len(REQUIRED_ITEMS), "항목 ID 가 겹친다"


# 표 지표(혼동행렬·클래스별 표)는 값이 없어 서술 지표가 아니다. 옛 요청(운영 main 프론트)은 value 0·info 줄로 싣는다.
_TABLE_METRICS = frozenset({"M21", "M22"})


def _narrative_metrics(fs: FactSheet) -> list:
    return [m for m in fs.metrics if m.metric_id not in _TABLE_METRICS]


def _selected_metrics(fs: FactSheet) -> set[str]:
    """의뢰자가 고른 지표. 새 요청은 facts.selected_metrics(표 지표 M21·M22 포함), 옛 요청은 지표 목록."""
    if fs.facts is not None and fs.facts.selected_metrics:
        return set(fs.facts.selected_metrics)
    return {m.metric_id for m in fs.metrics}


def select_items(fs: FactSheet, client_inputs: dict | None, task_type: str) -> list[Conclusion]:
    """이 성적서에 해당하는 필수 항목만 남기고 숫자를 채운 결론 목록(표 순서 = 칸 순서).

    작업 유형이 맞지 않거나 재료가 없으면(결론 함수가 None) 뺀다. D24 항목은 확인할 숫자(d24_values)를 함께 싣는다.
    facts 가 없는 옛 요청에서도 깨지지 않고 받은 것만으로 고른다.
    """
    ctx = Ctx(
        fs=fs,
        task_type=task_type,
        client_inputs=client_inputs or {},
        derived=compute_derived(fs, task_type=task_type),
        benchmark_refs=build_benchmark_refs(task_type, fs.metrics),
        selected=_selected_metrics(fs),
    )
    out: list[Conclusion] = []
    for item in REQUIRED_ITEMS:
        if task_type not in item.task_types:
            continue
        text = item.build(ctx)
        if text is None:
            continue
        values = item.d24(ctx) if item.d24 is not None else ()
        out.append(Conclusion(item.id, item.section, item.author, text, tuple(values), item.open_ended))
    return out


def conclusions_by_section(conclusions: list[Conclusion]) -> dict[str, list[Conclusion]]:
    """칸별로 묶는다(소절 7칸 순서, 결론이 없는 칸은 빈 목록)."""
    grouped: dict[str, list[Conclusion]] = {sid: [] for sid in SECTION_IDS}
    for c in conclusions:
        grouped[c.section].append(c)
    return grouped
