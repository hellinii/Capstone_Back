"""app/narrative/required_items.py — 필수 서술 항목 목록(서술 설계 5절)과 결론 목록(D7).

목록은 하나다 — AI 지시문(S5)과 규칙 템플릿(S6)이 같은 목록을 읽는다(계획 원칙 2). 심판은 이 함수를 쓰지 않는다
(사실 시트·파생값만 공유 — 서술 설계 8절).

- REQUIRED_ITEMS: 5절 41행을 코드 표로 — 항목 ID · 칸 · 작업 유형 · 재료 조건 · 필요한 값(결론을 만드는 함수) ·
  작성 주체 · 생성 뒤 확인(D24 숫자) · 규칙 템플릿 문장.
- select_items(사실 시트, 고객 입력, 작업 유형): 이 성적서에 해당하는 항목만 남기고, 숫자를 채운 결론을 돌려준다.
  숫자는 받은 사실(facts·지표·엔진 값)과 서버 파생값(혼동행렬 합계·백분율)뿐이다 — 여기서 새로 판정하지 않는다.

작성 주체(Author)와 두 글(결론 text · 템플릿 template_text):
- ai: AI 가 쓴다. text 는 숫자를 채운 서술체 결론 문장이고, 규칙 템플릿은 같은 문장을 찍는다(template_text = text,
  따로 정한 항목만 다름 — 판정 사유·불균형·임계값 위험처럼 규칙 문장이 더 자세한 것).
- ai_fact: AI 가 해석을 쓴다. 템플릿은 사실 문장만 찍는다(해석하지 않는다 — 모델 용도·최약 클래스 위험).
- fixed_interpret: 코드가 고정 문구(17·18)로 숫자를 찍고, AI·템플릿은 그 숫자를 되풀이하지 않고 해석만. text 는 숫자 없는
  해석 재료(우세한 오류·근소 지표의 판정), 템플릿은 해석 한 문장. 성적서가 그 문장을 찍지 않는 옛 요청(운영 main 프론트 —
  facts 가 없다)이면 템플릿이 숫자 문장을 쓴다(code_prints).
- fixed: 코드가 고정 문구로 찍는다(서술 칸에 쓰지 않는다 — 템플릿도 찍지 않는다).
- form: 양식이 의뢰자 선언·선택으로 조립한다(5절·6.1 — 서술 칸에 쓰지 않는다).
text 가운데 fixed·form·7.3(open_ended) 항목은 AI 용 설명이라 템플릿이 찍지 않는다(template_text = None).

작성 규칙(D37)은 결론과 템플릿이 함께 따른다: 클래스는 '이름(설명)'(고객 입력 class_label_descriptions), 영어 코드값·
필드명을 쓰지 않음, 서술체 완결 문장(개조식·'항목: 값' 꼴 없음). 조사·숫자 표기는 app.narrative.wording.

계획 결정 반영(00_계획.md 사용자 결정·설계 요지 3):
- 최약 클래스·주요 오류(7.1)는 M21·M22 선택과 무관하게 '항상'(결정 A).
- 8.2 신뢰구간 해석은 판정 대상 지표에 구간이 하나라도 있으면 — 목표 포함 / 모두 떨어짐 두 갈래. 판정 대상 전부가
  생략이면 생략 사유를 쓴다.
- 7.1 오류 성격·8.3 근소 차이는 고정 문구 17·18 + 해석. 8.2 평균 방식·지표 선택 기준은 양식.

상호작용
- 의존(import): app.narrative.schemas, app.narrative.derived(compute_derived), app.narrative.baselines, app.narrative.sections,
  app.narrative.wording
- 사용처: (S5) 지시문의 칸별 "반드시 다룰 결론", (S6) 규칙 템플릿(app.narrative.fallback), (S5) D24 생성 뒤 확인, tests
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.schemas import FactSheet
from app.narrative.sections import SECTION_IDS
from app.narrative.wording import (
    KOREAN_COUNT,
    VERDICT_LABEL,
    class_descriptions,
    class_name,
    join_and,
    metric_num,
    metric_vs_target,
    num,
    pct,
    pct_text,
    shown_value,
    target_text,
    unsigned,
    with_josa,
)

Author = Literal["ai", "ai_fact", "fixed_interpret", "fixed", "form"]
TaskType = Literal["binary", "multiclass", "multilabel"]
ALL: frozenset[str] = frozenset({"binary", "multiclass", "multilabel"})
BINARY: frozenset[str] = frozenset({"binary"})
MULTICLASS: frozenset[str] = frozenset({"multiclass"})
MULTILABEL: frozenset[str] = frozenset({"multilabel"})
MULTI: frozenset[str] = frozenset({"multiclass", "multilabel"})


@dataclass(frozen=True)
class Conclusion:
    """항목 하나의 결론 — 숫자를 채운 사실 문장(한국어). 지시문은 '반드시 다룰 결론'으로, 템플릿은 template_text 를 찍는다."""
    item_id: str
    section: str
    author: Author
    text: str
    # D24 생성 뒤 확인: 그 칸의 글에 이 값 중 하나 이상(허용 목록과 같은 표기 변형)이 있어야 한다. D24 항목만.
    d24_values: tuple[float, ...] = ()
    # 7.3 처럼 결론 목록 밖에서 AI 가 찾는 칸(정해진 결론이 없다).
    open_ended: bool = False
    # 규칙 템플릿이 그 칸에 찍는 문장. None 이면 찍지 않는다(fixed·form·7.3, 해당 재료가 없는 해석 항목).
    template_text: str | None = None


def code_prints(fs: FactSheet, sentence_id: str) -> bool:
    """성적서가 그 고정 문장(17·18 등)을 칸 옆에 따로 찍는가 — 그러면 AI·템플릿은 그 숫자를 되풀이하지 않는다.

    새 프론트(facts 를 보냄)는 찍을 문장을 facts.fixed_sentences 에 싣고 같은 문장을 화면에 찍는다. facts 가 상한을 넘어
    문장 목록을 비웠어도(omitted.fixed_sentences) 화면은 찍는다. 옛 요청(운영 main 프론트 — facts 가 없음)은 찍지 않는다.
    문장이 있어야 하는 조건(17 = 확률 컬럼, 18 = 근소 차이 지표)은 항목의 재료 조건이 따로 본다.
    """
    facts = fs.facts
    if facts is None:
        return False
    if any(s.id == sentence_id for s in facts.fixed_sentences):
        return True
    return bool(facts.omitted.get("fixed_sentences"))


@dataclass
class Ctx:
    fs: FactSheet
    task_type: str
    client_inputs: dict[str, Any]
    derived: dict
    benchmark_refs: list[dict]
    selected: set[str] = field(default_factory=set)
    descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def facts(self):
        return self.fs.facts

    def fixed(self, sentence_id: str):
        if self.facts is None:
            return None
        return next((s for s in self.facts.fixed_sentences if s.id == sentence_id), None)

    def name(self, label) -> str:
        """클래스·레이블을 부르는 꼴 — 이름(설명)(D37)."""
        return class_name(label, self.descriptions)

    def prints(self, sentence_id: str) -> bool:
        return code_prints(self.fs, sentence_id)


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
    # 템플릿 문장(결론과 다를 때만). ai 는 없으면 결론 문장, ai_fact·fixed_interpret 는 반드시 둔다, fixed·form 은 두지 않는다.
    template: Callable[[Ctx], "str | None"] | None = None


def _noun(ctx: Ctx) -> str:
    return "레이블" if ctx.task_type == "multilabel" else "클래스"


def _topic(ctx: Ctx) -> str:
    return "레이블은" if ctx.task_type == "multilabel" else "클래스는"


def _model_purpose(ctx: Ctx) -> str | None:
    model = ctx.client_inputs.get("model") if isinstance(ctx.client_inputs, dict) else None
    purpose = model.get("모델 용도") if isinstance(model, dict) else None
    if not isinstance(purpose, str) or not purpose.strip() or purpose.strip() == NOT_PROVIDED:
        return None
    return purpose.strip()


def _clauses(stems: list[str]) -> str:
    """'…이고, …이며, …다.' — 어간 목록을 이어 한 문장으로(마지막은 '다.')."""
    return "고, ".join(stems) + "다."


# ── 7.1 혼동행렬 해석 ──────────────────────────────────────────────────────────


def _misclassified(ctx: Ctx) -> str | None:
    conf = ctx.derived.get("confusion")
    if not conf:
        return None
    share = f"({conf['misclassified_pct']:g}%)" if "misclassified_pct" in conf else ""
    if conf.get("unit") == "label_cell":
        # 멀티레이블 — 합계는 표본이 아니라 레이블 칸(표본 수 × 레이블 수)이다. '전체 N건'이라 쓰면 7.2 의 표본 수와
        # 어긋난다(점검 BE-09). 경계·확신 오류 문장(D30)과 같은 '레이블 칸' 단위로 쓴다.
        text = f"레이블별 혼동행렬을 합친 레이블 칸 {conf['total']}칸 중 {conf['misclassified']}칸{share}이 틀렸다."
        if "fn" in conf and "fp" in conf:
            text += f" 있어야 할 레이블을 놓친 칸(FN)은 {conf['fn']}칸, 없는 레이블을 붙인 칸(FP)은 {conf['fp']}칸이다."
        return text
    return f"혼동행렬 기준 전체 {conf['total']}건 중 {conf['misclassified']}건{share}이 오분류되었다."


def _binary_counts(ctx: Ctx):
    """이진의 (양성 클래스, FP, FN) — 엔진 건수(confusion_counts), 없으면(옛 요청) 혼동행렬 2×2 의 서버 셈."""
    if ctx.task_type != "binary":
        return None
    rows = ctx.fs.confusion_counts
    if rows:
        return rows[0].label, rows[0].fp, rows[0].fn
    conf = ctx.derived.get("confusion") or {}
    if "fp" in conf and "fn" in conf:
        label = ctx.fs.confusion.positive_class if ctx.fs.confusion else None
        return label, conf["fp"], conf["fn"]
    return None


def _fp_fn(ctx: Ctx) -> str | None:
    got = _binary_counts(ctx)
    if got is None:
        return None
    label, fp, fn = got
    who = f"양성 클래스 {ctx.name(label)} 기준으로 " if label else ""
    return f"{who}거짓 양성(FP)은 {fp}건, 거짓 음성(FN)은 {fn}건이다."


_RISK_SIDE = {"fn": "거짓 음성(FN, 놓침)", "fp": "거짓 양성(FP, 잘못 잡음)"}


def _declared_risk(ctx: Ctx) -> str | None:
    dr = ctx.facts.declared_risk_errors if ctx.facts else None
    if dr is None:
        return None
    side = _RISK_SIDE[dr.side]
    if dr.share is None:
        return f"오분류가 없어 의뢰자가 더 위험하다고 밝힌 {side}도 {dr.count}건이다."
    return f"의뢰자가 더 위험하다고 밝힌 {side}은 {dr.count}건으로 전체 오류 {dr.total_errors}건의 {pct(dr.share)}이다."


def _declared_risk_values(ctx: Ctx) -> tuple[float, ...]:
    dr = ctx.facts.declared_risk_errors if ctx.facts else None
    if dr is None:
        return ()
    return (float(dr.count),) + ((dr.share,) if dr.share is not None else ())


def _trimmed(ctx: Ctx, *keys: str) -> bool:
    """프론트가 클래스 상한으로 그 표를 잘랐는가(facts.omitted). 잘린 표에서는 '가장 많은'·'전부'를 단정하지 않는다."""
    omitted = ctx.facts.omitted if ctx.facts is not None else {}
    return any(omitted.get(k) for k in keys)


def _per_class_errors(ctx: Ctx) -> str | None:
    rows = ctx.fs.confusion_counts
    if not rows:
        return None
    top = sorted(rows, key=lambda r: -(r.fp + r.fn))[:3]
    unit = "칸" if ctx.task_type == "multilabel" else "건"  # 멀티레이블은 레이블 칸 단위(7.1 의 합계·17 과 같은 단위)
    items = join_and([f"{ctx.name(r.label)} FP {r.fp}{unit}·FN {r.fn}{unit}" for r in top])
    noun = _noun(ctx)
    if _trimmed(ctx, "confusion_counts"):
        # 클래스가 많아 프론트가 표본 수 순으로 줄을 잘랐다 — 빠진 클래스의 오류가 더 많을 수 있다(세션 A 검토 P1).
        return (
            f"{noun}가 많아 표에는 일부 {noun}만 실렸으며, 실린 {noun} 가운데 오류(FP+FN)가 많은 "
            f"{with_josa(noun, '은/는')} {items}이다."
        )
    if len(rows) > len(top):
        # '상위 3개'처럼 숫자로 쓰면 허용 목록 밖 숫자다 — 우리말 수(세)로 쓴다.
        return f"오류(FP+FN)가 가장 많은 {KOREAN_COUNT[len(top)]} {with_josa(noun, '은/는')} {items}이다."
    return f"{noun}별 오류(FP+FN)는 {items}이다."


def _top_pairs(ctx: Ctx) -> str | None:
    pairs = ctx.facts.top_confusion_pairs if ctx.facts else []
    if not pairs:
        return None
    items = join_and([
        f"{ctx.name(p.actual)}→{ctx.name(p.predicted)} {p.count}건(전체 오분류의 {pct(p.share_of_errors)})" for p in pairs
    ])
    return f"가장 많이 헷갈린 쌍(실제→예측)은 {items}이다."


def _bias(ctx: Ctx, unit: str) -> str | None:
    bias = ctx.facts.prediction_bias if ctx.facts else None
    if bias is None or bias.unit != unit or not bias.rows:
        return None
    if unit == "positive_class":
        r = bias.rows[0]
        pos = f"양성 클래스 {ctx.name(r.label)}" if r.label else "양성"
        if not r.difference:
            return f"{with_josa(pos, '으로/로')} 예측한 것과 실제 양성이 {r.actual}건으로 같다."
        lean = "많이" if r.difference > 0 else "적게"
        return (
            f"{with_josa(pos, '으로/로')} 예측한 것은 {r.predicted}건, 실제 양성은 {r.actual}건으로, "
            f"양성을 실제보다 {abs(r.difference)}건 {lean} 예측했다."
        )
    if unit == "label_total":
        r = bias.rows[0]
        if not r.difference:
            return f"예측한 레이블 수와 실제 레이블 수가 {r.actual}개로 같다."
        lean = "많이" if r.difference > 0 else "적게"
        return (
            f"예측한 레이블은 모두 {r.predicted}개, 실제 레이블은 {r.actual}개로, "
            f"레이블을 실제보다 {abs(r.difference)}개 {lean} 붙였다."
        )
    top = sorted(bias.rows, key=lambda r: -abs(r.difference))[:3]
    if all(r.difference == 0 for r in top):
        return "클래스마다 예측한 수와 실제 수가 같다."
    items = join_and([f"{ctx.name(r.label)} 예측 {r.predicted}건·실제 {r.actual}건" for r in top if r.difference])
    return f"예측 수와 실제 수의 차이가 큰 클래스는 {items}이다."


def _weakest(ctx: Ctx) -> str | None:
    w = ctx.facts.weakest_class if ctx.facts else None
    if w is None:
        return None
    noun = _noun(ctx)
    base = "레이블 등장" if ctx.facts.class_share_basis == "label_occurrences" else "전체 표본"
    text = (
        f"F1 이 가장 낮은 {_topic(ctx)} F1 {num(w.f1)}인 {ctx.name(w.label)}이며, "
        f"표본은 {w.support}건으로 {base}의 {pct_text(w.share_text)}이다."
    )
    if w.gap_text is not None:
        text += f" 매크로 F1 과의 차이는 {w.gap_text}이다."
    # 사유별로 따로 쓴다 — 'A 또는 B' 로 쓰면 걸리지 않은 쪽까지 주장하게 된다.
    stems = []
    if w.support < 30:
        stems.append("표본이 30건 미만이")
    if float(w.share_text) < 0.05:  # share_text 는 프론트 판정과 같은 쪽으로 읽히게 만든 표기다
        stems.append(f"{base}의 5% 미만이")
    if w.far_below_macro:
        stems.append("매크로 F1 보다 0.1 이상 낮")
    if stems:
        text += f" 이 {with_josa(noun, '은/는')} " + _clauses(stems)
    small = [r for r in ctx.facts.low_support_classes if r.label != w.label]
    if small:
        listed = join_and([f"{ctx.name(r.label)} {r.support}건" for r in small[:5]])
        more = len(small) > 5 or _trimmed(ctx, "low_support_classes", "facts_class_lists")
        text += f" 그 밖에 표본이 작은 {with_josa(noun, '은/는')} {listed}{' 등' if more else ''}이다."
    return text


# 오류의 성격(D30) — 고정 문구 17 이 숫자를 찍는다. 해석 재료는 '우세한 오류'(경계 오류와 확신 오류 중 많은 쪽).


def _error_nature(ctx: Ctx) -> str | None:
    """AI 에게 주는 해석 재료(숫자 없음 — 숫자는 17 이 찍는다)."""
    er = ctx.fs.error_ratios
    if er is None:
        return None
    if er.misclassified == 0:
        return "확률이 있는 오분류가 없어 경계 오류·확신 오류를 산출하지 않았다."
    if er.boundary is None:
        return "경계 오류는 계산하지 않았다. " + ("확신 오류가 있다." if er.confident.count else "확신 오류는 없다.")
    b, c = er.boundary.count, er.confident.count
    if b > c:
        return "경계 오류가 확신 오류보다 많다(우세한 오류는 경계 오류)."
    if c > b:
        return "확신 오류가 경계 오류보다 많다(우세한 오류는 확신 오류)."
    return "경계 오류와 확신 오류가 같은 수다(우세한 오류 없음)." if b else "경계 오류와 확신 오류가 모두 없다."


def _error_interpretation(er) -> str | None:
    """오류 성격의 해석 한 문장 — 숫자를 쓰지 않는다(17 이 찍는 숫자를 되풀이하지 않게).

    어느 쪽이 많은지와 그 오류의 뜻만 쓴다. 비중이 크다·결과가 달라지기 쉽다 같은 정도의 주장은 하지 않는다 — 두 오류가
    오분류의 아주 작은 일부일 수도 있다(세션 A 검토). 다중 클래스는 결정 임계값이 없으므로 임계값을 말하지 않는다 —
    경계 오류는 '결정 경계 가까이에서 갈린 오류'로만 부른다.
    """
    if er.misclassified == 0:
        return None
    if er.boundary is None:
        # 경계 오류를 계산하지 못한 것은 이진·멀티레이블에서 임계값을 모를 때뿐이다.
        if er.confident.count:
            return (
                "확신 오류는 모델이 틀린 쪽에 높은 확률을 준 오류라 결정 임계값을 조정해도 잘 줄지 않으므로, "
                "해당 사례의 입력과 정답 라벨을 함께 점검할 필요가 있다."
            )
        return "확신 오류는 없어, 모델이 높은 확률로 틀린 사례는 관찰되지 않았다."
    b, c = er.boundary.count, er.confident.count
    if b > c:
        return (
            "확률이 있는 오분류 가운데 경계 오류가 확신 오류보다 많다. 경계 오류는 결정 경계 가까이에서 갈린 오류로, "
            "경계에 가까운 사례의 판단에 따라 결과가 달라질 수 있는 오류다."
        )
    if c > b:
        return (
            "확률이 있는 오분류 가운데 확신 오류가 경계 오류보다 많다. 확신 오류는 모델이 틀린 쪽에 높은 확률을 준 오류로, "
            "해당 사례의 입력과 정답 라벨을 점검할 필요가 있다."
        )
    if b == 0:
        return "경계 오류와 확신 오류가 모두 없어, 오분류는 결정 경계에서 떨어져 있으면서 높은 확신은 아닌 사례에서 일어났다."
    return "경계 오류와 확신 오류가 같은 수로 나타나 어느 한쪽 성격의 오류가 우세하지 않다."


def error_ratio_sentence(er) -> str:
    """7.1 오류의 성격 숫자 문장(서술 설계 D30) — 성적서가 고정 문구 17 을 찍지 않는 옛 요청에서만 템플릿이 쓴다.
    숫자는 error_ratios 의 값과 두 기준뿐이다."""
    cell = er.unit == "label_cell"
    unit, counter = ("틀린 레이블 칸", "칸") if cell else ("오분류", "건")
    if er.misclassified == 0:
        subject = "틀린 레이블 칸이" if cell else "오분류가"
        return f"확률이 있는 {subject} 없어 경계 오류·확신 오류 비율은 산출하지 않았다."

    def seg(part) -> str:
        return f"{part.count}{counter}({pct(part.ratio)})" if part.ratio is not None else f"{part.count}{counter}"

    confident = f"틀린 쪽 확률이 {er.confident_level:g} 이상인 확신 오류는 {seg(er.confident)}이다."
    if er.boundary is not None:
        return (
            f"{unit} {er.misclassified}{counter} 중 결정 경계와의 거리가 {er.boundary_margin:g} 미만인 "
            f"경계 오류는 {seg(er.boundary)}, {confident}"
        )
    if er.boundary_note == "threshold_unknown":
        reason = "예측값을 직접 제출해 결정 임계값을 알 수 없으므로 경계 오류는 계산하지 않았고, "
    else:
        reason = "경계 오류는 계산하지 않았고, "
    return f"{unit} {er.misclassified}{counter} 중 {reason}{confident}"


def _error_nature_template(ctx: Ctx) -> str | None:
    er = ctx.fs.error_ratios
    if er is None:
        return None
    interpretation = _error_interpretation(er)
    if ctx.prints("17"):
        return interpretation  # 숫자는 성적서가 7.1 옆에 찍는다
    return error_ratio_sentence(er) + (f" {interpretation}" if interpretation else "")


def _derivation(ctx: Ctx) -> str | None:
    if ctx.fs.derived_prediction is None:
        return None
    sentence = ctx.fixed("9")
    return sentence.text if sentence else "예측값은 제출된 확률에서 파생했다(고정 문구 9)."


# ── 7.2 분포·불균형 ───────────────────────────────────────────────────────────


def _imbalance_ratio(ctx: Ctx) -> str | None:
    ie = ctx.fs.imbalance_effect
    if ie is not None:
        shown = ie.display.imbalance_ratio if ie.display else num(ie.imbalance_ratio)
        text = f"불균형 비율은 {shown}이다."
    elif ctx.fs.distribution is not None and ctx.fs.distribution.imbalance_ratio is not None:
        text = f"불균형 비율은 {num(ctx.fs.distribution.imbalance_ratio)}이다."
    else:
        return None
    if ctx.task_type == "multilabel":
        # AI·템플릿이 쓰는 결론에는 절·문구 번호를 넣지 않는다(숫자 검문 대상이다). 정의 문장은 성적서가 찍는다.
        text += " 멀티레이블 불균형 비율의 정의 문장은 성적서가 따로 찍는다."
    return text


def _imbalance_ratio_template(ctx: Ctx) -> str | None:
    """불균형 판단(7.2-02)이 있으면 그 문장이 비율을 함께 쓴다. 없으면(옛 요청) 비율만."""
    if ctx.fs.imbalance_effect is not None:
        return None
    if ctx.fs.distribution is not None and ctx.fs.distribution.imbalance_ratio is not None:
        return f"불균형 비율은 {num(ctx.fs.distribution.imbalance_ratio)}이다."
    return None


# 엔진 판정 규칙(D20)의 설명 — 이진·다중은 '정확도 − 매크로 F1' 과 '다수 − 소수 F1' 둘을, 멀티레이블은 다수·소수
# 레이블 F1 차 하나만 본다(S4b 검토 RI-08).
_IMBALANCE_LEVEL = {
    "affected": "불균형이 성능 수치에 영향을 주었다고 판단했다(비율이 1.5 이상이고 {gaps} 0.1 이상)",
    "no_distortion": "불균형은 있으나 성능 왜곡은 관찰되지 않았다고 판단했다(비율이 1.5 이상이고 {gaps_all} 0.1 미만)",
    "mild": "불균형이 경미하다고 판단했다(비율이 1.5 미만)",
}


def _imbalance_effect(ctx: Ctx) -> str | None:
    ie = ctx.fs.imbalance_effect
    if ie is None:
        return None
    shown = ie.display
    gap = shown.majority_minus_minority_f1 if shown else num(ie.majority_minus_minority_f1)
    if ie.accuracy_minus_macro_f1 is None:
        gaps = gaps_all = "다수·소수 레이블 F1 차이가"
    else:
        gaps, gaps_all = "정확도 − 매크로 F1 과 다수 − 소수 F1 차이 중 하나가", "정확도 − 매크로 F1 과 다수 − 소수 F1 차이가 모두"
    rule = _IMBALANCE_LEVEL[ie.level].format(gaps=gaps, gaps_all=gaps_all)
    text = (
        f"평가 엔진은 {rule}. 다수 {_noun(ctx)} {ctx.name(ie.majority_class)}와 소수 {_noun(ctx)} "
        f"{ctx.name(ie.minority_class)}의 F1 차이는 {gap}"
    )
    if ie.accuracy_minus_macro_f1 is not None:
        acc = shown.accuracy_minus_macro_f1 if shown and shown.accuracy_minus_macro_f1 else num(ie.accuracy_minus_macro_f1)
        text += f"이고, 정확도 − 매크로 F1 은 {acc}"
    return text + "이다."


def imbalance_sentence(ie, name: Callable[[str], str] = str) -> str:
    """7.2 불균형 결론 한 문장(서술 설계 D20, 규칙 템플릿). level 판정은 평가 엔진이 했고 여기서는 읽기만 한다.

    엔진은 두 차이 중 **하나만** 0.10 이상이어도 '영향 있음'이다. 그래서 기준 이상인 차이만 근거로 쓰고,
    나머지는 '기준 미만'으로 따로 쓴다(종전에는 두 차이를 다 늘어놓고 '0.1 이상 벌어져'를 붙여, 기준 미만·
    음수 차이까지 넘은 것처럼 썼다 — 점검 BE-02). 다수 F1 − 소수 F1 이 음수면 소수 쪽이 더 잘 맞은 것이다.
    값은 엔진이 넷째 자리로 반올림해 싣는다 — 반올림하면 기준과 같아 보이는데 엔진이 기준 미만으로 본
    값은 숫자 없이 '기준 미만'만 쓴다(0.1 미만인데 '0.1' 로 찍히는 모순을 피한다).
    숫자는 imbalance_effect 의 값과 두 기준(1.5·0.1)뿐이라 허용 목록 안이다. 숫자는 괄호 안에 둬 조사를 숫자
    받침에 맞출 일이 없게 한다. 정확도 − 매크로 F1 이 없으면(멀티레이블) 다수·소수 '레이블'의 F1 차이만 쓴다.
    클래스 이름은 name(이름(설명) — D37)으로 부른다.
    """
    ir = ie.imbalance_ratio
    shown = ie.display  # 엔진이 판정과 같은 쪽으로 읽히게 자릿수를 늘린 표기(S2). 없으면(옛 요청) 4자리 값.
    ir_text = shown.imbalance_ratio if shown else f"{ir}"
    unit = "레이블" if ie.accuracy_minus_macro_f1 is None else "클래스"
    if ie.level == "mild":
        # 1.49996 은 넷째 자리로 1.5 가 된다 — 엔진은 1.5 미만(경미)으로 봤다. 표시값(1.49996)이 있으면 그대로 쓰고,
        # 없으면 숫자를 빼서 '1.5 로 1.5 미만' 모순을 피한다.
        ratio = "불균형 비율이" if ir >= 1.5 and not shown else f"불균형 비율({ir_text})이"
        return f"{ratio} 1.5 미만이어서 불균형은 경미하다."

    gaps = []  # (이름, 값)
    if ie.accuracy_minus_macro_f1 is not None:
        gaps.append(("정확도와 매크로 F1 의 차이", ie.accuracy_minus_macro_f1))
    gaps.append((
        f"다수 {unit}({name(ie.majority_class)}){'과' if unit == '레이블' else '와'} 소수 {unit}({name(ie.minority_class)})의 F1 차이",
        ie.majority_minus_minority_f1,
    ))
    texts = {}
    if shown:
        if ie.accuracy_minus_macro_f1 is not None and shown.accuracy_minus_macro_f1:
            texts[gaps[0][0]] = shown.accuracy_minus_macro_f1
        texts[gaps[-1][0]] = shown.majority_minus_minority_f1

    def val(gap_name: str, value: float) -> str:
        return texts.get(gap_name, f"{value}")

    def at_least(gap_name: str, value: float) -> bool:
        # 엔진 표시값은 판정과 같은 쪽으로 읽히게 자릿수를 늘린 값이라 그것으로 본다 — 4자리 값 0.1 이 실제로는
        # 0.09996(기준 미만)일 수 있다. 옛 요청(표시값 없음)만 4자리 값으로 본다. 표시값은 스키마가 숫자 꼴만 받는다.
        try:
            return float(texts[gap_name]) >= 0.1 if gap_name in texts else value >= 0.1
        except ValueError:
            return value >= 0.1

    over = [(n, v) for n, v in gaps if ie.level == "affected" and at_least(n, v)]
    under = [(n, v) for n, v in gaps if (n, v) not in over]

    class_gap_name = gaps[-1][0]

    def under_text(gap_name: str, value: float) -> str:
        # '소수 쪽 F1 이 더 높다'는 다수 F1 − 소수 F1 이 음수일 때만 맞다. 정확도 − 매크로 F1 이 음수인 것은
        # 평균 F1 이 정확도보다 높다는 뜻일 뿐이다(독립 검토 S1-4-FB-01).
        if value < 0 and gap_name == class_gap_name:
            return f"{gap_name}({val(gap_name, value)})는 음수로 소수 {unit}의 F1 이 더 높다"
        if value >= 0.1 and gap_name not in texts:  # 반올림해 기준과 같아 보이지만 엔진은 기준 미만으로 봤다
            return f"{gap_name}는 기준(0.1) 미만이다"
        return f"{gap_name}({val(gap_name, value)})는 기준(0.1) 미만이다"

    head = f"불균형 비율({ir_text})이 1.5 이상이어서 불균형이 있"
    if over:
        named = "와 ".join(f"{n}({val(n, v)})" for n, v in over)
        text = f"{head}고, {named}가 기준(0.1) 이상이어서 불균형이 성능 수치에 영향을 준 것으로 보인다."
        if under:
            text += " " + " ".join(under_text(n, v) + "." for n, v in under)
        return text + f" 소수 {unit}의 성능 지표 해석에 유의가 필요하다."
    return (
        f"{head}으나, 불균형에 의한 성능 왜곡은 관찰되지 않았다. "
        + " ".join(under_text(n, v) + "." for n, v in under)
    )


def _imbalance_effect_template(ctx: Ctx) -> str | None:
    ie = ctx.fs.imbalance_effect
    return imbalance_sentence(ie, ctx.name) if ie is not None else None


def _distribution_shift(ctx: Ctx) -> str | None:
    facts = ctx.facts
    if facts is None or facts.distribution_shift_status != "compared" or not facts.distribution_shift:
        return None
    flagged = [r for r in facts.distribution_shift if r.flagged]
    noun = _noun(ctx)
    if not flagged:
        return f"학습 분포와 평가 분포의 {noun} 비율 차는 모두 10%p 미만이다."
    items = join_and([
        f"{ctx.name(r.label)} 학습 {pct(r.training_ratio)}·평가 {pct(r.evaluation_ratio)}" for r in flagged
    ])
    return f"학습 분포와 평가 분포의 비율 차가 10%p 이상인 {with_josa(noun, '은/는')} {items}이다."


def _label_caution(ctx: Ctx) -> str | None:
    sentence = ctx.fixed("12")
    return sentence.text if sentence else None


def _additional(ctx: Ctx) -> str | None:
    return "결론 목록 밖에서 사실 시트로 찾은 관찰(있으면 허용 숫자를 하나 이상 인용, 없으면 고정 문구 15 '해당 없음')."


# ── 8.1 벤치마크 ──────────────────────────────────────────────────────────────

# 방향을 반영한 품질(D7[3]) — 낮을수록 좋은 지표의 'below'는 미흡이 아니라 우수. 같은 품질끼리 묶어 한 문장으로.
_QUALITY_GROUP = (
    ("better", "참조 범위보다 우수한 지표"),
    ("within", "참조 범위 안에 드는 지표"),
    ("worse", "참조 범위에 비해 미흡한 지표"),
)
# 목표값 수준(D31) — 단순 수치 위치라 방향(우수/미흡)으로 바꾸지 않는다.
_TARGET_GROUP = (
    ("below", "목표값이 참조 범위보다 낮은 지표"),
    ("within", "목표값이 참조 범위 안에 있는 지표"),
    ("above", "목표값이 참조 범위보다 높은 지표"),
)


def _ref_value(ctx: Ctx, ref: dict) -> str:
    """기준표 비교의 모델 값 — 판정 대상이면 성적서 6.1 과 같은 표기(value_text), 아니면 넷째 자리(날값을 찍지 않는다)."""
    metric = next((m for m in ctx.fs.metrics if m.display_name == ref["metric"]), None)
    return shown_value(metric) if metric is not None else num(float(ref["model_value"]))


def _grouped(groups, refs: list[dict], key: str, item) -> str:
    """같은 위치끼리 묶어 '…인 지표는 A(…), B(…)이고, …인 지표는 C(…)이다.' — 지표 이름 뒤에 조사를 붙이지 않는다."""
    stems = []
    for code, head in groups:
        listed = [item(r) for r in refs if r.get(key) == code]
        if listed:
            stems.append(f"{head}는 {join_and(listed)}이")
    unknown = [item(r) for r in refs if r.get(key) not in {c for c, _ in groups}]
    if unknown:
        stems.append(f"그 밖의 지표는 {join_and(unknown)}이")
    return _clauses(stems)


def _benchmark(ctx: Ctx) -> str | None:
    if not ctx.benchmark_refs:
        return None
    return "내부 참조 기준과 비교하면 " + _grouped(
        _QUALITY_GROUP, ctx.benchmark_refs, "quality",
        lambda r: f"{r['metric']}({_ref_value(ctx, r)}, 참조 범위 {num(float(r['ref_low']))}~{num(float(r['ref_high']))})",
    )


def _target_level(ctx: Ctx) -> str | None:
    refs = [r for r in ctx.benchmark_refs if r.get("target_value") is not None]
    if not refs:
        return None
    return _grouped(_TARGET_GROUP, refs, "target_position", lambda r: f"{r['metric']}(목표 {target_text(r['target_value'])})")


def _no_benchmark(ctx: Ctx) -> str | None:
    covered = {r["metric"] for r in ctx.benchmark_refs}
    missing = [m.display_name for m in _narrative_metrics(ctx.fs) if m.display_name not in covered]
    if not missing:
        return None
    return f"내부 참조 기준치가 없어 비교하지 않는 지표: {', '.join(missing)}(고정 문구 16)."


# ── 8.2 종합 총평 ─────────────────────────────────────────────────────────────

_CAUSE = {
    "core_not_computed": "핵심 지표 산출 실패",
    "no_target": "핵심 지표 목표값 없음",
    "no_judged": "판정 대상 지표 없음",
}


def _verdict_rule(ctx: Ctx) -> str | None:
    return "판정 규칙(핵심 지표 미달 → 불합격, 산출 실패·판정 대상 없음 → 판정 불가 …)은 6.1 표에 고정 문구 10 으로 찍힌다."


def _names(refs) -> str:
    return join_and([m.display_name for m in refs])


def _verdict_reason(ctx: Ctx) -> str | None:
    vd = ctx.facts.verdict_detail if ctx.facts else None
    name = VERDICT_LABEL.get(ctx.fs.verdict, ctx.fs.verdict)
    parts = [f"종합 판정은 '{name}'이다."]
    if vd is None:
        return parts[0]
    if vd.failed_core:
        parts.append(f"목표에 미달한 핵심 지표는 {_names(vd.failed_core)}이다.")
    if vd.inconclusive_cause:
        detail = f"({_names(vd.inconclusive_metrics)})" if vd.inconclusive_metrics else ""
        parts.append(f"판정 불가 원인은 {_CAUSE[vd.inconclusive_cause]}{detail}이다.")
    if vd.unavailable_core and vd.inconclusive_cause != "core_not_computed":
        parts.append(f"산출에 실패한 핵심 지표는 {_names(vd.unavailable_core)}이다.")
    if vd.failed_noncore:
        parts.append(f"목표에 미달한 비핵심 지표는 {_names(vd.failed_noncore)}이다.")
    if vd.judged_count:
        parts.append(f"참고로 판정 대상 {vd.judged_count}개 중 {vd.passed_count}개가 목표를 충족했다(통과율은 판정 근거가 아니다).")
    return " ".join(parts)


def _judged(fs: FactSheet) -> list:
    """목표값이 설정된 지표(판정 대상). 목표값 0·음수도 목표다 — 없음은 None 뿐."""
    return [m for m in fs.metrics if m.threshold is not None]


def failed_metrics(fs: FactSheet) -> list:
    """목표에 미달한 판정 대상 지표(목표값 0·음수 미달도 미달이다)."""
    return [m for m in fs.metrics if m.status == "fail" and m.threshold is not None]


def verdict_sentence(fs: FactSheet, derived: dict) -> str:
    """8.2 판정 문장(규칙 템플릿) — 판정 규칙(서술 설계 D19)의 사유로 쓴다. 통과율은 판정 근거가 아니라 참고다.

    종전에는 'N개 중 M개가 기준을 충족하여(통과율 x%), 종합 판정은 …'이라 써서, 불합격이 '대부분 충족해서
    불합격'처럼 읽히고 판정 불가에도 통과율이 근거처럼 붙었다(점검 BE-03).
    - FAIL: 핵심 지표 미달. 사실 시트에 is_core 가 없으면(옛 프론트) 어느 지표가 핵심인지 모르므로 미달 지표를
      적고 '핵심 지표가 목표에 미달'이라는 규칙만 쓴다(FAIL 이면 규칙상 핵심 미달이 있다).
    - CONDITIONAL_PASS: 핵심은 모두 충족, 핵심이 아닌 지표 미달(규칙상 미달 지표는 모두 비핵심이다).
    - INCONCLUSIVE: 문제 목록의 판정 불가 원인(evidence.cause·metrics). 문제 목록이 없으면 판정 사유(facts.verdict_detail),
      그것도 없으면(옛 프론트) 규칙의 두 경우.
    - PASS: 판정 대상 지표가 모두 목표 충족.
    통과율은 판정 대상(목표가 설정된 지표)이 있을 때만 '참고로' 덧붙인다.
    """
    vlabel = VERDICT_LABEL.get(fs.verdict, fs.verdict)
    failed = failed_metrics(fs)
    has_core_info = any(m.is_core for m in fs.metrics)

    def listing(label: str, metrics: list) -> str:
        # 지표 목록은 문장 끝 괄호에 둔다 — 이름 뒤 조사(가/이)를 이름 받침에 맞출 일이 없게.
        return f"({label}: {', '.join(metric_vs_target(m) for m in metrics)})" if metrics else ""

    if fs.verdict == "FAIL":
        core_failed = [m for m in failed if m.is_core] if has_core_info else []
        if core_failed:
            text = f"핵심 지표가 목표에 미달하여 종합 판정은 '{vlabel}'이다{listing('미달한 핵심 지표', core_failed)}."
            others = [m for m in failed if not m.is_core]
            if others:
                text += f" 핵심이 아닌 지표 중에도 목표에 미달한 지표가 있다{listing('목표 미달 지표', others)}."
        else:
            text = f"핵심 지표가 목표에 미달하여 종합 판정은 '{vlabel}'이다{listing('목표 미달 지표', failed)}."
    elif fs.verdict == "CONDITIONAL_PASS" and failed:
        text = (
            "핵심 지표는 모두 목표를 충족하였으나 핵심이 아닌 지표가 목표에 미달하여 "
            f"종합 판정은 '{vlabel}'이다{listing('목표 미달 지표', failed)}."
        )
    elif fs.verdict == "CONDITIONAL_PASS":
        # 운영 프론트(9-22)의 옛 판정 규칙은 미달이 없어도 조건부 합격을 낸다 — 목표값이 하나도 없을 때와
        # 핵심 지표가 산출되지 않았을 때. 그때 '비핵심 미달'이라 쓰면 거짓이다(S1 배포 전 점검에서 발견).
        if not _judged(fs):
            text = f"목표값이 설정된 지표가 없어 목표 충족 여부로 판정하지 못했으며, 종합 판정은 '{vlabel}'이다."
        else:
            text = (
                "목표값이 설정된 지표 중 목표에 미달한 지표는 없으나 핵심 지표가 산출되지 않아 "
                f"종합 판정은 '{vlabel}'이다."
            )
    elif fs.verdict == "INCONCLUSIVE":
        cause = next((p for p in fs.problems if p.kind == "inconclusive"), None)
        ev = cause.evidence if cause else {}
        metrics = [str(x) for x in (ev.get("metrics") or [])]
        vd = fs.facts.verdict_detail if fs.facts is not None else None
        if ev.get("cause"):
            what = f"{ev['cause']} — {', '.join(metrics)}" if metrics else str(ev["cause"])
            text = f"합격 여부를 판정할 수 없어 종합 판정은 '{vlabel}'이다(원인: {what})."
        elif vd is not None and vd.inconclusive_cause:
            names = _names(vd.inconclusive_metrics)
            what = f"{_CAUSE[vd.inconclusive_cause]} — {names}" if names else _CAUSE[vd.inconclusive_cause]
            text = f"합격 여부를 판정할 수 없어 종합 판정은 '{vlabel}'이다(원인: {what})."
        else:
            text = (
                "핵심 지표가 산출되지 않았거나 판정 대상 지표가 없어 합격 여부를 판정할 수 없으므로 "
                f"종합 판정은 '{vlabel}'이다."
            )
    elif fs.verdict == "PASS":
        text = f"목표값이 설정된 판정 대상 지표가 모두 목표를 충족하여 종합 판정은 '{vlabel}'이다."
    else:
        text = f"종합 판정은 '{vlabel}'이다."

    counts = derived.get("counts", {})
    target, passed = counts.get("target", 0), counts.get("passed", 0)
    if target > 0:
        text += f" 참고로 목표값이 설정된 지표 {target}개 중 {passed}개가 목표를 충족하였다(통과율 {fs.score}%)."
    return text


def _verdict_template(ctx: Ctx) -> str | None:
    return verdict_sentence(ctx.fs, ctx.derived)


def _ci_end(value: float, target: str) -> str:
    """구간 끝 — 성적서와 같은 셋째 자리. 그렇게 줄이면 찍힌 목표값과 같아 보이는데 실제로는 다르면 넷째 자리, 그래도 같으면
    반올림하지 않은 표기(허용 목록이 받는 'g' 표기) — 목표와 떨어진 구간이 목표에 닿은 것처럼 보이지 않게."""
    for text in (metric_num(float(value)), num(float(value)), f"{float(value):g}"):
        if float(text) != float(target) or float(value) == float(target):
            return text
    return f"{float(value):g}"


def _ci(ctx: Ctx) -> str | None:
    """8.2 신뢰구간 해석(D29) — 목표 포함 / 모두 떨어짐 두 갈래와, 구간을 내지 않은 판정 대상 지표의 사유."""
    judged = [m for m in _narrative_metrics(ctx.fs) if m.threshold is not None]
    if not judged:
        return None
    with_ci = [m for m in judged if m.ci_low is not None and m.ci_high is not None]
    missing = [m for m in judged if m not in with_ci]
    skipped = ctx.facts.ci_skipped if ctx.facts else []
    skipped_names = {s.display_name for s in skipped}
    skip_items = [f"{s.display_name}({s.reason})" if s.reason else s.display_name for s in skipped]
    skip_items += [m.display_name for m in missing if m.display_name not in skipped_names]
    skip_text = join_and(skip_items)
    if with_ci:
        contains = set(ctx.derived.get("ci_contains_target") or [])
        inside = [m for m in with_ci if m.display_name in contains]
        outside = [m for m in with_ci if m.display_name not in contains]
        sentences = []
        if inside:
            def item(m) -> str:
                # 목표값 — 새 요청(value_text 가 있음)은 반올림하지 않은 값, 옛 요청은 성적서처럼 셋째 자리(S10).
                target = target_text(m.threshold) if m.value_text else metric_num(m.threshold)
                return f"{m.display_name}(구간 {_ci_end(m.ci_low, target)}~{_ci_end(m.ci_high, target)}, 목표 {target})"

            items = join_and([item(m) for m in inside])
            sentences.append(
                f"95% 신뢰구간이 목표값을 포함하는 지표는 {with_josa(items, '으로/로')}, 표본 변동에 따라 해당 지표의 판정이 달라질 수 있다."
            )
            if outside:
                sentences.append(f"그 밖에 신뢰구간을 산출한 지표({_names(outside)})는 구간이 목표값과 떨어져 있다.")
        elif missing:
            # 구간이 없는 판정 대상이 있다 — '모두'라고 보증하지 않는다(S4b 검토 RI-02).
            sentences.append(f"신뢰구간을 산출한 지표({_names(with_ci)})는 95% 신뢰구간이 목표값과 떨어져 있다.")
        elif any(m.near_target for m in judged):
            # 근소 차이 지표(8.3·고정 문구 18)는 표본에 따라 바뀔 수 있다고 쓴다 — 여기서 '가능성은 낮다'고 하면 부딪친다.
            sentences.append("판정 대상 지표의 95% 신뢰구간은 모두 목표값과 떨어져 있다.")
        else:
            sentences.append(
                "판정 대상 지표의 95% 신뢰구간이 모두 목표값과 떨어져 있어, 표본 변동에 따라 판정이 달라질 가능성은 낮다."
            )
        if missing:
            sentences.append(f"신뢰구간을 산출하지 않은 판정 대상 지표는 {skip_text}이다.")
        return " ".join(sentences)
    if skipped:
        return f"판정 대상 지표의 신뢰구간은 산출하지 않았다(생략한 지표와 사유: {skip_text})."
    if ctx.facts is not None:
        # 평가 엔진이 구간을 보내지 않았다(facts.ci_available false) — 구간 해석을 지어내지 않고 그 사실만 쓴다.
        return "이 평가에서는 판정 대상 지표의 신뢰구간을 산출하지 않았다."
    return None


def _generalization(ctx: Ctx) -> str | None:
    sentence = ctx.fixed("11a") or ctx.fixed("11b")
    return sentence.text if sentence else None


def _metric_value(ctx: Ctx, metric_id: str, value: float) -> str:
    """파생값에 실린 값이 판정 대상 지표의 값이면 성적서 표기(value_text)로 — 넷째 자리로 줄이면 미달 값 0.69996 이 목표 0.7
    처럼 찍힌다(세션 A 검토). 그 밖에는 넷째 자리."""
    m = next((m for m in ctx.fs.metrics if m.metric_id == metric_id), None)
    if m is not None and m.threshold is not None and abs(float(m.value) - float(value)) < 1e-9:
        return shown_value(m)
    return num(float(value))


def _pr_gap(ctx: Ctx) -> str | None:
    g = ctx.facts.precision_recall_gap if ctx.facts else None
    if g is None or not {"M2", "M3"} <= ctx.selected:
        return None
    p, r = _metric_value(ctx, "M2", g.precision), _metric_value(ctx, "M3", g.recall)
    if g.difference == 0:
        return f"양성 클래스의 Precision 과 Recall 은 {with_josa(p, '으로/로')} 같다."
    higher = "Precision" if g.difference > 0 else "Recall"
    judge = "기준(0.1) 이상이어서 한쪽으로 치우쳐 있다" if g.flagged else "기준(0.1) 미만이어서 치우침은 크지 않다"
    return (
        f"양성 클래스의 Precision 은 {p}, Recall 은 {with_josa(r, '으로/로')} {higher} 쪽이 "
        f"{unsigned(g.difference_text)} 높고, 두 값의 차이는 {judge}."
    )


def _threshold_gain(ctx: Ctx) -> str | None:
    g = ctx.fs.f1_threshold_gain
    if g is None or not ctx.derived.get("threshold_tuning"):
        return None
    best = num(g.best_threshold)
    return (
        f"결정 임계값을 {with_josa(best, '으로/로')} 바꾸면 F1 이 {num(g.current_f1)}에서 {num(g.max_f1)}까지 오를 수 있다. "
        f"다만 이 향상폭({g.gain_text or num(g.gain)})은 같은 평가 데이터로 고른 임계값에서 나온 이 데이터의 최대폭이므로 "
        "별도 데이터에서 확인해야 한다."
    )


def _average_gap(ctx: Ctx, kind: str) -> str | None:
    g = ctx.facts.average_gap if ctx.facts else None
    if g is None or g.kind != kind or "M22" not in ctx.selected:
        return None
    other = "가중 평균" if kind == "macro_vs_weighted" else "마이크로 평균"
    size = unsigned(g.difference_text)
    # difference = 다른 평균 − 매크로(프론트 narrativeFacts.averageGap).
    if g.difference > 0:
        relation = f"{other} F1 이 매크로 F1 보다 {size} 높다"
    elif g.difference < 0:
        relation = f"{other} F1 이 매크로 F1 보다 {size} 낮다"
    else:
        relation = "두 값이 같다"
    text = (
        f"매크로 F1 은 {_metric_value(ctx, 'M4', g.macro_f1)}, {other} F1 은 {with_josa(num(g.other_f1), '으로/로')} {relation}."
    )
    ref = g.reference_class
    f1 = f"F1 {num(ref.f1)}" if ref.f1 is not None else None
    if kind == "macro_vs_weighted":
        detail = join_and([x for x in (f"표본 {ref.support}건" if ref.support is not None else None, f1) if x])
        text += f" 표본이 가장 많은 클래스는 {ctx.name(ref.label)}" + (f"이며 {detail}이다." if detail else "이다.")
    else:
        detail = join_and([x for x in (f1, f"표본 {ref.support}건" if ref.support is not None else None) if x])
        text += f" F1 이 가장 낮은 레이블은 {ctx.name(ref.label)}" + (f"이며 {detail}이다." if detail else "이다.")
    return text


def _hamming_exact(ctx: Ctx) -> str | None:
    by_id = {m.metric_id: m for m in _narrative_metrics(ctx.fs)}
    if "M15" not in by_id or "M16" not in by_id:
        return None
    h, e = by_id["M15"], by_id["M16"]
    return (
        f"{h.display_name}의 값은 {shown_value(h)}, {e.display_name}의 값은 {shown_value(e)}이다. "
        f"{with_josa(h.display_name, '은/는')} 레이블 칸 단위로 틀린 비율이고 {with_josa(e.display_name, '은/는')} 모든 레이블을 "
        "맞힌 표본의 비율이어서, 두 값으로 칸 단위 오류와 표본 단위 완전 일치를 나누어 볼 수 있다."
    )


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
    return f"의뢰자가 클래스 중요도를 {why} 밝혔으므로, 그에 맞춘 {name} F1 은 {num(d.f1)}이다."


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


def _purpose_fact(ctx: Ctx) -> str | None:
    """해석형 항목의 템플릿 — 사실 문장만(용도에 비춘 해석은 AI 몫)."""
    purpose = _model_purpose(ctx)
    return f"의뢰자가 밝힌 모델 용도는 '{purpose}'이다." if purpose else None


# ── 8.3 리스크 ───────────────────────────────────────────────────────────────


def _weakest_risk(ctx: Ctx) -> str | None:
    w = ctx.facts.weakest_class if ctx.facts else None
    if w is None:
        return None
    purpose = _model_purpose(ctx)
    noun = _noun(ctx)
    tail = f" 모델 용도('{purpose}')에서 그 {noun}가 가지는 위험을 쓴다." if purpose else ""
    return f"가장 약한 {_topic(ctx)} F1 {num(w.f1)}인 {ctx.name(w.label)}이다.{tail}"


def _weakest_risk_fact(ctx: Ctx) -> str | None:
    """해석형 항목의 템플릿 — 사실 문장만(용도상 위험의 해석은 AI 몫)."""
    w = ctx.facts.weakest_class if ctx.facts else None
    if w is None:
        return None
    return f"F1 이 가장 낮은 {_topic(ctx)} F1 {num(w.f1)}인 {ctx.name(w.label)}이다."


def _near(ctx: Ctx) -> list:
    return [m for m in _narrative_metrics(ctx.fs) if m.near_target and m.threshold is not None]


def _near_target(ctx: Ctx) -> str | None:
    """AI 에게 주는 해석 재료(숫자 없음 — 숫자는 고정 문구 18 이 찍는다): 근소 지표가 충족했는지 미달했는지."""
    near = _near(ctx)
    if not near:
        return None
    passed = [m.display_name for m in near if m.status == "pass"]
    failed = [m.display_name for m in near if m.status == "fail"]
    stems = []
    if passed:
        stems.append(f"목표를 충족한 지표는 {join_and(passed)}이")
    if failed:
        stems.append(f"목표에 미달한 지표는 {join_and(failed)}이")
    return "근소 차이 지표 가운데 " + _clauses(stems) if stems else "근소 차이 지표가 있다."


def _near_item(m) -> str:
    # 값·차이는 판정과 같은 쪽으로 읽히는 표기(value_text·margin_text)가 있으면 그것을 쓴다 — 셋째 자리
    # 반올림은 근소 차이를 '차이 0' 으로 만들었다.
    value = m.value_text or metric_num(m.value)
    target = target_text(m.threshold) if m.value_text else metric_num(m.threshold)
    if m.margin is None:
        return f"{m.display_name}({value}, 목표 {target})"
    return f"{m.display_name}({value}, 목표 {target}, 차이 {m.margin_text or metric_num(m.margin)})"


def _near_template(ctx: Ctx) -> str | None:
    near = _near(ctx)
    passed = [m for m in near if m.status == "pass"]
    failed = [m for m in near if m.status == "fail"]
    if ctx.prints("18"):
        # 숫자는 성적서가 8.3 서술 아래에 찍는다 — 해석 한 문장만(판정과 맞게: 충족 지표는 미달로, 미달 지표는 충족으로).
        if passed and failed:
            return "목표값에 근소한 차이로 걸친 지표는 표본이 달라지면 충족과 미달이 서로 바뀔 수 있어 판정의 여유가 작다."
        if passed:
            return "목표값에 근소한 차이로 충족한 지표는 표본이 달라지면 미달로 바뀔 수 있어 판정의 여유가 작다."
        if failed:
            return "목표값에 근소한 차이로 미달한 지표는 표본이 달라지면 충족으로 바뀔 수도 있어, 그 판정은 표본 변동에 민감하다."
        return None
    # 옛 요청 — 성적서가 18 을 찍지 않으니 합격한 근소 지표를 숫자와 함께 쓴다(종전 문장).
    if not passed:
        return None
    items = join_and([_near_item(m) for m in passed])
    # 숫자는 값·목표값·margin·3(%)뿐이라 허용 목록 안이다(3% 는 near_target 이 있을 때만 허용).
    return (
        f"목표값과의 차이가 목표값의 3% 미만인 아슬아슬 합격 지표는 {with_josa(items, '으로/로')}, "
        "표본이 달라지면 목표에 미달할 위험이 있다."
    )


def _threshold_risk(ctx: Ctx) -> str | None:
    if ctx.fs.f1_threshold_gain is None or not ctx.derived.get("threshold_tuning"):
        return None
    return "현재 결정 임계값이 최적이 아닐 수 있다(모델 쪽 임계값 조정 권고의 근거)."


def _threshold_risk_template(ctx: Ctx) -> str | None:
    if ctx.fs.f1_threshold_gain is None or not ctx.derived.get("threshold_tuning"):
        return None
    # 숫자(임계값·F1·향상폭)는 8.2 가 쓴다 — 같은 숫자를 두 칸에 되풀이하지 않는다.
    return "현재 결정 임계값이 최적이 아닐 수 있어, 임계값을 그대로 두면 이 데이터에서 가능한 것보다 낮은 F1 로 운영될 위험이 있다."


_LATENCY_NAMES = (("평균", "mean"), ("중앙값(p50)", "p50"), ("95번째 백분위수(p95)", "p95"), ("99번째 백분위수(p99)", "p99"))


def _latency(ctx: Ctx) -> str | None:
    lat = ctx.fs.latency
    if not lat.available:
        return None
    parts = [f"{label} {num(getattr(lat, key))}{lat.unit}" for label, key in _LATENCY_NAMES if getattr(lat, key) is not None]
    if not parts:
        return None
    return f"지연시간은 {join_and(parts)}이다."


def _realtime(ctx: Ctx) -> str | None:
    decl = ctx.facts.declarations if ctx.facts else None
    if decl is None or decl.usage_mode != "realtime" or not ctx.fs.latency.available:
        return None
    return "의뢰자가 대화형(실시간) 사용을 밝혔으므로 응답 시간(지연시간 값)을 그 기준으로 해석한다."


def _realtime_template(ctx: Ctx) -> str | None:
    if _realtime(ctx) is None:
        return None
    # 숫자는 같은 칸의 지연시간 문장(8.3-04)이 쓴다 — D24 확인 숫자가 그 문장에 있다.
    return "의뢰자가 대화형(실시간) 사용을 밝혔으므로, 이 지연시간은 사용자가 결과를 기다리는 응답 시간으로 보아야 한다."


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


def _no_template(ctx: Ctx) -> None:
    """템플릿이 찍지 않는 항목(9절 권고는 칸 템플릿이 문제 목록으로 따로 만든다 — app.narrative.fallback)."""
    return None


# ── 표(5절 41행) ──────────────────────────────────────────────────────────────

REQUIRED_ITEMS: tuple[RequiredItem, ...] = (
    RequiredItem("7.1-01", "7.1", "오분류 총건수·비율", ALL, "ai", "항상(혼동행렬·엔진 건수)", _misclassified),
    RequiredItem("7.1-02", "7.1", "오분류 방향 FP·FN", BINARY, "ai", "항상(엔진 건수)", _fp_fn),
    RequiredItem("7.1-03", "7.1", "선언한 위험 쪽 오류", BINARY, "ai", "위험 방향이 FN·FP", _declared_risk, d24=_declared_risk_values),
    RequiredItem("7.1-04", "7.1", "클래스별 FP·FN", MULTICLASS, "ai", "항상(엔진 건수)", _per_class_errors),
    RequiredItem("7.1-05", "7.1", "레이블별 FP·FN", MULTILABEL, "ai", "항상(엔진 건수)", _per_class_errors),
    RequiredItem("7.1-06", "7.1", "최다 오분류 쌍 상위 3", MULTICLASS, "ai", "항상(혼동행렬)", _top_pairs),
    RequiredItem("7.1-07", "7.1", "예측 편향(이진)", BINARY, "ai", "항상", lambda c: _bias(c, "positive_class")),
    RequiredItem("7.1-08", "7.1", "예측 편향(다중)", MULTICLASS, "ai", "항상", lambda c: _bias(c, "per_class")),
    RequiredItem("7.1-09", "7.1", "예측 편향(멀티)", MULTILABEL, "ai", "항상", lambda c: _bias(c, "label_total")),
    RequiredItem("7.1-10", "7.1", "최약 클래스·작은 표본", ALL, "ai", "항상(결정 A)", _weakest),
    RequiredItem("7.1-11", "7.1", "오분류의 성격(경계·확신 오류)", ALL, "fixed_interpret", "확률 컬럼", _error_nature, template=_error_nature_template),
    RequiredItem("7.1-12", "7.1", "사용된 결정 임계값", ALL, "fixed", "예측을 점수에서 파생", _derivation),
    RequiredItem("7.2-01", "7.2", "불균형 비율", ALL, "ai", "불균형 비율이 있음(멀티 정의는 고정 문구 7)", _imbalance_ratio, template=_imbalance_ratio_template),
    RequiredItem("7.2-02", "7.2", "불균형 영향 판단", ALL, "ai", "정답 + 예측(엔진 판단)", _imbalance_effect, template=_imbalance_effect_template),
    RequiredItem("7.2-03", "7.2", "학습↔평가 분포 차이", ALL, "ai", "학습 분포 숫자 입력", _distribution_shift),
    RequiredItem("7.2-04", "7.2", "정답 라벨 주의", ALL, "fixed", "신뢰성 검토 없음·구축 방법 규칙·운영 로그", _label_caution),
    RequiredItem("7.3-01", "7.3", "추가 관찰", ALL, "ai", "결론 목록 밖의 관찰", _additional, open_ended=True),
    RequiredItem("8.1-01", "8.1", "참조 범위 대비 위치", ALL, "ai", "기준표에 있는 지표", _benchmark),
    RequiredItem("8.1-02", "8.1", "목표값 수준", ALL, "ai", "기준표에 있는 지표 + 목표값", _target_level),
    RequiredItem("8.1-03", "8.1", "비교 기준 없음", ALL, "fixed", "기준표에 없는 지표", _no_benchmark),
    RequiredItem("8.2-01", "8.2", "판정 규칙", ALL, "form", "항상(6.1 표)", _verdict_rule),
    RequiredItem("8.2-02", "8.2", "종합 판정과 근거", ALL, "ai", "항상(판정 사유 D19, 통과율은 참고)", _verdict_reason, template=_verdict_template),
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
    RequiredItem("8.2-13", "8.2", "모델 용도에 비춘 해석", ALL, "ai_fact", "모델 용도", _purpose_reading, template=_purpose_fact),
    RequiredItem("8.3-01", "8.3", "최약 클래스의 용도상 위험", ALL, "ai_fact", "항상(결정 A) + 모델 용도", _weakest_risk, template=_weakest_risk_fact),
    RequiredItem("8.3-02", "8.3", "아슬아슬 지표", ALL, "fixed_interpret", "목표값 지표 중 근소 차이", _near_target, template=_near_template),
    RequiredItem("8.3-03", "8.3", "임계값 부적합 위험", BINARY, "ai", "엔진이 부적합으로 봄", _threshold_risk, template=_threshold_risk_template),
    RequiredItem("8.3-04", "8.3", "지연시간", ALL, "ai", "latency 컬럼", _latency),
    RequiredItem("8.3-05", "8.3", "대화형 용도의 응답 시간", ALL, "ai", "사용 방식이 대화형 + latency", _realtime, d24=_realtime_values, template=_realtime_template),
    RequiredItem("9-01", "9", "문제 목록", ALL, "form", "항상", _problem_table),
    RequiredItem("9-02", "9", "권고", ALL, "ai", "문제 1개 이상", _recommendations, template=_no_template),
    RequiredItem("9-03", "9", "권고 없음", ALL, "fixed", "문제 0개", _no_problem),
)

assert {i.section for i in REQUIRED_ITEMS} == set(SECTION_IDS), "필수 항목의 칸은 소절 7칸이다"
assert len({i.id for i in REQUIRED_ITEMS}) == len(REQUIRED_ITEMS), "항목 ID 가 겹친다"
assert all(i.template is not None for i in REQUIRED_ITEMS if i.author in ("ai_fact", "fixed_interpret")), \
    "해석형·고정 문구 + 해석 항목은 템플릿 문장을 따로 둔다(결론 문장은 AI 용이다)"
assert all(i.template is None for i in REQUIRED_ITEMS if i.author in ("fixed", "form") or i.open_ended), \
    "고정 문구·양식·추가 관찰 항목은 템플릿이 찍지 않는다"


# 표 지표(혼동행렬·클래스별 표)는 값이 없어 서술 지표가 아니다. 옛 요청(운영 main 프론트)은 value 0·info 줄로 싣는다.
_TABLE_METRICS = frozenset({"M21", "M22"})


def _narrative_metrics(fs: FactSheet) -> list:
    return [m for m in fs.metrics if m.metric_id not in _TABLE_METRICS]


def _selected_metrics(fs: FactSheet) -> set[str]:
    """의뢰자가 고른 지표. 새 요청은 facts.selected_metrics(표 지표 M21·M22 포함), 옛 요청은 지표 목록."""
    if fs.facts is not None and fs.facts.selected_metrics:
        return set(fs.facts.selected_metrics)
    return {m.metric_id for m in fs.metrics}


def infer_task_type(fs: FactSheet, derived: dict | None = None) -> str:
    """작업 유형을 모르는 옛 호출(build_fallback_narrative 를 task_type 없이 부른 경우)의 짐작 — 앱의 두 경로는 늘 요청의
    task_type 을 넘긴다. 이진 신호(양성 클래스·2×2·엔진 줄 하나)가 없으면 다중으로 본다(이진만의 주장 — '양성 클래스' — 을
    지어내지 않게). 멀티레이블은 행렬이 2×2 합이라 이것으로 가를 수 없어, 레이블 칸·레이블 등장 분모(호출부가 작업 유형을
    넘겨 만든 파생값의 unit, facts 의 분모)가 보일 때만 멀티로 본다."""
    if derived and (
        (derived.get("confusion") or {}).get("unit") == "label_cell"
        or (derived.get("distribution") or {}).get("unit") == "label"
    ):
        return "multilabel"
    facts = fs.facts
    if facts is not None:
        if facts.class_share_basis == "label_occurrences" or (
            facts.prediction_bias is not None and facts.prediction_bias.unit == "label_total"
        ):
            return "multilabel"
        if facts.positive_class is not None:
            return "binary"
    if len(fs.confusion_counts) == 1 or fs.f1_threshold_gain is not None:
        return "binary"  # 엔진 줄 하나·임계값 조정 여지(PR 곡선 — 이진에서만 계산)는 이진의 신호다
    if fs.confusion and len(fs.confusion.matrix) == 2 and not fs.confusion_counts:
        return "binary"
    return "multiclass"


def select_items(
    fs: FactSheet,
    client_inputs: dict | None,
    task_type: str | None,
    *,
    derived: dict | None = None,
    benchmark_refs: list[dict] | None = None,
) -> list[Conclusion]:
    """이 성적서에 해당하는 필수 항목만 남기고 숫자를 채운 결론 목록(표 순서 = 칸 순서).

    작업 유형이 맞지 않거나 재료가 없으면(결론 함수가 None) 뺀다. D24 항목은 확인할 숫자(d24_values)를 함께 싣는다.
    facts 가 없는 옛 요청에서도 깨지지 않고 받은 것만으로 고른다. task_type 이 없으면 infer_task_type 으로 짐작한다.
    derived·benchmark_refs 를 주면(규칙 템플릿 — 호출부가 이미 만든 값) 그것을 쓴다 — 칸 머리 문장과 같은 값을 읽게.
    """
    task = task_type or infer_task_type(fs, derived)
    ctx = Ctx(
        fs=fs,
        task_type=task,
        client_inputs=client_inputs if isinstance(client_inputs, dict) else {},
        derived=derived if derived is not None else compute_derived(fs, task_type=task),
        benchmark_refs=benchmark_refs if benchmark_refs is not None else build_benchmark_refs(task, fs.metrics),
        selected=_selected_metrics(fs),
        descriptions=class_descriptions(client_inputs),
    )
    out: list[Conclusion] = []
    for item in REQUIRED_ITEMS:
        if task not in item.task_types:
            continue
        text = item.build(ctx)
        if text is None:
            continue
        values = item.d24(ctx) if item.d24 is not None else ()
        if item.template is not None:
            template_text = item.template(ctx)
        elif item.author == "ai" and not item.open_ended:
            template_text = text
        else:
            template_text = None
        out.append(Conclusion(item.id, item.section, item.author, text, tuple(values), item.open_ended, template_text))
    return out


def conclusions_by_section(conclusions: list[Conclusion]) -> dict[str, list[Conclusion]]:
    """칸별로 묶는다(소절 7칸 순서, 결론이 없는 칸은 빈 목록)."""
    grouped: dict[str, list[Conclusion]] = {sid: [] for sid in SECTION_IDS}
    for c in conclusions:
        grouped[c.section].append(c)
    return grouped


# 옛 이름(fact_view 가 읽는다).
_VERDICT_LABEL = VERDICT_LABEL
