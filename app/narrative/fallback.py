"""
narrative_fallback.py — 규칙 기반 폴백 서술 생성.

무키 / LLM 호출 실패 / grounding 검증 실패 시 사용. 절대 가짜 MOCK 수치로 회귀하지 않고,
fact_sheet 의 실제 계산값(+ compute_derived 파생값)만 템플릿에 삽입한다(환각 0 보장).
출력의 모든 숫자는 build_number_whitelist 가 포함하는 값이므로 grounding 검증을 통과한다.
"""
from app.narrative.schemas import ConclusionOut, FactSheet, GroundingInfo, InterpretationOut, NarrativeMeta, NarrativeResponse, ProblemFact, RecommendationNarrativeOut, RecommendationOut

_VERDICT_LABEL = {
    "PASS": "적합(합격)",
    "CONDITIONAL_PASS": "조건부 적합",
    "FAIL": "부적합(불합격)",
    "INCONCLUSIVE": "판정 불가",
}
_POSITION_LABEL = {"above": "상위", "within": "범위 내", "below": "하위"}
# 방향 반영 품질 라벨(D7[3]) — 낮을수록 좋은 지표의 'below'는 미흡이 아니라 우수.
_QUALITY_LABEL = {"better": "우수", "within": "기준 범위 내", "worse": "미흡"}
# 목표값 수준(D31) — 단순 수치 위치라 방향(우수/미흡)으로 바꾸지 않는다.
_TARGET_POSITION_LABEL = {"below": "참조 범위보다 낮음", "within": "참조 범위 안", "above": "참조 범위보다 높음"}


def _pct(ratio: float) -> str:
    """비율을 백분율 소수 1자리로 — 허용 목록(_add)의 백분율 표기와 같은 반올림."""
    return f"{round(ratio * 100, 1):g}%"


def _error_ratio_sentence(er) -> str:
    """7.1 오류의 성격 한 문장(서술 설계 D30). 숫자는 error_ratios 의 값과 두 기준뿐이다."""
    unit = "틀린 레이블 칸" if er.unit == "label_cell" else "오분류"
    if er.misclassified == 0:
        subject = "틀린 레이블 칸이" if er.unit == "label_cell" else "오분류가"
        return f"확률이 있는 {subject} 없어 경계 오류·확신 오류 비율은 산출하지 않았다."

    def seg(part) -> str:
        return f"{part.count}건({_pct(part.ratio)})" if part.ratio is not None else f"{part.count}건"

    confident = (
        f"틀린 쪽 확률이 {er.confident_level:g} 이상인 확신 오류는 {seg(er.confident)}이다."
    )
    if er.boundary is not None:
        return (
            f"{unit} {er.misclassified}건 중 결정 경계와의 거리가 {er.boundary_margin:g} 미만인 "
            f"경계 오류는 {seg(er.boundary)}, {confident}"
        )
    if er.boundary_note == "threshold_unknown":
        reason = "예측값을 직접 제출해 결정 임계값을 알 수 없으므로 경계 오류는 계산하지 않았고, "
    else:
        reason = "경계 오류는 계산하지 않았고, "
    return f"{unit} {er.misclassified}건 중 {reason}{confident}"


def _imbalance_sentence(ie) -> str:
    """7.2 불균형 결론 한 문장(서술 설계 D20). level 판정은 평가 엔진이 했고 여기서는 읽기만 한다.

    숫자는 imbalance_effect 의 값과 두 기준(1.5·0.1)뿐이라 허용 목록 안이다. 정확도 − 매크로 F1 이
    없으면(멀티레이블) 다수·소수 '레이블'의 F1 차이만 쓴다.
    """
    ir = ie.imbalance_ratio
    if ie.level == "mild":
        return f"불균형 비율은 {ir}로 1.5 미만이어서 불균형은 경미하다."
    unit = "레이블" if ie.accuracy_minus_macro_f1 is None else "클래스"
    gaps = []
    if ie.accuracy_minus_macro_f1 is not None:
        gaps.append(f"정확도와 매크로 F1 의 차이는 {ie.accuracy_minus_macro_f1}")
    gaps.append(
        f"다수 {unit}({ie.majority_class})와 소수 {unit}({ie.minority_class})의 F1 차이는 "
        f"{ie.majority_minus_minority_f1}"
    )
    gap_text = ", ".join(gaps)
    if ie.level == "affected":
        return (
            f"불균형 비율은 {ir}로 불균형이 있고, {gap_text}로 0.1 이상 벌어져 불균형이 성능 수치에 "
            f"영향을 준 것으로 보인다. 소수 {unit}의 성능 지표 해석에 유의가 필요하다."
        )
    return (
        f"불균형 비율은 {ir}로 불균형이 있으나, {gap_text}로 0.1 미만이어서 불균형에 의한 성능 왜곡은 "
        "관찰되지 않았다."
    )


def _interpretation(fs: FactSheet, derived: dict) -> InterpretationOut:
    conf = derived.get("confusion")
    if conf:
        parts = [
            f"혼동 행렬 기준 전체 {conf['total']}건 중 {conf['misclassified']}건이 오분류되었다"
            f"(정분류 {conf['correct']}건)."
        ]
        if "fn" in conf:
            parts.append(f"거짓 음성(FN)은 {conf['fn']}건, 거짓 양성(FP)은 {conf['fp']}건이다.")
        confusion_analysis = " ".join(parts)
    else:
        confusion_analysis = "혼동 행렬 데이터가 제공되지 않아 클래스 간 오분류 분석을 생략한다."
    if fs.error_ratios is not None:
        confusion_analysis += " " + _error_ratio_sentence(fs.error_ratios)

    dist = derived.get("distribution")
    if dist and fs.distribution:
        cd = fs.distribution.class_distribution
        pct = dist["percentages"]
        items = ", ".join(f"{k} {v}건({pct.get(k, 0)}%)" for k, v in cd.items())
        s = f"평가 데이터셋은 총 {dist['total']}건이며, 클래스 분포는 {items}이다."
        ir = fs.distribution.imbalance_ratio
        # 불균형의 결론은 평가 엔진이 낸 imbalance_effect.level 로만 쓴다(D20). 종전에는 여기서
        # IR ≤ 1.5 를 '균형'으로 따로 판정해, 엔진의 '1.5 미만 = 경미'와 경계가 어긋났다.
        # 키가 없는 요청(옛 프론트)에는 판정 없이 비율만 적는다 — 근거 없는 결론을 만들지 않는다.
        if fs.imbalance_effect is not None:
            s += " " + _imbalance_sentence(fs.imbalance_effect)
        elif ir is not None:
            s += f" 불균형 비율은 {ir}이다."
        distribution_analysis = s
    else:
        distribution_analysis = "클래스 분포 데이터가 제공되지 않아 분포 분석을 생략한다."

    return InterpretationOut(
        confusion_analysis=confusion_analysis,
        distribution_analysis=distribution_analysis,
    )


def _conclusion(fs: FactSheet, benchmark_refs: list[dict], derived: dict) -> ConclusionOut:
    if benchmark_refs:
        segs = [
            f"{r['metric']} {r['model_value']}는 내부 참조 기준 {r['ref_low']}~{r['ref_high']} 대비 "
            f"{_QUALITY_LABEL.get(r.get('quality'), _POSITION_LABEL.get(r['position'], r['position']))}"
            for r in benchmark_refs
        ]
        benchmark = "; ".join(segs) + f". ({benchmark_refs[0]['source_note']})"
        # 목표값 수준(D31): 목표값이 있는 지표만. 숫자는 목표값(threshold)뿐이라 허용 목록 안이다.
        targets = [
            f"{r['metric']} {r['target_value']}"
            f"({_TARGET_POSITION_LABEL.get(r['target_position'], r['target_position'])})"
            for r in benchmark_refs
            if r.get("target_value") is not None
        ]
        if targets:
            benchmark += f" 목표값 수준은 {', '.join(targets)}이다."
    else:
        benchmark = "비교 가능한 내부 참조 기준치가 없어 벤치마크 비교를 생략한다."

    counts = derived.get("counts", {})
    target = counts.get("target", 0)
    passed = counts.get("passed", 0)
    vlabel = _VERDICT_LABEL.get(fs.verdict, fs.verdict)
    narrative = (
        f"임계값이 설정된 시험항목 {target}개 중 {passed}개가 기준을 충족하여"
        f"(통과율 {fs.score}%), 종합 판정은 '{vlabel}'이다."
    )
    failed_metrics = [
        m for m in fs.metrics
        if m.status == "fail" and m.threshold is not None  # 목표값 0·음수 미달도 미달이다
    ]
    failed = [m.display_name for m in failed_metrics]
    if failed:
        narrative += f" 기준 미달 항목은 {', '.join(failed)}이다."
    # 신뢰구간이 목표값을 포함하는 지표(D29) — 판정은 compute_derived 가 한 곳에서 한다.
    # 8.2(종합 총평)에 둔다: 판정을 얼마나 믿을 수 있는지는 판정을 설명하는 자리의 말이다.
    # 숫자는 구간 양 끝·목표값·신뢰수준(95)뿐이라 허용 목록 안이다.
    ci_names = set(derived.get("ci_contains_target") or [])
    if ci_names:
        items = ", ".join(
            f"{m.display_name}(구간 {m.ci_low}~{m.ci_high}, 목표 {m.threshold})"
            for m in fs.metrics
            if m.display_name in ci_names
        )
        narrative += (
            f" 95% 신뢰구간이 목표값을 포함하는 지표는 {items}로, "
            "표본 변동에 따라 해당 지표의 판정이 달라질 수 있다."
        )

    # 아슬아슬 합격(D12) — near_target 판정은 프론트가 했고 여기서는 합격한 지표만 골라 읽는다.
    near = [m for m in fs.metrics if m.near_target and m.status == "pass"]
    if failed_metrics:
        items = ", ".join(
            f"{m.display_name}({m.value}, 기준 {m.threshold})" for m in failed_metrics
        )
        risks = f"기준 미달 지표는 {items}로, 해당 영역의 성능 저하 위험이 존재한다."
    elif near:
        # "식별된 기준 미달 위험은 없다"와 아슬아슬 문장이 함께 실리면 서로 어긋나므로 앞 문장을 바꾼다.
        risks = "임계값이 설정된 모든 시험항목이 기준을 충족하였다."
    else:
        risks = "임계값이 설정된 모든 시험항목이 기준을 충족하여 식별된 기준 미달 위험은 없다."
    if near:
        # 숫자는 값·목표값·margin·3(%)뿐이라 허용 목록 안이다(3% 는 near_target 이 있을 때만 허용).
        items = ", ".join(
            f"{m.display_name}({m.value}, 기준 {m.threshold}, 차이 {m.margin})"
            if m.margin is not None
            else f"{m.display_name}({m.value}, 기준 {m.threshold})"
            for m in near
        )
        risks += (
            f" 목표값과의 차이가 목표값의 3% 미만인 아슬아슬 합격 지표는 {items}로, "
            "표본이 달라지면 기준에 미달할 위험이 있다."
        )
    # 임계값 조정 여지(D40) — 기준(0.10) 판정은 compute_derived 가 한 곳에서 한다.
    if derived.get("threshold_tuning"):
        g = fs.f1_threshold_gain
        risks += (
            f" 결정 임계값을 {g.best_threshold}로 바꾸면 F1 이 {g.current_f1}에서 {g.max_f1}까지 오를 수 있어 "
            f"현재 임계값이 최적이 아닐 수 있다. 다만 이 {g.gain}는 같은 평가 데이터로 고른 임계값에서 나온 "
            "'이 데이터에서 오를 수 있는 최대폭'이므로 별도 데이터에서 확인해야 한다."
        )

    return ConclusionOut(
        verdict=fs.verdict, benchmark=benchmark, narrative=narrative, risks=risks
    )


# 문제 종류별 권고 템플릿(서술 설계 D22) — (분류, 권고, 기대 효과). 9절 권고표의 한 줄이 된다.
# 기대 효과에는 숫자를 쓰지 않는다: 개선 폭은 이 평가로 알 수 없는 예측값이라, 숫자로 쓰면 근거 없는
# 약속이 된다(LLM 권고도 같은 규칙으로 막는다 — service._impact_has_number).
# 권고 문장에도 숫자를 넣지 않는다. 무엇이 문제인지(지표·클래스 이름과 수치)는 9절 표의 "대응 문제 번호"
# 칸(P1 · 문제 이름)과 원천 절이 이미 말한다.
_PROBLEM_TEMPLATES: dict[str, tuple[str, str, str]] = {
    "core_fail": (
        "핵심 지표 개선",
        "미달한 핵심 지표의 오분류 사례를 분석하고 재학습·데이터 보강·결정 임계값 재조정을 검토한 뒤 재평가한다",
        "핵심 지표 충족 여부 재확인 및 종합 판정 개선 근거 확보",
    ),
    "inconclusive": (
        "판정 조건 보완",
        "핵심 지표가 산출되도록 입력 데이터와 지표 선택을 점검하고, 판정 대상 지표의 목표값을 정한 뒤 재평가한다",
        "합격 여부를 판정할 수 있는 시험 결과 확보",
    ),
    "noncore_fail": (
        "보조 지표 개선",
        "미달한 비핵심 지표가 운영 목적에 주는 영향을 검토하고, 필요하면 재학습·결정 임계값 조정으로 보완한다",
        "보조 지표 미달에 따른 운영 위험 감소",
    ),
    "core_near_target": (
        "판정 안정성 확인",
        "목표값에 근접해 합격한 핵심 지표를 별도 평가 데이터나 더 큰 표본으로 다시 측정해 합격이 유지되는지 확인한다",
        "표본 변동에 따른 판정 뒤집힘 위험 확인",
    ),
    "threshold_tuning": (
        "결정 임계값 조정",
        "평가 데이터와 분리된 검증 데이터로 결정 임계값을 다시 고르고, 같은 조건으로 재평가한다",
        "현재 임계값이 최적이 아니어서 생기는 성능 손실 축소",
    ),
    "imbalance_effect": (
        "데이터 균형",
        "소수 클래스 데이터를 보강하거나 클래스 가중치·재표집으로 불균형을 완화한 뒤 재평가한다",
        "소수 클래스 성능 개선 및 불균형에 의한 성능 왜곡 축소",
    ),
    "weakest_class": (
        "취약 클래스 보강",
        "성능이 가장 낮은 클래스의 데이터를 추가 수집하고 오분류 사례를 분석해 보강한다",
        "취약 클래스 성능 개선 및 클래스 간 성능 편차 축소",
    ),
    "distribution_shift": (
        "분포 일치 확인",
        "학습 데이터와 평가 데이터의 클래스 비율이 다른 원인을 확인하고, 운영 환경 분포를 반영한 데이터로 재평가한다",
        "운영 환경 성능 추정의 정확도 향상",
    ),
    "label_caution": (
        "정답 라벨 검증",
        "정답 라벨 일부를 표본 추출해 사람이 검토하고 라벨 오류 비율을 확인한다",
        "정답 오류가 성능 수치에 주는 영향 파악",
    ),
}

# 편향 감소 노력을 "수행 안 함"으로 밝힌 불균형 영향(note 가 있음) — 권고 앞에 그 사실을 붙인다.
_IMBALANCE_NOTE_PREFIX = "의뢰자가 편향 감소 노력을 수행하지 않았다고 밝혔으므로, "


def problem_recommendation(problem: ProblemFact) -> RecommendationOut:
    """문제 하나의 폴백 권고(종류별 템플릿). 우선순위·번호는 문제의 것을 그대로 쓴다.

    LLM 이 권고를 하나도 달지 않은 문제를 채울 때도(service) 이 함수를 쓴다 — 문제마다 권고가 하나 이상
    있어야 9절 표가 문제 목록과 맞는다.
    """
    category, action, impact = _PROBLEM_TEMPLATES[problem.kind]
    if problem.kind == "imbalance_effect" and problem.note:
        action = _IMBALANCE_NOTE_PREFIX + action
    return RecommendationOut(
        problem_id=problem.id,
        priority=problem.priority,
        category=category,
        action=action,
        expected_impact=impact,
    )


def _problem_names(problems: list[ProblemFact]) -> str:
    return ", ".join(f"{p.id}({p.title})" for p in problems)


def _recommendations(fs: FactSheet):
    """9절 — 권고는 문제 목록(problems)에만 묶는다(서술 설계 D22).

    종전에는 문제와 무관하게 권고를 지어냈다: 아무 문제가 없어도 '운영 모니터링'을 넣고, 제외 행·
    불균형 비율 1.5 초과·미달 지표마다 따로 권고를 만들었다. 문제 목록의 조건(불균형은 '영향 있음'만,
    미달은 판정 사유)과 어긋나 9절이 문제 목록에 없는 문제를 권고했다. 이제 문제가 없으면 권고도 없다.

    9.1(data_quality)은 데이터 쪽 문제(recommend_section 9.1), 9.2(model_ops)는 모델 쪽 문제만 말한다.
    제외 행 수는 사실로만 적는다 — 문제 목록에 없으니 권고하지 않는다.
    """
    data_side = [p for p in fs.problems if p.recommend_section == "9.1"]
    model_side = [p for p in fs.problems if p.recommend_section != "9.1"]

    if data_side:
        data_quality = (
            f"데이터 측면에서 지목된 문제는 {_problem_names(data_side)}이며, "
            "9절 권고표의 해당 항목에 따른 보완 후 재평가를 권고한다."
        )
    else:
        data_quality = "데이터 측면에서 권고 대상으로 지목된 문제는 없다."
    if fs.dropped_rows > 0:
        data_quality += f" 전처리에서 {fs.dropped_rows}건이 평가에서 제외되었다."

    if model_side:
        model_ops = (
            f"모델 측면에서 지목된 문제는 {_problem_names(model_side)}이며, "
            "9절 권고표의 해당 항목에 따른 개선 후 재평가를 권고한다."
        )
    else:
        model_ops = "모델 측면에서 권고 대상으로 지목된 문제는 없다."

    recs = [problem_recommendation(p) for p in fs.problems]
    return RecommendationNarrativeOut(data_quality=data_quality, model_ops=model_ops), recs


def build_fallback_narrative(
    fs: FactSheet,
    benchmark_refs: list[dict],
    derived: dict,
    reason: str,
) -> NarrativeResponse:
    """규칙 기반 폴백 서술. reason: no_key | api_error | grounding_failed"""
    rec_narrative, recs = _recommendations(fs)
    return NarrativeResponse(
        interpretation=_interpretation(fs, derived),
        conclusion=_conclusion(fs, benchmark_refs, derived),
        recommendation_narrative=rec_narrative,
        recommendations=recs,
        meta=NarrativeMeta(
            source="fallback",
            model=None,
            reason=reason,
            grounding=GroundingInfo(passed=True),
        ),
    )
