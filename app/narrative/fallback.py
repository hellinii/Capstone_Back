"""
narrative_fallback.py — 규칙 기반 폴백 서술 생성.

무키 / LLM 호출 실패 / grounding 검증 실패 시 사용. 절대 가짜 MOCK 수치로 회귀하지 않고,
fact_sheet 의 실제 계산값(+ compute_derived 파생값)만 템플릿에 삽입한다(환각 0 보장).
출력의 모든 숫자는 build_number_whitelist 가 포함하는 값이므로 grounding 검증을 통과한다.
"""
from app.core.schemas import (
    FactSheet,
    NarrativeResponse,
    InterpretationOut,
    ConclusionOut,
    RecommendationNarrativeOut,
    RecommendationOut,
    NarrativeMeta,
    GroundingInfo,
)

_VERDICT_LABEL = {
    "PASS": "적합(합격)",
    "CONDITIONAL_PASS": "조건부 적합",
    "FAIL": "부적합(불합격)",
}
_POSITION_LABEL = {"above": "상위", "within": "범위 내", "below": "하위"}
# 방향 반영 품질 라벨(D7[3]) — 낮을수록 좋은 지표의 'below'는 미흡이 아니라 우수.
_QUALITY_LABEL = {"better": "우수", "within": "기준 범위 내", "worse": "미흡"}


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

    dist = derived.get("distribution")
    if dist and fs.distribution:
        cd = fs.distribution.class_distribution
        pct = dist["percentages"]
        items = ", ".join(f"{k} {v}건({pct.get(k, 0)}%)" for k, v in cd.items())
        s = f"평가 데이터셋은 총 {dist['total']}건이며, 클래스 분포는 {items}이다."
        ir = fs.distribution.imbalance_ratio
        if ir is not None:
            bal = (
                "균형 상태이다"
                if ir <= 1.5
                else "불균형 상태로, 소수 클래스의 성능 지표 해석에 유의가 필요하다"
            )
            s += f" 불균형 비율은 {ir}로 {bal}."
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
    failed = [m.display_name for m in fs.metrics if m.status == "fail"]
    if failed:
        narrative += f" 기준 미달 항목은 {', '.join(failed)}이다."

    failed_metrics = [m for m in fs.metrics if m.status == "fail"]
    if failed_metrics:
        items = ", ".join(
            f"{m.display_name}({m.value}, 기준 {m.threshold})" for m in failed_metrics
        )
        risks = f"기준 미달 지표는 {items}로, 해당 영역의 성능 저하 위험이 존재한다."
    else:
        risks = "임계값이 설정된 모든 시험항목이 기준을 충족하여 식별된 기준 미달 위험은 없다."

    return ConclusionOut(
        verdict=fs.verdict, benchmark=benchmark, narrative=narrative, risks=risks
    )


def _recommendations(fs: FactSheet):
    ir = fs.distribution.imbalance_ratio if fs.distribution else None
    failed = [m for m in fs.metrics if m.status == "fail"]
    failed_names = [m.display_name for m in failed]

    # 9.1 / 9.2 서술
    dq_segs = []
    if ir is not None and ir > 1.5:
        dq_segs.append(
            f"클래스 불균형 비율 {ir}가 허용 기준을 초과하므로, 오버샘플링(SMOTE 등) 또는 "
            "클래스 가중치 조정을 통한 균형화 후 재평가를 권고한다."
        )
    if fs.dropped_rows > 0:
        dq_segs.append(
            f"전처리에서 {fs.dropped_rows}건이 제외되었으므로, 결측·오류 데이터 정제를 통한 "
            "유효 표본 확보를 권고한다."
        )
    if not dq_segs:
        dq_segs.append("데이터 품질·균형은 허용 범위로 판단되며, 현 데이터셋 기준 추가 보완은 필수적이지 않다.")
    data_quality = " ".join(dq_segs)

    if failed_names:
        model_ops = (
            f"기준 미달 지표({', '.join(failed_names)})의 개선을 위해 임계값 재조정·재학습을 검토하고, "
            "운영 단계에서 해당 지표의 주기적 모니터링을 권고한다."
        )
    else:
        model_ops = "현 성능 수준 유지를 위해 운영 단계의 주기적 성능 모니터링과 데이터 드리프트 감시를 권고한다."

    # 권고 표 (정적 규칙 매핑)
    recs: list[RecommendationOut] = []
    for m in failed:
        recs.append(RecommendationOut(
            priority="HIGH",
            category=f"{m.display_name} 개선",
            action=f"{m.display_name} 기준 미달({m.value}) — 임계값 재조정/재학습/데이터 보강 검토",
            expected_impact=f"{m.display_name} 향상 및 종합 판정 개선",
        ))
    if ir is not None and ir > 1.5:
        recs.append(RecommendationOut(
            priority="MEDIUM",
            category="데이터 균형",
            action=f"클래스 비율 균형화(불균형 비율 {ir}) 후 추가 평가",
            expected_impact="불균형에 의한 성능 왜곡 사전 검증",
        ))
    if fs.dropped_rows > 0:
        recs.append(RecommendationOut(
            priority="LOW",
            category="데이터 품질",
            action=f"결측·오류 데이터 정제(제외 {fs.dropped_rows}건)",
            expected_impact="유효 표본 확대 및 평가 신뢰도 향상",
        ))
    if not recs:
        recs.append(RecommendationOut(
            priority="LOW",
            category="운영 모니터링",
            action="운영 단계 성능·데이터 드리프트 주기적 모니터링 체계 수립",
            expected_impact="성능 저하 조기 감지",
        ))

    return RecommendationNarrativeOut(data_quality=data_quality, model_ops=model_ops), recs[:5]


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
