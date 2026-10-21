"""
app/narrative/fallback.py — 규칙 기반 문장(규칙 템플릿).

무키 / LLM 호출 실패 / 검문 실패 시 사용. 절대 가짜 MOCK 수치로 회귀하지 않고, fact_sheet 의 실제 계산값
(+ compute_derived 파생값)만 템플릿에 삽입한다(환각 0 보장). 출력의 모든 숫자는 build_number_whitelist 가 포함하는
값이므로 grounding 검증을 통과한다(작업 유형 × 판정 자기 검문 — tests/test_s6_templates.py).

문장은 소절 7칸 함수(SECTION_TEMPLATES, 서술 라운드 S5)로 나뉜다 — 작업 경로의 검문 사다리(app.narrative.generator)가
걸린 칸만 그 칸 함수로 바꾸고, 옛 경로(app.narrative.legacy)·키 없음은 일곱 칸을 모은 build_fallback_narrative 를 쓴다.

칸 함수는 AI 지시문과 같은 결론 목록(app.narrative.required_items.select_items — 서술 설계 5절, 계획 원칙 2)을 읽어 칸마다
결론의 템플릿 문장(Conclusion.template_text)을 잇는다(서술 라운드 S6). 결론 목록에 없는 문장은 칸의 앞뒤 사실 문장뿐이다:
7.1 혼동행렬이 없다는 말, 7.2 클래스 분포 나열, 8.1 기준의 출처, 8.3 목표 미달 위험, 9절 9.1·9.2 문장과 권고표.
- 고정 문구 17·18 을 성적서가 따로 찍으면(새 프론트) 그 숫자를 되풀이하지 않고 해석 한 문장만(required_items.code_prints).
- 해석형 항목(모델 용도·최약 클래스의 용도상 위험)은 사실 문장만.
- 작성 규칙(D37): 클래스는 '이름(설명)', 영어 코드값·필드명 없음, 서술체.
"""
from app.narrative.required_items import (
    Conclusion,
    failed_metrics,
    infer_task_type,
    select_items,
)
from app.narrative.schemas import ConclusionOut, FactSheet, GroundingInfo, InterpretationOut, NarrativeMeta, NarrativeResponse, ProblemFact, RecommendationNarrativeOut, RecommendationOut
from app.narrative.wording import VERDICT_LABEL, class_descriptions, class_name, metric_vs_target

# 판정 이름 — 성적서 8.2 판정 상자와 같은 이름 한 벌(app.narrative.wording). 옛 이름으로도 읽힌다.
_VERDICT_LABEL = VERDICT_LABEL


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


# ── 칸별 규칙 템플릿(소절 7칸, 서술 설계 D36) ────────────────────────────────────
# 검문 사다리(app.narrative.generator)는 걸린 칸만 이 함수로 바꾼다. 칸 함수는 그 칸의 응답 필드(점 표기,
# app.narrative.sections 의 response_paths)마다 값을 돌려준다. 칸마다 결론 목록의 템플릿 문장을 표 순서대로 잇는다(S6).
# 7.3(추가 관찰)은 규칙 문장이 없다 — 빈 글이면 성적서가 고정 문구 15("해당 없음")를 찍는다(D38).


class _Input:
    """칸 함수 한 번의 재료 — 사실 시트·기준표·파생값·결론 목록·클래스 설명."""

    def __init__(self, fs: FactSheet, refs: list[dict], derived: dict, conclusions: list[Conclusion], descriptions: dict):
        self.fs, self.refs, self.derived, self.conclusions, self.descriptions = fs, refs, derived, conclusions, descriptions

    def lines(self, section_id: str) -> list[str]:
        return [c.template_text for c in self.conclusions if c.section == section_id and c.template_text]

    def has(self, item_id: str) -> bool:
        return any(c.item_id == item_id and c.template_text for c in self.conclusions)

    def name(self, label) -> str:
        return class_name(label, self.descriptions)


def _section_7_1(inp: _Input) -> dict:
    """7.1 혼동행렬 해석 — 오분류 총건수·방향·선언한 위험 쪽 오류·클래스별 오류·헷갈린 쌍·예측 편향·최약 클래스·오류의 성격."""
    lines = inp.lines("7.1")
    if not any(inp.has(i) for i in ("7.1-01", "7.1-02", "7.1-04", "7.1-05")):
        lines.insert(0, "혼동 행렬 데이터가 제공되지 않아 클래스 간 오분류 분석을 생략한다.")
    return {"interpretation.confusion_analysis": " ".join(lines)}


def _distribution_intro(inp: _Input) -> str | None:
    """7.2 머리 — 클래스 분포 나열(결론 목록 밖의 사실 문장)."""
    fs, dist = inp.fs, inp.derived.get("distribution")
    if dist and fs.distribution:
        pct = dist["percentages"]
        items = ", ".join(f"{inp.name(k)} {v}건({pct.get(k, 0)}%)" for k, v in fs.distribution.class_distribution.items())
        # 프론트가 클래스 상한(256)에 맞춰 분포를 잘랐으면 잘린 목록을 전체처럼 쓰지 않는다(S4a 미리 자르기).
        cut = fs.facts.omitted.get("class_distribution", 0) if fs.facts is not None else 0
        if cut:
            items += f" 외 {cut}개 {'레이블' if dist.get('unit') == 'label' else '클래스'}"
        if dist.get("unit") == "label":
            return (
                f"평가 데이터셋은 총 {dist['total']}건이며, 레이블별 등장 분포는 {items}이다"
                "(한 표본에 레이블이 여럿일 수 있어 비율을 모두 더하면 전체를 넘을 수 있다)."
            )
        return f"평가 데이터셋은 총 {dist['total']}건이며, 클래스 분포는 {items}이다."
    if fs.imbalance_effect is not None:
        # 클래스가 많아 프론트가 분포 표를 비웠어도(본문 예산) 엔진의 불균형 결론은 남는다 — 9절에 불균형 문제가 있는데
        # 7.2 가 분포를 모른다고 쓰지 않게(S4b 검토 FB-02).
        cut = fs.facts.omitted.get("class_distribution", 0) if fs.facts is not None else 0
        head = f"클래스가 많아 클래스별 건수({cut}개)는 생략했다." if cut else "클래스별 건수는 싣지 않았다."
        return f"평가 데이터셋은 총 {fs.n_samples}건이며, {head}"
    return None


def _section_7_2(inp: _Input) -> dict:
    """7.2 분포·불균형 — 분포 나열 + 불균형 비율·불균형 영향(엔진 판단, D20)·학습↔평가 분포 차이."""
    intro = _distribution_intro(inp)
    lines = inp.lines("7.2")
    if intro is None and not lines:
        text = "클래스 분포 데이터가 제공되지 않아 분포 분석을 생략한다."
    else:
        text = " ".join(([intro] if intro else []) + lines)
    return {"interpretation.distribution_analysis": text}


def _section_8_1(inp: _Input) -> dict:
    """8.1 벤치마크 — 참조 범위 대비 위치와 목표값 수준, 기준의 출처."""
    if not inp.refs:
        return {"conclusion.benchmark": "비교 가능한 내부 참조 기준치가 없어 벤치마크 비교를 생략한다."}
    lines = inp.lines("8.1")
    lines.insert(1 if lines else 0, f"비교 기준은 {inp.refs[0]['source_note']}이다.")
    return {"conclusion.benchmark": " ".join(lines)}


def _section_8_2(inp: _Input) -> dict:
    """8.2 종합 총평 — 판정 사유(D19)·신뢰구간·지표 관계·선언한 평균 방식·모델 용도(사실 문장만)."""
    return {"conclusion.narrative": " ".join(inp.lines("8.2"))}


def _risks_lead(fs: FactSheet) -> str:
    """8.3 첫 문장 — 목표 미달·판정 불가에 따른 위험(결론 목록 밖의 사실 문장).

    판정 불가에 '모든 시험항목이 기준을 충족'이라 쓰면 확인하지 못한 것을 확인한 것처럼 읽히고 판정 모순
    검사에도 걸린다(점검 BE-03) — 판정 불가는 미달이 없어도 '위험을 확인하지 못했다'로 쓴다.
    '임계값이 설정된'은 '목표값이 설정된'으로 쓴다 — 같은 8.3 에 결정 임계값 문장이 함께 나온다.
    """
    failed = failed_metrics(fs)
    judged = [m for m in fs.metrics if m.threshold is not None]
    # 아슬아슬 합격(D12) — near_target 판정은 프론트가 했고 여기서는 합격한 지표만 골라 읽는다.
    near = [m for m in fs.metrics if m.near_target and m.status == "pass"]
    if failed:
        risks = (
            "목표에 미달한 지표가 있어 해당 영역의 성능 저하 위험이 존재한다"
            f"(목표 미달 지표: {', '.join(metric_vs_target(m) for m in failed)})."
        )
        if fs.verdict == "INCONCLUSIVE":
            risks += " 다만 종합 판정이 '판정 불가'여서 판정에 따른 위험은 확인하지 못했다."
        return risks
    if fs.verdict == "INCONCLUSIVE":
        return "종합 판정이 '판정 불가'여서 목표 충족 여부에 따른 위험을 확인하지 못했다."
    if fs.verdict == "CONDITIONAL_PASS" and judged:
        # 미달 없는 조건부 합격(옛 규칙 — 핵심 지표 미산출). '모두 충족하여 위험은 없다'는 판정과 모순이다.
        return "목표값이 설정된 지표 중 목표에 미달한 지표는 없으나, 핵심 지표가 산출되지 않아 그에 따른 위험은 확인하지 못했다."
    if not judged:
        # 목표값이 하나도 없으면 '모든 시험항목이 목표를 충족'은 확인하지 않은 것을 확인한 것처럼 읽힌다.
        return "목표값이 설정된 지표가 없어 목표 미달 위험을 판단하지 못했다."
    if near:
        # "식별된 기준 미달 위험은 없다"와 근소 차이 문장이 함께 실리면 서로 어긋나므로 앞 문장을 바꾼다.
        return "목표값이 설정된 모든 시험항목이 목표를 충족하였다."
    return "목표값이 설정된 모든 시험항목이 목표를 충족하여 식별된 목표 미달 위험은 없다."


def _section_8_3(inp: _Input) -> dict:
    """8.3 리스크 — 목표 미달 위험 + 최약 클래스(사실 문장만)·근소 차이(17·18 규칙)·임계값 부적합·지연시간·대화형 응답 시간."""
    return {"conclusion.risks": " ".join([_risks_lead(inp.fs)] + inp.lines("8.3"))}


def _section_9(inp: _Input) -> dict:
    rec_narrative, recs = _recommendations(inp.fs)
    return {
        "recommendation_narrative.data_quality": rec_narrative.data_quality,
        "recommendation_narrative.model_ops": rec_narrative.model_ops,
        "recommendations": recs,
    }


SECTION_TEMPLATES = {
    "7.1": _section_7_1,
    "7.2": _section_7_2,
    "7.3": lambda inp: {"interpretation.additional_observations": ""},
    "8.1": _section_8_1,
    "8.2": _section_8_2,
    "8.3": _section_8_3,
    "9": _section_9,
}


def _input(fs, benchmark_refs, derived, *, task_type, client_inputs, conclusions) -> _Input:
    if conclusions is None:
        conclusions = select_items(
            fs, client_inputs, task_type or infer_task_type(fs, derived), derived=derived, benchmark_refs=benchmark_refs,
        )
    return _Input(fs, benchmark_refs, derived, conclusions, class_descriptions(client_inputs))


def section_template(
    section_id: str,
    fs: FactSheet,
    benchmark_refs: list[dict],
    derived: dict,
    *,
    task_type: str | None = None,
    client_inputs: dict | None = None,
    conclusions: list[Conclusion] | None = None,
) -> dict:
    """칸 하나의 규칙 문장 — {응답 필드(점 표기): 값}.

    conclusions 를 주면(작업 경로 — 지시문과 같은 목록) 그것을 읽고, 없으면 select_items 로 만든다. task_type 이 없는 옛
    호출은 사실 시트로 짐작한다(required_items.infer_task_type). client_inputs 의 클래스 설명으로 클래스를 부른다(D37)."""
    inp = _input(fs, benchmark_refs, derived, task_type=task_type, client_inputs=client_inputs, conclusions=conclusions)
    return SECTION_TEMPLATES[section_id](inp)


def assemble_response(parts: dict, fs: FactSheet, meta: NarrativeMeta) -> NarrativeResponse:
    """칸별 값(응답 필드 → 값)을 응답 하나로 묶는다. 판정은 늘 사실 시트 값이다(AI echo 를 쓰지 않는다)."""
    return NarrativeResponse(
        interpretation=InterpretationOut(
            confusion_analysis=parts.get("interpretation.confusion_analysis", ""),
            distribution_analysis=parts.get("interpretation.distribution_analysis", ""),
            additional_observations=parts.get("interpretation.additional_observations", ""),
        ),
        conclusion=ConclusionOut(
            verdict=fs.verdict,
            benchmark=parts.get("conclusion.benchmark", ""),
            narrative=parts.get("conclusion.narrative", ""),
            risks=parts.get("conclusion.risks", ""),
        ),
        recommendation_narrative=RecommendationNarrativeOut(
            data_quality=parts.get("recommendation_narrative.data_quality", ""),
            model_ops=parts.get("recommendation_narrative.model_ops", ""),
        ),
        recommendations=parts.get("recommendations", []),
        meta=meta,
    )


def build_fallback_narrative(
    fs: FactSheet,
    benchmark_refs: list[dict],
    derived: dict,
    reason: str,
    *,
    task_type: str | None = None,
    client_inputs: dict | None = None,
) -> NarrativeResponse:
    """규칙 기반 폴백 서술 — 일곱 칸의 템플릿을 모은다. reason: no_key | api_error | grounding_failed …

    옛 경로(legacy)는 요청의 task_type·client_inputs 를 넘긴다. 둘 없이 부르면(옛 호출) 작업 유형을 짐작하고 클래스 설명 없이 쓴다."""
    inp = _input(fs, benchmark_refs, derived, task_type=task_type, client_inputs=client_inputs, conclusions=None)
    parts: dict = {}
    for section_id, fn in SECTION_TEMPLATES.items():
        parts.update(fn(inp))
    return assemble_response(
        parts, fs,
        NarrativeMeta(source="fallback", model=None, reason=reason, grounding=GroundingInfo(passed=True)),
    )
