"""
narrative_prompt.py — LLM 서술 생성 프롬프트 + structured output 스키마.

analyzer.py 의 프롬프트/스키마 패턴을 따른다. 핵심 제약:
  - fact_sheet JSON 안의 숫자만 사용. 새 숫자 계산·반올림·백분율 환산·추정 금지.
  - latency 가 available=false 면 지연 수치를 언급하지 않는다.
  - conclusion.verdict 는 주어진 값을 그대로 echo (서버가 최종 강제).
  - benchmark 서술에는 source_note(기준 출처/성격)를 포함한다.
"""
import json

from app.narrative.derived import threshold_tuning_needed


# 분류 유형의 성적서 이름(재료 목록 2절 "작업 유형 — 전 절, AI 예"). 한국어 이름은 이 셋으로 통일한다(W9,
# 2026-10-18 사용자 결정) — 프론트 성적서의 TASK_TYPE_LABELS 와 같은 글자다. 이미 발급된 성적서는 저장된
# 이름 그대로 두고, 새 서술 요청부터 이 이름이 AI 에 간다.
TASK_TYPE_LABELS = {"binary": "이진 분류", "multiclass": "다중 클래스 분류", "multilabel": "다중 레이블 분류"}


# 규칙 10 — 예측을 확률에서 파생했을 때 방법마다 다른 한 문장(고정 문구 9 와 같은 구분).
# 다중 클래스는 임계값이 아니라 가장 큰 확률의 클래스를 고른다(argmax).
_DERIVATION_RULES = {
    "threshold": (
        "fact_sheet.derived_prediction 이 있으므로 예측값은 모델이 직접 낸 것이 아니라 확률에 "
        "결정 임계값을 적용해 만든 것이다. 수치가 그 임계값에 따라 달라진다는 점을 해석에 반영하라."
    ),
    "threshold_per_label": (
        "fact_sheet.derived_prediction 이 있으므로 예측값은 모델이 직접 낸 것이 아니라 레이블마다 "
        "점수에 결정 임계값을 적용해 만든 것이다. 수치가 그 임계값들에 따라 달라진다는 점을 해석에 반영하라."
    ),
    "argmax": (
        "fact_sheet.derived_prediction 이 있으므로 예측값은 모델이 직접 낸 것이 아니라 클래스별 확률 중 "
        "가장 큰 클래스를 고른 것이다. 결정 임계값은 쓰지 않았으니 임계값을 언급하지 마라."
    ),
}
_DERIVATION_RULE_OTHER = (
    "fact_sheet.derived_prediction 이 있으므로 예측값은 모델이 직접 낸 것이 아니라 확률에서 파생한 것이다."
)


def _derivation_rule(fs) -> str | None:
    derived = getattr(fs, "derived_prediction", None) if fs is not None else None
    if derived is None:
        return None
    return _DERIVATION_RULES.get(derived.method, _DERIVATION_RULE_OTHER)


def _error_ratio_rule(fs) -> str | None:
    """규칙 11 — 오분류의 성격(서술 설계 D30). error_ratios 가 있을 때만, 사실에 맞는 문장만 붙인다."""
    er = getattr(fs, "error_ratios", None) if fs is not None else None
    if er is None:
        return None
    rule = (
        "fact_sheet.error_ratios 가 있으므로 interpretation.confusion_analysis(7.1)에서 오류의 성격을 "
        "서술하라. 오분류(misclassified) 중 결정 경계 근처에서 틀린 경계 오류(boundary: 경계와의 거리가 "
        "boundary_margin 미만)와 높은 확신으로 틀린 확신 오류(confident: 틀린 쪽 확률이 confident_level "
        "이상)의 count 와 ratio 를 적힌 그대로 써라."
    )
    if er.boundary is None and er.boundary_note == "threshold_unknown":
        rule += (
            " boundary 가 null 이고 boundary_note 가 threshold_unknown 이다 — 예측값을 직접 제출해 결정 "
            "임계값을 알 수 없으므로 경계 오류는 계산하지 않았다고 적고, 경계 오류 수치를 지어내지 마라."
        )
    if er.unit == "label_cell":
        rule += " unit 이 label_cell 이므로 단위는 표본이 아니라 '틀린 레이블 칸'이다. 건수를 표본 수로 쓰지 마라."
    return rule


def _confidence_interval_rule(fs) -> str | None:
    """규칙 12 — 신뢰구간 해석(서술 설계 D29). 지표에 ci_low·ci_high 가 실렸을 때만 붙인다."""
    metrics = getattr(fs, "metrics", None) if fs is not None else None
    if not metrics or not any(m.ci_low is not None and m.ci_high is not None for m in metrics):
        return None
    return (
        "fact_sheet.metrics 의 ci_low·ci_high 는 지표마다의 95% 신뢰구간이다. "
        "conclusion.narrative(8.2)에서 derived.ci_contains_target 에 이름이 있는 지표는 구간이 목표값을 "
        "포함하므로 표본 변동에 따라 그 지표의 판정이 달라질 수 있다고 적어라. 목록에 없는 지표에 이 말을 "
        "붙이지 말고, 구간 폭·차이 같은 새 숫자를 계산하지 마라. 단일 모델 평가라 통계적 유의성 검정은 "
        "하지 않았다 — 유의하다·유의미한 차이 같은 검정 결과를 주장하지 마라."
    )


def _near_target_rule(fs) -> str | None:
    """규칙 13 — 아슬아슬 합격(서술 설계 D12). near_target 이고 합격한 지표가 있을 때만 붙인다."""
    metrics = getattr(fs, "metrics", None) if fs is not None else None
    if not metrics or not any(m.near_target and m.status == "pass" for m in metrics):
        return None
    return (
        "fact_sheet.metrics 에서 near_target=true 이고 status=pass 인 지표는 목표값과의 차이(margin)가 "
        "목표값의 3% 미만인 아슬아슬 합격이다. conclusion.risks(8.3)에 그 지표와 margin 을 적힌 그대로 적고, "
        "표본이 달라지면 기준에 미달할 위험이 있다고 써라. 합격한 지표이므로 미달했다고 쓰지 마라."
    )


def _threshold_gain_rule(fs) -> str | None:
    """규칙 14 — 임계값 조정 여지(서술 설계 D40). gain 이 기준(0.10) 이상일 때만 붙인다."""
    if fs is None or getattr(fs, "f1_threshold_gain", None) is None or not threshold_tuning_needed(fs):
        return None
    return (
        "fact_sheet.f1_threshold_gain 의 gain 이 0.1 이상이다. conclusion.narrative(8.2)와 "
        "conclusion.risks(8.3)에서 결정 임계값을 best_threshold 로 바꾸면 F1 이 current_f1 에서 max_f1 까지 "
        "오를 수 있다고 적되, gain 은 '이 데이터에서 오를 수 있는 최대폭'이고 같은 평가 데이터로 고른 "
        "임계값이라 별도 데이터에서 확인해야 한다고 반드시 덧붙여라. 성능 향상을 보장한다고 쓰지 마라."
    )


# 규칙 15 — 불균형 영향(서술 설계 D20). level 판정은 평가 엔진이 했고, 서술은 level 에 맞는 결론만 쓴다.
_IMBALANCE_RULES = {
    "affected": (
        "불균형 영향 있음이다 — 불균형 비율(imbalance_ratio)이 1.5 이상이고 정확도와 매크로 F1 의 차이"
        "(accuracy_minus_macro_f1) 또는 다수·소수 클래스 F1 의 차이(majority_minus_minority_f1)가 0.1 이상이다. "
        "불균형이 성능 수치에 영향을 주었다고 쓰고, 소수 클래스의 성능 해석에 유의하라고 적어라."
    ),
    "no_distortion": (
        "불균형은 있으나 성능 왜곡은 관찰되지 않았다 — 불균형 비율(imbalance_ratio)이 1.5 이상이지만 두 F1 "
        "차이가 모두 0.1 미만이다. 불균형이 성능을 왜곡했다고 쓰지 마라."
    ),
    "mild": (
        "불균형이 경미하다 — 불균형 비율(imbalance_ratio)이 1.5 미만이다. 불균형 때문에 성능이 왜곡되었다고 "
        "쓰지 마라."
    ),
}


def _imbalance_rule(fs) -> str | None:
    ie = getattr(fs, "imbalance_effect", None) if fs is not None else None
    if ie is None:
        return None
    return (
        f"fact_sheet.imbalance_effect 의 level 이 {ie.level} 이므로 interpretation.distribution_analysis(7.2)의 "
        f"결론은 {_IMBALANCE_RULES[ie.level]} 수치는 imbalance_effect 에 적힌 그대로 써라."
    )


def build_system_prompt(report_purpose: str, task_type: str = "", fs=None) -> str:
    """`fs`(FactSheet)는 사실에 따라 달라지는 규칙(규칙 10~15)을 고르는 데만 쓴다."""
    task_label = TASK_TYPE_LABELS.get(task_type, task_type)
    derivation_rule = _derivation_rule(fs)
    error_ratio_rule = _error_ratio_rule(fs)
    ci_rule = _confidence_interval_rule(fs)
    near_rule = _near_target_rule(fs)
    gain_rule = _threshold_gain_rule(fs)
    imbalance_rule = _imbalance_rule(fs)
    return (
        "너는 ISO/IEC TS 4213:2022 기반 AI 분류 모델 성능 시험 성적서의 서술 작성 전문가다.\n"
        "주어진 JSON 사실 시트(fact_sheet)와 파생값(derived)·벤치마크(benchmark_refs)만을 근거로 "
        "한국어 성적서 서술을 작성한다.\n\n"
        "[필수 규칙]\n"
        "1. JSON 안에 명시된 숫자만 사용한다. 새로운 숫자를 계산·반올림·백분율 환산·추정하지 마라.\n"
        "2. JSON 에 없는 지표·수치(특히 지연시간/latency 가 available=false 인 경우 어떤 지연 수치도) 언급 금지.\n"
        "3. conclusion.verdict 는 fact_sheet.verdict 값을 그대로 사용한다. 값은 PASS(합격), "
        "CONDITIONAL_PASS(조건부 합격), FAIL(불합격), INCONCLUSIVE(판정 불가: 핵심 지표가 산출되지 "
        "않았거나 판정 대상 지표가 없음) 중 하나다. 판정 불가면 합격이나 불합격으로 단정하지 마라. "
        "metrics 의 is_core=true 는 판정에 쓰는 핵심 지표다(의뢰자가 정했거나 분류 유형의 기본값).\n"
        "4. benchmark 서술에는 benchmark_refs 의 source_note(기준의 출처·성격)를 반드시 포함하고, "
        "'공개 벤치마크 평균'처럼 단정하지 마라. benchmark_refs 가 비어 있으면 비교를 주장하지 마라.\n"
        "5. 과장 없이 사실 기반으로, 평가기관 성적서 문체(객관적·간결)로 작성한다.\n"
        f"6. 이 성적서의 용도는 '{report_purpose}' 이다.\n"
        "7. 반드시 지정된 JSON 스키마로만 응답한다. 설명 문장을 덧붙이지 마라.\n"
        "8. client_inputs 는 의뢰자가 제공한 정보다(선언, 학습·평가 데이터, 정답 라벨, 실행 환경 등). "
        "서술에 자유롭게 반영하되, 값이 '제공되지 않음'인 항목은 그 사실만 적고 내용을 지어내지 마라. "
        "client_inputs 의 숫자는 적힌 그대로만 쓴다.\n"
        + (f"9. 평가한 모델의 분류 유형은 '{task_label}' 이다.\n" if task_label else "")
        + (f"10. {derivation_rule}\n" if derivation_rule else "")
        + (f"11. {error_ratio_rule}\n" if error_ratio_rule else "")
        + (f"12. {ci_rule}\n" if ci_rule else "")
        + (f"13. {near_rule}\n" if near_rule else "")
        + (f"14. {gain_rule}\n" if gain_rule else "")
        + (f"15. {imbalance_rule}\n" if imbalance_rule else "")
    )


def _problems_text(problems: list) -> str:
    """9절 권고의 근거(서술 설계 D22). 권고는 문제 목록에만 묶는다 — 문제가 없으면 권고도 없다."""
    if not problems:
        return (
            "fact_sheet.problems 가 비어 있다 — 이 평가에서 지목된 데이터·모델상의 문제가 없다. "
            "recommendations 는 빈 배열로 두고 권고를 지어내지 마라. recommendation_narrative 에도 "
            "문제나 권고를 새로 만들지 말고, 지목된 문제가 없다는 사실만 적어라.\n"
        )
    return (
        "fact_sheet.problems 는 9절 권고가 대응할 문제 목록이다(id 순서가 우선순위 순서). "
        "recommendations 의 각 항목은 반드시 problems 의 id 하나를 problem_id 로 가지며 그 문제에 대응해야 "
        "한다. 모든 문제에 권고를 하나 이상 달고, 목록에 없는 문제에 대한 권고는 쓰지 마라. "
        "priority 는 그 문제의 priority 를 쓴다. expected_impact(기대 효과)에는 숫자나 예상 수치(개선 폭·"
        "목표 달성값 등)를 쓰지 말고 무엇이 나아지는지만 말로 적어라. "
        "recommendation_narrative.data_quality(9.1)는 recommend_section 이 9.1 인 데이터 쪽 문제를, "
        "model_ops(9.2)는 9.2 인 모델 쪽 문제를 다룬다. 어느 쪽에 문제가 없으면 지목된 문제가 없다고만 "
        "적고, problems 에 없는 문제를 지어내지 마라. note 가 있으면 그 사실을 권고에 반영하라.\n"
    )


def build_user_prompt(
    fact_sheet: dict,
    benchmark_refs: list,
    derived: dict,
    client_inputs: dict | None = None,
) -> str:
    payload = {
        "fact_sheet": fact_sheet,
        "derived": derived,
        "benchmark_refs": benchmark_refs,
    }
    # 고객 입력은 계산값과 섞이지 않게 따로 싣는다(서술 설계 D25).
    if client_inputs:
        payload["client_inputs"] = client_inputs
    return (
        "다음 사실 시트를 근거로 성적서 7·8·9절 서술을 작성하라.\n"
        "각 필드 의미: interpretation(7절 정밀분석: confusion_analysis=혼동행렬 기반 오분류 해석, "
        "distribution_analysis=클래스 분포/불균형 해석), "
        "conclusion(8절: benchmark=벤치마크 비교, narrative=종합 총평, risks=리스크), "
        "recommendation_narrative(9절 서술: data_quality, model_ops), "
        "recommendations(9절 권고표 항목들).\n"
        "benchmark_refs 의 각 항목은 direction(higher/lower=높을수록/낮을수록 좋음)과 "
        "quality(better/within/worse=우수/기준 범위 내/미흡)를 포함한다. 벤치마크 우열은 "
        "position(단순 수치 위치)이 아니라 quality 로 서술하라(낮을수록 좋은 지표는 범위 아래가 오히려 우수).\n"
        "target_value 는 그 지표의 목표값(없으면 null)이고, target_position 은 목표값이 참조 범위보다 "
        "낮음(below)/범위 안(within)/높음(above)인지를 나타내는 단순 수치 위치다. 목표 설정 수준을 "
        "8.1(conclusion.benchmark)에서 해석하되, 높을수록/낮을수록 좋은지 방향 해석은 하지 말고 위치만 말하라.\n"
        + _problems_text(fact_sheet.get("problems") or [])
        + "\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def build_response_schema(problems: list | None = None) -> dict:
    """서술 응답의 JSON 스키마 본문(Claude 구조화 출력 `output_config.format.schema`). meta 는 서버가 채우므로 제외.

    권고는 문제 목록(list[ProblemFact])에 묶는다(서술 설계 D22): problem_id 는 그 요청의 문제 번호들만 고를 수 있는 enum 이다.
    문제가 없으면 recommendations 속성 자체를 뺀다 — 구조화 출력은 maxItems 를 받지 않아 "빈 배열만"을 스키마로
    말할 수 없다. 서버 후처리(service._align_recommendations)가 같은 규칙을 다시 지킨다 — 스키마는 첫 방어선이다.
    maxItems·minimum·maximum·minLength·maxLength 를 쓰지 않고 minItems 는 0·1 만 쓴다(구조화 출력 제약).
    추론 과정·근거 칸을 두지 않는다(reasoning_extraction 거절 — app.narrative.llm 머리말).
    """
    ids = [p.id for p in (problems or [])]
    properties: dict = {
        "interpretation": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "confusion_analysis": {"type": "string"},
                "distribution_analysis": {"type": "string"},
            },
            "required": ["confusion_analysis", "distribution_analysis"],
        },
        "conclusion": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "verdict": {"type": "string"},
                "benchmark": {"type": "string"},
                "narrative": {"type": "string"},
                "risks": {"type": "string"},
            },
            "required": ["verdict", "benchmark", "narrative", "risks"],
        },
        "recommendation_narrative": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "data_quality": {"type": "string"},
                "model_ops": {"type": "string"},
            },
            "required": ["data_quality", "model_ops"],
        },
    }
    required = ["interpretation", "conclusion", "recommendation_narrative"]
    if ids:
        properties["recommendations"] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "problem_id": {"type": "string", "enum": ids},
                    "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "category": {"type": "string"},
                    "action": {"type": "string"},
                    "expected_impact": {"type": "string"},
                },
                "required": ["problem_id", "priority", "category", "action", "expected_impact"],
            },
        }
        required.append("recommendations")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }
