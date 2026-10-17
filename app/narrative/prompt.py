"""
narrative_prompt.py — LLM 서술 생성 프롬프트 + structured output 스키마.

analyzer.py 의 프롬프트/스키마 패턴을 따른다. 핵심 제약:
  - fact_sheet JSON 안의 숫자만 사용. 새 숫자 계산·반올림·백분율 환산·추정 금지.
  - latency 가 available=false 면 지연 수치를 언급하지 않는다.
  - conclusion.verdict 는 주어진 값을 그대로 echo (서버가 최종 강제).
  - benchmark 서술에는 source_note(기준 출처/성격)를 포함한다.
"""
import json


# 분류 유형의 성적서 이름(재료 목록 2절 "작업 유형 — 전 절, AI 예").
TASK_TYPE_LABELS = {"binary": "이진 분류", "multiclass": "다중 클래스 분류", "multilabel": "멀티레이블 분류"}


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


def build_system_prompt(report_purpose: str, task_type: str = "", fs=None) -> str:
    """`fs`(FactSheet)는 사실에 따라 달라지는 규칙(규칙 10·11 등)을 고르는 데만 쓴다."""
    task_label = TASK_TYPE_LABELS.get(task_type, task_type)
    derivation_rule = _derivation_rule(fs)
    error_ratio_rule = _error_ratio_rule(fs)
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
        "metrics 의 is_core=true 는 의뢰자가 지정한 핵심 지표다.\n"
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
        "8.1(conclusion.benchmark)에서 해석하되, 높을수록/낮을수록 좋은지 방향 해석은 하지 말고 위치만 말하라.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def build_response_schema() -> dict:
    """OpenAI structured output (strict json_schema). meta 는 서버가 채우므로 제외."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "narrative_result",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
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
                    "recommendations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                "category": {"type": "string"},
                                "action": {"type": "string"},
                                "expected_impact": {"type": "string"},
                            },
                            "required": ["priority", "category", "action", "expected_impact"],
                        },
                    },
                },
                "required": [
                    "interpretation",
                    "conclusion",
                    "recommendation_narrative",
                    "recommendations",
                ],
            },
        },
    }
