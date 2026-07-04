"""
narrative_prompt.py — LLM 서술 생성 프롬프트 + structured output 스키마.

analyzer.py 의 프롬프트/스키마 패턴을 따른다. 핵심 제약:
  - fact_sheet JSON 안의 숫자만 사용. 새 숫자 계산·반올림·백분율 환산·추정 금지.
  - latency 가 available=false 면 지연 수치를 언급하지 않는다.
  - conclusion.verdict 는 주어진 값을 그대로 echo (서버가 최종 강제).
  - benchmark 서술에는 source_note(기준 출처/성격)를 포함한다.
"""
import json


def build_system_prompt(report_purpose: str) -> str:
    return (
        "너는 ISO/IEC TS 4213:2022 기반 AI 분류 모델 성능 시험 성적서의 서술 작성 전문가다.\n"
        "주어진 JSON 사실 시트(fact_sheet)와 파생값(derived)·벤치마크(benchmark_refs)만을 근거로 "
        "한국어 성적서 서술을 작성한다.\n\n"
        "[필수 규칙]\n"
        "1. JSON 안에 명시된 숫자만 사용한다. 새로운 숫자를 계산·반올림·백분율 환산·추정하지 마라.\n"
        "2. JSON 에 없는 지표·수치(특히 지연시간/latency 가 available=false 인 경우 어떤 지연 수치도) 언급 금지.\n"
        "3. conclusion.verdict 는 fact_sheet.verdict 값을 그대로 사용한다.\n"
        "4. benchmark 서술에는 benchmark_refs 의 source_note(기준의 출처·성격)를 반드시 포함하고, "
        "'공개 벤치마크 평균'처럼 단정하지 마라. benchmark_refs 가 비어 있으면 비교를 주장하지 마라.\n"
        "5. 과장 없이 사실 기반으로, 평가기관 성적서 문체(객관적·간결)로 작성한다.\n"
        f"6. 이 성적서의 용도는 '{report_purpose}' 이다.\n"
        "7. 반드시 지정된 JSON 스키마로만 응답한다. 설명 문장을 덧붙이지 마라."
    )


def build_user_prompt(fact_sheet: dict, benchmark_refs: list, derived: dict) -> str:
    payload = {
        "fact_sheet": fact_sheet,
        "derived": derived,
        "benchmark_refs": benchmark_refs,
    }
    return (
        "다음 사실 시트를 근거로 성적서 7·8·9절 서술을 작성하라.\n"
        "각 필드 의미: interpretation(7절 정밀분석: confusion_analysis=혼동행렬 기반 오분류 해석, "
        "distribution_analysis=클래스 분포/불균형 해석), "
        "conclusion(8절: benchmark=벤치마크 비교, narrative=종합 총평, risks=리스크), "
        "recommendation_narrative(9절 서술: data_quality, model_ops), "
        "recommendations(9절 권고표 항목들).\n"
        "benchmark_refs 의 각 항목은 direction(higher/lower=높을수록/낮을수록 좋음)과 "
        "quality(better/within/worse=우수/기준 범위 내/미흡)를 포함한다. 벤치마크 우열은 "
        "position(단순 수치 위치)이 아니라 quality 로 서술하라(낮을수록 좋은 지표는 범위 아래가 오히려 우수).\n\n"
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
