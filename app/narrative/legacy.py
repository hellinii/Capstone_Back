"""app/narrative/legacy.py — 옛 동기 경로(`POST /api/generate-narrative`)의 서술 생성(S1 판, 서술 라운드 S5 에서 얼림).

옛 프론트(main 9-22)가 쓰던 경로다. 운영 프론트는 2026-10-23 배포부터 작업 경로를 쓰고, 이 경로는 옛 클라이언트를 위해
얼려 둔다. 서술 라운드 S5 가 작업 번호 경로의 생성기를 장별 호출·칸별 검문 사다리로
바꾸면서(app.narrative.generator), 이 경로는 **S1 그대로** 둔다 — 한 번 호출·빠른 effort(llm 기본)·서버 마감 120초·
옛 지시문(아래 build_system_prompt·build_user_prompt·build_response_schema). 같은 서술 예산에서 보고서 1건으로 센다.
다음 라운드에 지운다(계획 설계 요지 7).

공유하는 것: 호출 한 곳(app.narrative.llm), 숫자 검문·판정 모순 검사(app.narrative.grounding — 검사가 나아지면 이 경로도
같이 나아진다), 규칙 문장(app.narrative.fallback), 파생값(derived)·기준치(baselines). 지시문과 생성 흐름은 얼린다.

fact_sheet 로부터 파생값·기준치·숫자 화이트리스트를 준비하고 LLM 을 호출한 뒤, grounding
검증(환각 방어)을 거쳐 응답을 조립한다. 키 없음/호출 실패/검증 실패/조립 오류 시 규칙 폴백으로
대체한다.

상호작용
- 의존(import): app.narrative.llm(Claude 호출 한 곳), app.narrative.baselines/fallback/derived/grounding,
  app.narrative.prompt(TASK_TYPE_LABELS 만)
- 사용처: app.narrative.router(옛 경로 generate_narrative_endpoint), tests
"""
import asyncio
import json
import contextlib
import copy
import logging
import time
from dataclasses import asdict

from app.narrative.schemas import BenchmarkRefOut, ConclusionOut, InterpretationOut, NarrativeMeta, NarrativeRequest, NarrativeResponse, RecommendationNarrativeOut, RecommendationOut
from app.narrative.baselines import build_benchmark_refs
from app.narrative.fallback import build_fallback_narrative, problem_recommendation
from app.narrative.prompt import TASK_TYPE_LABELS
from app.narrative.derived import compute_derived
from app.narrative.llm import call_narrative_llm
from app.narrative.grounding import (
    build_number_whitelist,
    find_verdict_contradictions,
    has_number_token,
    verify_grounding,
    _collect_grounding_texts,
)
from app.core import llm_budget
from app.core.concurrency import narrative_slot, run_cpu_bound

logger = logging.getLogger(__name__)

# 옛 경로는 늘 판 3 이다(판 4 라운드 S2 ②) — 요청에 report_edition 이 4 로 와도 읽지 않는다. 안쪽 함수(기준표·허용 목록·
# 규칙 문장)는 기본값 없는 edition 을 받으므로 이 입구에서 판 3 을 넘긴다. 기본값 3 을 두는 곳은 요청 경계와 여기뿐이다.
EDITION = 3

# 판 4 요청에만 있는 칸(계약 fact_sheet_contract_ed4.json 의 edition4_only_keys 중 사실 시트 안). 옛 지시문은 사실 시트를
# 통째로 덤프하므로 이 칸들을 빼야 판 3 지시문이 한 글자도 바뀌지 않는다(값이 None 이어도 키가 늘면 글이 바뀐다).
EDITION4_ONLY_SHEET_FIELDS: dict = {
    "metrics": {"__all__": {"ci_low_text", "ci_high_text"}},
    "f1_threshold_gain": {
        "best_threshold_text", "max_f1_text", "current_f1_text", "best_recall_text", "best_precision_text",
        "current_recall_text", "current_precision_text", "risk_side",
    },
    "facts": {"distribution_shift": {"__all__": {"evaluation_minus_training_text"}}, "weakest_class": {"f1_text"}},
}


# ── 옛 지시문(S1 판 그대로 — 얼림) ─────────────────────────────────────────────────

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


def _threshold_gain_rule(fs, derived: dict | None) -> str | None:
    """규칙 14 — 임계값 조정 여지(서술 설계 D40). 엔진이 부적합으로 본 때(derived.threshold_tuning)만 붙인다."""
    if fs is None or getattr(fs, "f1_threshold_gain", None) is None or not (derived or {}).get("threshold_tuning"):
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


def build_system_prompt(report_purpose: str, task_type: str = "", fs=None, derived: dict | None = None) -> str:
    """`fs`(FactSheet)·`derived`(compute_derived)는 사실에 따라 달라지는 규칙(규칙 10~15)을 고르는 데만 쓴다."""
    task_label = TASK_TYPE_LABELS.get(task_type, task_type)
    derivation_rule = _derivation_rule(fs)
    error_ratio_rule = _error_ratio_rule(fs)
    ci_rule = _confidence_interval_rule(fs)
    near_rule = _near_target_rule(fs)
    gain_rule = _threshold_gain_rule(fs, derived)
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


# ── 옛 생성 흐름 ────────────────────────────────────────────────────────────────

# 옛 동기 경로(/api/generate-narrative)의 서버 마감. 슬롯 대기와 호출을 합쳐 이 시간을 넘으면 200 + 규칙 문장
# (사유: 슬롯을 얻지 못했으면 busy, 호출 중이었으면 timeout). 프론트는 160초에 요청을 끊으므로 그보다 짧아야
# 빈 서술 대신 규칙 문장이 간다. llm.CALL_TIMEOUT_SECONDS × (llm.MAX_RETRIES + 1) 이 이 값 안에 든다(테스트 고정).
NARRATIVE_DEADLINE_SECONDS = 120.0

# 서술 AI 에 보내는 user 프롬프트의 글자 수 상한. 무인증 공개 경로라 칸별 상한만으로는 본문 2MB 안에서
# 약 96만 자짜리 요청을 만들 수 있고, Sonnet 5.5 는 그것을 받아 한 건에 약 0.7달러를 받는다(독립 검토 OPS-1).
# 정상 요청은 클래스가 많아도 수만 자 안이다. 넘으면 AI 를 부르지 않고 지갑도 쓰지 않은 채 규칙 문장
# (사유 input_too_large). 사실 시트 칸별 상한·422 는 S4b 에서 따로 정한다.
MAX_NARRATIVE_PROMPT_CHARS = 80_000


# 지시문이 상한을 넘을 때 지시문에서만 덜어 내는 차례(S4b). 혼동행렬·클래스별 표는 M21·M22 선택과 무관하게 늘
# 실리므로(S4a) 클래스가 많으면 지시문이 커진다 — 서술 결론(최약 클래스·최다 오분류 쌍 등)은 facts 에 이미 있어
# 큰 표부터 뺀다. 숫자 허용 목록은 원본 사실 시트로 만든다(뺀 표의 숫자를 인용해도 사실이다).
# 덜어 낸 수는 프론트 미리 자르기와 같은 자리(facts.omitted)에 적는다 — 빈 목록을 '해당 없음'으로 읽지 않게
# (S4b 검토 SV-01). facts 가 없는 옛 요청은 시트에 trimmed_for_prompt 로 적는다.
def _note(d: dict, key: str, n: int) -> None:
    if n <= 0:
        return
    facts = d.get("facts")
    if isinstance(facts, dict):
        omitted = facts.setdefault("omitted", {})
        omitted[key] = omitted.get(key, 0) + n
    else:
        d.setdefault("trimmed_for_prompt", {})[key] = n


def _drop_matrix(d: dict, derived: dict) -> None:
    if d.get("confusion"):
        _note(d, "confusion", len(d["confusion"].get("labels") or []))
        d["confusion"] = {**d["confusion"], "matrix": []}


def _drop_key(key: str, empty):
    def drop(d: dict, derived: dict) -> None:
        value = d.get(key)
        if key == "distribution":
            _note(d, "class_distribution", len((value or {}).get("class_distribution") or {}))
            # 서버 파생 분포 백분율도 클래스 수만큼이라 함께 뺀다(총건수는 남긴다).
            if isinstance(derived.get("distribution"), dict):
                derived["distribution"] = {k: v for k, v in derived["distribution"].items() if k != "percentages"}
        else:
            _note(d, key, len(value or []))
        d[key] = empty
    return drop


def _drop_fact_lists(d: dict, derived: dict) -> None:
    facts = d.get("facts")
    if isinstance(facts, dict):
        bias = facts.get("prediction_bias")
        n = len(facts.get("low_support_classes") or []) + len(facts.get("distribution_shift") or [])
        n += len(bias.get("rows") or []) if isinstance(bias, dict) else 0
        facts["low_support_classes"] = []
        facts["distribution_shift"] = []
        if isinstance(bias, dict):
            bias["rows"] = []
        _note(d, "facts_class_lists", n)


_PROMPT_TRIM_STEPS = (
    ("confusion.matrix", _drop_matrix),
    ("per_class", _drop_key("per_class", [])),
    ("confusion_counts", _drop_key("confusion_counts", [])),
    ("distribution", _drop_key("distribution", None)),
    ("facts.class_lists", _drop_fact_lists),
)


def build_prompts_within_limit(req, fs, derived, benchmark_refs) -> tuple[str, str, list[str]]:
    """system·user 지시문을 만들고, 합이 MAX_NARRATIVE_PROMPT_CHARS 를 넘으면 _PROMPT_TRIM_STEPS 차례로 덜어 낸다.

    돌려주는 세 번째 값은 덜어 낸 자리(내부 기록). 다 덜어도 넘으면 호출부가 input_too_large 로 끝낸다.
    """
    system = build_system_prompt(req.report_purpose.value, req.task_type.value, fs, derived)
    sheet = fs.model_dump(exclude=EDITION4_ONLY_SHEET_FIELDS)  # 판 4 칸은 옛 지시문에 싣지 않는다
    view = copy.deepcopy(derived)  # 지시문용 파생값 사본 — 허용 목록·폴백은 원본을 쓴다
    user = build_user_prompt(sheet, benchmark_refs, view, req.client_inputs)
    dropped: list[str] = []
    for name, step in _PROMPT_TRIM_STEPS:
        if len(system) + len(user) <= MAX_NARRATIVE_PROMPT_CHARS:
            break
        step(sheet, view)
        dropped.append(name)
        user = build_user_prompt(sheet, benchmark_refs, view, req.client_inputs)
    return system, user, dropped


def _prepare_facts(fs, task_type_value: str, client_inputs: dict | None = None):
    """파생값·기준치·숫자 화이트리스트 준비 — 순수 CPU 구간(스레드풀로 오프로드된다).

    build_number_whitelist 는 혼동행렬을 행x열 이중 루프로 돌며 셀마다 문자열을
    여러 개 만들므로 행렬 차원에 대해 제곱으로 커진다(G-04b 실측: 2500x2500 입력이
    이벤트 루프를 15.1 초 막았다).
    """
    derived = compute_derived(fs, task_type=task_type_value)
    benchmark_refs = build_benchmark_refs(task_type_value, fs.metrics, edition=EDITION)
    whitelist = build_number_whitelist(fs, benchmark_refs, derived, client_inputs, edition=EDITION)
    return derived, benchmark_refs, whitelist


def _verify_llm_output(data: dict, whitelist):
    """LLM 출력 전수 수집 + grounding 검증 — 순수 CPU 구간."""
    return verify_grounding(_collect_grounding_texts(data), whitelist, edition=EDITION)


class _ImpactNumber(Exception):
    """권고의 기대 효과에 숫자가 있다 — 전체 폴백(impact_number)으로 강등하라는 신호."""


def _align_recommendations(recs: list, problems: list) -> list[RecommendationOut]:
    """LLM 권고를 문제 목록에 맞춘다(서술 설계 D22). 규칙은 계약서 §3 그대로다.

    - problem_id 가 비었거나 목록에 없는 권고는 버린다(어느 문제에도 대응하지 않는 권고).
    - 남은 권고의 기대 효과에 숫자가 있으면 _ImpactNumber — 근거 없는 예측값이라 전체를 폴백한다.
    - priority 는 문제의 것으로 강제한다(LLM 값 무시). 정렬은 문제 번호 순서, 같은 문제 안은 LLM 순서.
    - 권고가 하나도 없는 문제는 그 종류의 폴백 템플릿으로 채운다 — 9절 표가 문제 목록과 맞아야 한다.
    - 개수로 자르지 않는다(종전 [:5] 는 문제가 많을 때 뒤쪽 문제의 권고를 말없이 지웠다).
    문제가 없으면 권고도 없다. 권고 항목이 dict 가 아니면 예외가 나서 호출부가 assembly_error 로 강등한다.
    """
    by_id: dict[str, list[dict]] = {p.id: [] for p in problems}
    for r in recs:
        pid = str(r.get("problem_id") or "").strip()
        if pid not in by_id:
            continue
        if has_number_token(r.get("expected_impact", "")):
            raise _ImpactNumber(pid)
        by_id[pid].append(r)

    aligned: list[RecommendationOut] = []
    for p in problems:
        matched = by_id[p.id]
        if not matched:
            aligned.append(problem_recommendation(p, edition=EDITION))
            continue
        for r in matched:
            aligned.append(RecommendationOut(**{**r, "problem_id": p.id, "priority": p.priority}))
    return aligned


async def generate_narrative(client, req: NarrativeRequest, **narrate_options) -> NarrativeResponse:
    """
    LLM 서술 생성 진입점.
    - client is None(무키) / LLM 호출 실패 / grounding 위반 → 규칙 폴백.
    - verdict 는 항상 fact_sheet 값으로 강제(LLM echo 무시).
    - 벤치마크 비교값(benchmark_refs)은 서술 경로와 무관하게 응답에 함께 싣는다 — 8.1 에 참조
      범위를 숫자로 찍는다(재료 목록 9절).
    - narrate_options 는 작업 번호 경로(app.narrative.jobs)가 넘기는 _narrate 의 키워드 인자다. 옛 경로는
      넘기지 않아 S1 동작(서버 마감 120초·호출마다 지갑 1건) 그대로다.
    """
    fs = req.fact_sheet
    derived, benchmark_refs, whitelist = await run_cpu_bound(
        _prepare_facts, fs, req.task_type.value, req.client_inputs
    )
    resp = await _narrate(client, req, fs, derived, benchmark_refs, whitelist, **narrate_options)
    resp.benchmark_refs = [BenchmarkRefOut(**ref) for ref in benchmark_refs]
    return resp


async def _no_stage(_stage: str) -> None:
    return None


# _narrate 의 deadline 기본값 표시 — 옛 경로의 서버 마감(NARRATIVE_DEADLINE_SECONDS)을 호출 때 읽는다.
# None 은 "마감 없음"이다(작업 번호 경로: 작업 전체 제한이 바깥에서 덮는다).
_LEGACY_DEADLINE = object()


async def _narrate(
    client,
    req: NarrativeRequest,
    fs,
    derived,
    benchmark_refs,
    whitelist,
    *,
    budget_reserved: bool = False,
    forced_reason: str | None = None,
    deadline=_LEGACY_DEADLINE,
    slot_wait: float | None = None,
    effort: str | None = None,
    on_stage=_no_stage,
    trace: dict | None = None,
) -> NarrativeResponse:
    """서술 본문 — LLM 또는 규칙 폴백. 반환 경로가 여럿이라 공통 부착은 호출자가 한다.

    작업 번호 경로가 쓰는 인자(옛 경로는 모두 기본값):
    - budget_reserved: 작업을 만들 때 서술 지갑을 이미 예약했다 — 여기서 다시 쓰지 않는다.
    - forced_reason: AI 를 부르지 않고 이 사유의 규칙 문장으로 끝낸다(입력 상한 초과·지갑 소진 등).
    - deadline: 슬롯 대기 + 호출 전체 마감(초). 주지 않으면 옛 경로의 NARRATIVE_DEADLINE_SECONDS, None 이면 마감 없음.
    - slot_wait: 슬롯 대기만의 상한. 넘으면 규칙 문장(busy). None 이면 따로 두지 않는다(마감이 덮음).
    - effort: 이 호출의 effort. None 이면 llm 의 기본(NARRATIVE_EFFORT).
    - on_stage: 단계가 바뀔 때 부르는 async 함수 — generating(슬롯을 얻고 호출 직전) · checking(응답을 받고
      검사 시작).
    - trace: 내부 기록을 채울 dict(고객에게 가지 않는다) — reason·calls(호출 기록)·slot_wait_seconds·grounding.
    """
    record = trace if trace is not None else {}
    record.setdefault("calls", [])
    resp = await _narrate_body(
        client, req, fs, derived, benchmark_refs, whitelist, record,
        budget_reserved=budget_reserved, forced_reason=forced_reason,
        deadline=NARRATIVE_DEADLINE_SECONDS if deadline is _LEGACY_DEADLINE else deadline,
        slot_wait=slot_wait, effort=effort, on_stage=on_stage,
    )
    record["source"] = resp.meta.source
    record["reason"] = resp.meta.reason
    record["grounding"] = resp.meta.grounding.model_dump()
    return resp


def _rule_text(req, fs, benchmark_refs, derived, *, reason: str) -> NarrativeResponse:
    """규칙 문장(app.narrative.fallback) — 요청의 작업 유형·고객 입력을 넘긴다(S6: 결론 목록·클래스 설명 D37).
    옛 경로 흐름은 얼렸고, 규칙 문장만 작업 경로와 같은 템플릿을 쓴다(옛 경로 골든 — tests/test_s6_templates.py)."""
    return build_fallback_narrative(
        fs, benchmark_refs, derived, reason, task_type=req.task_type.value, client_inputs=req.client_inputs,
        edition=EDITION,
    )


async def _narrate_body(
    client, req, fs, derived, benchmark_refs, whitelist, record: dict, *,
    budget_reserved: bool, forced_reason: str | None, deadline: float | None, slot_wait: float | None,
    effort: str | None, on_stage,
) -> NarrativeResponse:
    # 0. 호출 전에 사유가 정해진 경우(작업 번호 경로) → 폴백
    if forced_reason is not None:
        return _rule_text(req, fs, benchmark_refs, derived, reason=forced_reason)

    # 1. 무키 → 폴백
    if client is None:
        return _rule_text(req, fs, benchmark_refs, derived, reason="no_key")

    # 1-1. 요청 조립 + 크기 상한 — 지갑을 쓰기 전에 본다(너무 큰 요청이 예산을 먹지 않게).
    try:
        system, user, prompt_trimmed = await run_cpu_bound(
            build_prompts_within_limit, req, fs, derived, benchmark_refs
        )
        schema = build_response_schema(fs.problems)
    except Exception:
        # 종전처럼 500 이 아니라 규칙 폴백 — 요청 조립 실패도 서술 실패의 하나다.
        logger.exception("서술 요청 조립 실패 → 규칙 폴백(assembly_error)")
        return _rule_text(req, fs, benchmark_refs, derived, reason="assembly_error")
    record["prompt_chars"] = len(system) + len(user)
    if prompt_trimmed:
        record["prompt_trimmed"] = prompt_trimmed
        logger.info("서술 지시문이 상한을 넘어 지시문에서만 덜어 냄: %s", prompt_trimmed)
    if len(system) + len(user) > MAX_NARRATIVE_PROMPT_CHARS:
        logger.warning(
            "서술 요청이 너무 큼(%d자 > %d자) → 규칙 폴백(input_too_large)",
            len(system) + len(user), MAX_NARRATIVE_PROMPT_CHARS,
        )
        return _rule_text(req, fs, benchmark_refs, derived, reason="input_too_large")

    # 1-2. 시간당 예산 소진 → 폴백(G-03, 결정 9).
    #      **429 가 아니라 200 + 폴백이다** — 프론트는 비 200 을 폴백이 아니라 오류로
    #      처리해 7·8·9절이 '생성 예정' 안내로 인쇄된다(사용자에게는 서비스 고장으로 보인다).
    #      서술 지갑은 매핑 지갑과 따로다 — 옛 경로는 호출 1번이 보고서 1건이다. 작업 번호 경로는 작업을
    #      만들 때 이미 예약했다(budget_reserved).
    if not budget_reserved and not llm_budget.narrative.try_consume():
        logger.warning(
            "서술 시간당 예산 소진(보고서 %d건) → 규칙 폴백으로 강등합니다.",
            llm_budget.narrative.limit,
        )
        return _rule_text(req, fs, benchmark_refs, derived, reason="budget_exceeded")

    # 2. LLM 호출 — 슬롯 대기까지 서버 마감 안에서. 슬롯 대기에 따로 상한이 있으면 그것도 본다.
    slot = {"acquired": False}
    waited_from = time.perf_counter()

    async def _call():
        async with asyncio.timeout(slot_wait):
            await stack.enter_async_context(narrative_slot())
        slot["acquired"] = True
        record["slot_wait_seconds"] = round(time.perf_counter() - waited_from, 3)
        await on_stage("generating")
        # 시도를 호출 **전에** 센다 — 바깥 제한(작업 전체 10분)이 호출 도중에 끊으면 호출 기록이 남지 않는데,
        # 요청은 이미 나가 과금됐을 수 있다. 내부 기록은 이 수로 '시도는 있었고 비용은 모름'을 적는다.
        record["attempts_started"] = record.get("attempts_started", 0) + 1
        return await call_narrative_llm(client, system=system, user=user, schema=schema, effort=effort)

    try:
        async with contextlib.AsyncExitStack() as stack, asyncio.timeout(deadline):
            result = await _call()
    except TimeoutError:
        reason = "timeout" if slot["acquired"] else "busy"
        logger.warning("서술 마감(마감 %s초·슬롯 대기 %s초) 초과 → 규칙 폴백(%s)", deadline, slot_wait, reason)
        return _rule_text(req, fs, benchmark_refs, derived, reason=reason)
    except Exception:
        # call_narrative_llm 은 SDK 오류를 failure 로 돌려준다. 그 밖의 예외(조용한 품질 강등 지점, G-06)도
        # 500 이 아니라 규칙 폴백으로.
        logger.exception("LLM 서술 호출 중 예기치 않은 오류 → 규칙 폴백(api_error)")
        return _rule_text(req, fs, benchmark_refs, derived, reason="api_error")
    record["calls"].append(asdict(result.record))
    await on_stage("checking")
    if result.failure is not None:
        # 거절(세 경우 모두)·잘림·모델 없음·시간 초과·API 오류·JSON 오류 — 사유는 그대로 싣고, 자세한 것은
        # llm 의 호출 기록 로그에 있다(거절 종류·응답 모델·토큰).
        logger.warning("LLM 서술 호출 실패 → 규칙 폴백(%s)", result.failure)
        return _rule_text(req, fs, benchmark_refs, derived, reason=result.failure)
    data = result.data
    model = result.record.response_model or result.record.requested_model

    # 3~5. grounding 검증 + 응답 조립. 스키마 불일치/누락으로 조립이 실패해도
    #      500 대신 규칙 폴백(assembly_error)으로 강등한다(D7[1]).
    try:
        interp = data.get("interpretation", {})
        concl = data.get("conclusion", {})
        rec_narr = data.get("recommendation_narrative", {})
        recs = data.get("recommendations", [])

        # LLM 이 생성한 모든 문자열 필드를 전수 수집해 검증(수동 나열 제거 — D3b).
        grounding = await run_cpu_bound(_verify_llm_output, data, whitelist)

        # 위반 시 report_purpose 무관하게 폴백(internal fail-open 제거 — D3c).
        if not grounding.passed:
            # 어떤 숫자가 걸렸는지까지 남긴다 — '환각이 있었다'만으로는 원인을 못 찾는다.
            logger.warning(
                "서술 grounding 위반 → 규칙 폴백(grounding_failed). 검사 %d건, 위반 토큰 %s",
                grounding.checked, grounding.violations,
            )
            fb = _rule_text(req, fs, benchmark_refs, derived, reason="grounding_failed")
            fb.meta.grounding = grounding  # 어떤 환각이 잡혔는지 추적성 보존
            return fb

        # 숫자가 없는 정성 서술은 위 검사를 **무조건 통과**한다(G-05).
        # 강제된 판정과 직접 모순되는 주장만 따로 잡는다 — 판정이 FAIL 인 성적서에
        # "모든 항목이 충족되었다"가 실리면 독자는 정반대 결론을 읽는다.
        contradictions = find_verdict_contradictions(_collect_grounding_texts(data), fs.verdict)
        if contradictions:
            logger.warning(
                "서술이 판정(%s)과 모순 → 규칙 폴백(verdict_contradiction): %s",
                fs.verdict, contradictions,
            )
            return _rule_text(req, fs, benchmark_refs, derived, reason="verdict_contradiction")

        # 권고를 문제 목록에 묶는다(D22). 기대 효과에 숫자가 있으면 전체 폴백 — 권고 하나만 바꿔 끼우면
        # 같은 LLM 이 쓴 9절 서술(recommendation_narrative)과 어긋날 수 있다.
        # AI 가 권고를 쓰지 않아 규칙 템플릿으로 채운 문제 — 9칸 출처를 AI 로만 적지 않게 내부 기록에 남긴다.
        answered = {str(r.get("problem_id") or "").strip() for r in recs if isinstance(r, dict)}
        record["recommendations_from_template"] = [p.id for p in fs.problems if p.id not in answered]
        try:
            aligned = _align_recommendations(recs, fs.problems)
        except _ImpactNumber as exc:
            logger.warning("권고 기대 효과에 숫자 → 규칙 폴백(impact_number): 문제 %s", exc)
            return _rule_text(req, fs, benchmark_refs, derived, reason="impact_number")

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
            recommendations=aligned,
            meta=NarrativeMeta(source="llm", model=model, grounding=grounding),
        )
    except Exception:
        logger.exception("narrative 조립/검증 실패 → 규칙 폴백(assembly_error)")
        return _rule_text(req, fs, benchmark_refs, derived, reason="assembly_error")
