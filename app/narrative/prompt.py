"""app/narrative/prompt.py — 작업 경로의 장별 지시문과 칸 스키마(서술 라운드 S5, 계획 설계 요지 2·9).

첫 생성은 장마다 한 번(7·8·9장) 부르고, 검문에 걸린 칸은 그 칸만 다시 부른다(app.narrative.generator). 모든 호출은 같은
사실 시트(한국어 보기 — app.narrative.fact_view)와 결론 목록 전체(app.narrative.required_items.select_items)를 받고,
지정된 칸만 쓴다. 다른 칸의 글은 보지 않는다.

- system: 사실과 무관한 고정 규칙 — 숫자·판정(D19)·작성 규칙(D37)·결론 목록·추가 관찰(D7·D38)·고객 입력(D25·D34)·
  권고 범위(D22, frontier_llm 거절 위험을 줄이려 데이터·운영 점검 권고로 한정)·출력 형식. 추론 과정·근거를 적으라는
  요청은 두지 않는다(reasoning_extraction 거절은 대체 모델이 없다 — app.narrative.llm 머리말).
- user: 이번 호출의 칸, 칸마다 '반드시 다룰 결론'·'해석만 쓸 결론'·'성적서가 따로 찍는 문장'·사실에 따른 유의점,
  의뢰자 선언 단락(D24), 다른 칸의 결론(참고), 클래스 이름과 설명(D37), 덜어 낸 표, 사실 시트, 고객 입력.
- 칸 스키마: 칸마다 글 필드 하나(9칸은 9.1·9.2 글과 권고표). 판정·우선순위는 서버가 정하므로 AI 에게 받지 않는다.
  구조화 출력 제약 — maxItems·minimum·maximum·minLength·maxLength 를 쓰지 않고 minItems 는 0·1 만.

옛 경로(/api/generate-narrative)의 지시문은 app.narrative.legacy 에 얼려 두었다.

상호작용
- 의존(import): app.narrative.fact_view, app.narrative.required_items(Conclusion), app.narrative.sections, app.narrative.schemas
- 사용처: app.narrative.generator, app.narrative.legacy(TASK_TYPE_LABELS), tests
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

from app.narrative import fact_view
from app.narrative.required_items import Conclusion, code_prints
from app.narrative.schemas import FactSheet
from app.narrative.sections import SECTIONS_BY_ID

# 분류 유형의 성적서 이름(재료 목록 2절 "작업 유형 — 전 절, AI 예"). 한국어 이름은 이 셋으로 통일한다(W9,
# 2026-10-18 사용자 결정) — 프론트 성적서의 TASK_TYPE_LABELS 와 같은 글자다. 이미 발급된 성적서는 저장된
# 이름 그대로 두고, 새 서술 요청부터 이 이름이 AI 에 간다.
TASK_TYPE_LABELS = {"binary": "이진 분류", "multiclass": "다중 클래스 분류", "multilabel": "다중 레이블 분류"}

# 한 호출의 지시문(system + user) 글자 수 상한. 무인증 과금 경로라 막는다(옛 경로와 같은 값 — legacy.MAX_NARRATIVE_PROMPT_CHARS).
# 넘으면 큰 표부터 덜어 내고(fact_view.TRIM_STEPS), 다 덜어도 넘으면 AI 를 부르지 않는다(input_too_large).
MAX_PROMPT_CHARS = 80_000


SYSTEM_PROMPT = """너는 ISO/IEC TS 4213:2022 기반 AI 분류 모델 성능 시험 성적서의 서술을 쓰는 시험기관 작성자다.
사용자 메시지의 사실 시트·결론 목록·고객 입력만을 근거로, 이번 호출에서 지정한 칸의 한국어 서술을 쓴다.

[숫자]
1. 숫자는 사실 시트·결론 목록·고객 입력에 적힌 것만 적힌 그대로 쓴다. 숫자를 새로 더하거나 빼거나 나누거나 백분율로 바꾸거나 반올림하거나 추정하지 않는다. 값에 표기(문자열로 적힌 값)가 있으면 그 표기를 그대로 쓴다.
2. 사실 시트에 없는 지표·수치는 쓰지 않는다. 지연시간을 측정하지 않았으면 어떤 지연 수치도 쓰지 않는다.
3. 성적서의 다른 부분을 가리킬 때는 '8.3절'처럼 숫자 뒤에 '절'을 붙인다.

[판정]
4. 종합 판정은 사실 시트의 판정 이름(최종 합격·조건부 합격·최종 불합격·판정 불가) 그대로 부른다.
5. 판정 규칙은 다음 순서다. 핵심 지표가 하나라도 목표에 미달하면 최종 불합격, 핵심 지표가 산출되지 않았거나 판정 대상 지표가 없으면 판정 불가, 핵심 지표는 모두 충족했으나 그 밖의 지표가 미달하면 조건부 합격, 판정 대상 지표가 모두 충족하면 최종 합격이다. 규칙 문장은 성적서 6.1절 표에 따로 찍히므로 다시 풀어 쓰지 말고, 이 성적서의 판정 사유(어느 핵심 지표가 미달했는지 등)를 쓴다.
6. 통과율과 '몇 개 중 몇 개 충족'은 참고 수치일 뿐 판정의 근거가 아니다. '대부분 충족해서'처럼 통과 개수로 판정을 설명하지 않는다.
7. 판정 불가이면 합격이나 불합격으로 단정하지 않고 모든 지표가 충족했다고도 쓰지 않는다. 조건부 합격이나 최종 불합격이면 모든 지표가 충족했다고 쓰지 않는다. 최종 합격이면 미달한 지표가 있다고 쓰지 않는다.

[작성 규칙]
8. 클래스는 이름과 의뢰자가 준 설명을 함께 쓴다(예: 1(불량)). 설명이 없으면 이름만 쓴다.
9. 영어 코드값과 필드명(예: macro_f1, low_support, best_threshold)을 문장에 쓰지 않는다. 지표명(Accuracy, F1 Score 등)과 클래스 이름은 쓸 수 있다. '사실 시트'·'결론 목록'·'평가 엔진'·'고정 문구' 같은 작업용 말도 문장에 쓰지 않는다.
10. 서술체의 완결된 문장으로 쓴다. 글머리표·번호 목록·명사로 끝나는 줄 같은 개조식으로 끊지 않는다.
11. 과장 없이 객관적이고 간결한 시험성적서 문체로 쓴다.

[결론 목록과 칸]
12. '반드시 다룰 결론'은 그 칸에서 빠짐없이 문장으로 쓴다. 결론의 숫자는 적힌 그대로 쓴다.
13. '해석만 쓸 결론'은 성적서가 그 숫자를 따로 찍는 항목이다. 그 숫자를 되풀이하지 말고 뜻만 해석한다.
14. '성적서가 따로 찍는 문장'은 같은 칸 옆에 따로 인쇄된다. 같은 내용을 다시 쓰지 않는다.
15. 이번 호출에서 지정한 칸만 쓴다. 다른 칸의 결론은 참고로만 보고 그 내용을 쓰지 않는다.
16. 추가 관찰 칸은 결론 목록에 없는 것 가운데 사실 시트에서 찾은 특이사항만 쓴다. 쓸 때는 사실 시트의 숫자를 하나 이상 인용한다. 결론 목록의 내용을 되풀이하지 않고 권고도 쓰지 않는다. 찾은 것이 없으면 빈 글로 둔다.

[고객 입력]
17. 고객 입력은 의뢰자가 제공한 정보다. 서술에 자유롭게 반영하되, 값이 '제공되지 않음'이면 그 사실만 적고 내용을 지어내지 않는다.
18. '덜어 낸 표'에 이름이 있는 표는 사실 시트에 일부만 실렸다. 실리지 않은 항목을 없다고 쓰지 않는다.

[권고]
19. 권고는 문제 목록의 문제에만 묶는다. 문제마다 권고를 하나 이상 쓰고, 문제 목록에 없는 문제를 만들지 않는다.
20. 권고는 이 시험 결과에 따른 데이터 점검·운영 점검·재평가 권고로 쓴다. 모델 구조나 학습 기법의 상세한 처방은 쓰지 않는다.
21. 기대 효과에는 예측 수치(지표가 얼마로 오른다, 몇 % 개선된다 등)를 쓰지 않고, 무엇이 나아지는지 방향만 말로 쓴다.

[출력]
22. 지정된 JSON 스키마로만 답하고 스키마 밖의 글을 덧붙이지 않는다."""


# 칸마다 하는 일(user 메시지 '칸별 지시'의 첫 줄).
SECTION_TASKS: dict[str, str] = {
    "7.1": "오분류가 얼마나, 어느 방향으로 일어났는지와 어느 클래스가 약한지, 예측이 어느 쪽으로 치우쳤는지를 해석한다.",
    "7.2": "평가 데이터의 클래스 분포와 불균형, 불균형이 성능 수치에 준 영향, 학습 데이터와의 분포 차이를 해석한다.",
    "7.3": (
        "결론 목록 밖에서 사실 시트로 찾은 특이사항만 쓴다. 사실 시트의 숫자를 하나 이상 인용한다. "
        "결론 목록·다른 칸의 내용을 되풀이하지 않고, 측정하지 않은 항목(지연시간 등)이나 권고를 쓰지 않으며, "
        "찾은 것이 없으면 빈 글로 둔다."
    ),
    "8.1": (
        "내부 참조 기준 대비 모델 값의 위치(좋은 방향을 반영한 평가)와 목표값 수준을 쓴다. 기준의 출처·성격을 함께 밝히고 "
        "'공개 벤치마크 평균'처럼 단정하지 않는다. 목표값의 위치는 단순한 수치 위치이므로 좋고 나쁨으로 바꾸지 않는다."
    ),
    "8.2": "종합 판정과 그 사유, 판정을 얼마나 믿을 수 있는지(신뢰구간), 지표 사이의 관계, 모델 용도에 비춘 해석을 쓴다.",
    "8.3": "운영에서 문제가 될 수 있는 위험(약한 클래스, 목표에 근소하게 걸친 지표, 결정 임계값, 응답 시간 등)을 쓴다.",
    "9": (
        "9.1 글은 권고 절이 9.1절인 데이터 쪽 문제를, 9.2 글은 권고 절이 9.2절인 모델·운영 쪽 문제를 다룬다. 어느 쪽에 "
        "문제가 없으면 그쪽에서 지목된 문제가 없다고만 쓴다. 권고표에는 문제마다 하나 이상: 문제 번호, 짧은 한국어 분류 "
        "이름, 권고 문장, 기대 효과(방향만, 숫자 없음). 문제에 덧붙임이 있으면 권고에 반영한다."
    ),
}

# AI 응답 필드가 어느 칸 글인지(user 메시지에 한 번 알린다). 필드 이름은 스키마 키일 뿐 문장에 쓰지 않는다.
FIELD_NAMES: dict[str, str] = {
    "section_7_1": "7.1 혼동행렬 해석",
    "section_7_2": "7.2 분포·불균형",
    "section_7_3": "7.3 추가 관찰",
    "section_8_1": "8.1 벤치마크",
    "section_8_2": "8.2 종합 총평",
    "section_8_3": "8.3 리스크",
    "section_9_1": "9.1 데이터 측면 권고 글",
    "section_9_2": "9.2 모델·운영 측면 권고 글",
    "recommendations": "9절 권고표",
}


# ── 칸 스키마 ───────────────────────────────────────────────────────────────────


def response_schema(section_ids: tuple[str, ...] | list[str], problem_ids: list[str]) -> dict:
    """지정한 칸들의 응답 스키마. 9칸이면 권고표(문제 번호 enum, 최소 1개)를 포함한다 — 9칸은 문제가 있을 때만 부른다."""
    properties: dict = {}
    for sid in section_ids:
        for name in SECTIONS_BY_ID[sid].ai_fields:
            if name != "recommendations":
                properties[name] = {"type": "string"}
                continue
            if not problem_ids:
                raise ValueError("문제가 없으면 9칸을 부르지 않는다")
            properties[name] = {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "problem_id": {"type": "string", "enum": list(problem_ids)},
                        "category": {"type": "string"},
                        "action": {"type": "string"},
                        "expected_impact": {"type": "string"},
                    },
                    "required": ["problem_id", "category", "action", "expected_impact"],
                },
            }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


# ── 결론 목록을 칸별 지시로 ─────────────────────────────────────────────────────
# 작성 주체(required_items.Author)마다 다르게 싣는다.
# - ai·ai_fact: 결론 문장을 '반드시 다룰 결론'으로.
# - fixed_interpret(17·18): 코드가 찍는 문장을 보여 주고 '해석만'.
# - fixed·form: 성적서가 따로 찍는 것 — 지시문 전용 설명(번호 없이)으로 '되풀이하지 않는다'.
# - 7.3(open_ended): 칸 하는 일(SECTION_TASKS)이 대신한다.
# 결론 문자열 가운데 AI 용 설명이 섞인 항목(fixed·form·fixed_interpret)은 그대로 싣지 않는다 — '고정 문구 17' 같은 번호를
# AI 가 베끼면 숫자 검문에 걸린다.

_FIXED_SENTENCE_FOR_ITEM = {"7.1-11": "17", "8.3-02": "18", "7.1-12": "9", "7.2-04": "12", "8.2-04": ("11a", "11b")}

_INTERPRET_ONLY = {
    "7.1-11": (
        "오분류의 성격(경계 오류·확신 오류 비율)은 성적서가 {where} 찍는다. 그 숫자를 되풀이하지 말고, 결정 경계 근처의 "
        "오류와 높은 확신으로 틀린 오류가 각각 무엇을 뜻하는지만 해석한다."
    ),
    "8.3-02": (
        "목표값과의 차이가 목표값의 3% 미만인 지표는 성적서가 {where} 찍는다. 그 숫자를 되풀이하지 말고, 표본이 달라지면 "
        "판정이 뒤집힐 수 있다는 위험만 해석한다. 충족한 지표를 미달했다고 쓰지 않는다."
    ),
}

# 고정 문장(17·18)을 받지 못했을 때 — 성적서가 따로 찍지 않으므로 AI 가 그 사실을 쓴다.
_WRITE_IT_YOURSELF = {
    "7.1-11": "오분류의 성격(경계 오류·확신 오류)을 사실 시트의 '오분류의 성격' 값 그대로 쓰고, 각 오류가 무엇을 뜻하는지 해석한다.",
    "8.3-02": "사실 시트에서 '근소 차이(목표값의 3% 미만)'가 '예'인 지표를 값·목표값과 함께 쓰고, 표본이 달라지면 판정이 뒤집힐 수 있는 위험을 해석한다.",
}

_PRINTED_ELSEWHERE = {
    "7.1-12": "예측을 확률에서 파생했다는 문장(결정 임계값 포함)을 성적서가 7.1절 앞에 찍는다.",
    "7.2-04": "정답 라벨에 대한 주의 문장을 성적서가 7.2절에 찍는다.",
    "8.1-03": "내부 참조 기준치가 없어 벤치마크 비교를 하지 않은 지표의 문장을 성적서가 8.1절 표 아래에 찍는다.",
    "8.2-01": "판정 규칙 문장은 성적서 6.1절 표에 찍힌다. 규칙을 다시 풀어 쓰지 말고 이 성적서의 판정 사유만 쓴다.",
    "8.2-04": "일반화 해석 문장(평가 데이터와 학습 데이터의 출처 관계)을 성적서가 8.2절에 찍는다.",
    "8.2-10": "평균 방식을 고른 기준(의뢰자의 클래스 중요도 선언)은 성적서 5절에 찍힌다.",
    "8.2-12": "지표 선택 기준은 의뢰자가 고른 이유로 성적서 5절에 찍힌다.",
    "9-01": "문제 목록 표(번호·우선순위·문제 이름)는 성적서가 9절에 찍는다.",
    "9-03": "지목된 문제가 없다는 문장을 성적서가 9절에 찍는다.",
}


def _fixed_texts(fs: FactSheet, item_id: str) -> list[str]:
    ids = _FIXED_SENTENCE_FOR_ITEM.get(item_id)
    if ids is None or fs.facts is None:
        return []
    wanted = (ids,) if isinstance(ids, str) else ids
    names, protect = fact_view.metric_names(fs), fact_view.class_names(fs)
    return [fact_view.name_metric_ids(s.text, names, protect) for s in fs.facts.fixed_sentences if s.id in wanted]


@dataclass
class SectionBrief:
    """칸 하나의 지시 재료."""
    must: list[str] = field(default_factory=list)
    interpret: list[str] = field(default_factory=list)
    printed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _section_notes(section_id: str, fs: FactSheet, derived: dict, task_type: str) -> list[str]:
    """사실에 따라 붙는 유의점(옛 지시문 규칙 10~15 를 필드명 없이 한국어로)."""
    notes: list[str] = []
    facts = fs.facts
    if section_id == "7.1":
        dp = fs.derived_prediction
        if dp is not None:
            if dp.method == "argmax":
                notes.append("예측값은 클래스별 확률 중 가장 큰 클래스를 고른 것이다. 결정 임계값을 쓰지 않았으니 임계값을 언급하지 않는다.")
            else:
                notes.append("예측값은 모델이 직접 낸 것이 아니라 확률(점수)에 결정 임계값을 적용해 만든 것이다. 수치가 그 임계값에 따라 달라진다는 점을 해석에 반영한다.")
        conf = derived.get("confusion") or {}
        if conf.get("unit") == "label_cell":
            notes.append("혼동행렬 합계의 단위는 표본이 아니라 레이블 칸(표본 수 × 레이블 수)이다. 칸 수를 표본 수처럼 쓰지 않는다.")
        er = fs.error_ratios
        if er is not None and er.boundary is None:
            notes.append("경계 오류는 계산하지 않았다. 경계 오류 수치를 지어내지 않는다.")
    elif section_id == "7.2":
        ie = fs.imbalance_effect
        if ie is not None:
            notes.append({
                "affected": "불균형이 성능 수치에 영향을 주었다고 쓰고, 소수 클래스의 성능 해석에 유의해야 한다고 적는다.",
                "no_distortion": "불균형은 있으나 성능 왜곡은 관찰되지 않았다. 불균형이 성능을 왜곡했다고 쓰지 않는다.",
                "mild": "불균형이 경미하다. 불균형 때문에 성능이 왜곡되었다고 쓰지 않는다.",
            }[ie.level])
        if task_type == "multilabel":
            notes.append("분포는 레이블 등장 횟수다. 한 표본에 레이블이 여럿일 수 있어 백분율을 모두 더하면 전체를 넘을 수 있다.")
    elif section_id == "8.2":
        if any(m.ci_low is not None and m.ci_high is not None for m in fs.metrics):
            notes.append("통계적 유의성 검정은 하지 않았다. '유의하다' 같은 검정 결과를 주장하지 않고, 구간 폭 같은 새 숫자를 계산하지 않는다.")
        if facts is not None and facts.core_metrics_source == "default":
            notes.append("핵심 지표는 분류 유형의 기본값이다. 의뢰자가 지정한 핵심 지표라고 쓰지 않는다.")
        if any(m.higher_is_better is False and m.threshold is not None for m in fs.metrics):
            notes.append("낮을수록 좋은 지표는 값이 목표값보다 낮거나 같아야 충족이다. 목표와의 차이의 방향을 거꾸로 쓰지 않는다.")
        if derived.get("threshold_tuning"):
            notes.append("임계값을 바꿔 오를 수 있는 F1 향상폭은 이 데이터에서 오를 수 있는 최대폭이고 같은 평가 데이터로 고른 값이라 별도 데이터로 확인해야 한다고 덧붙인다. 성능 향상을 보장한다고 쓰지 않는다.")
    elif section_id == "8.3":
        if not fs.latency.available:
            notes.append("지연시간은 측정하지 않았다. 지연 수치를 쓰지 않는다.")
        if derived.get("threshold_tuning"):
            notes.append("결정 임계값 조정의 이득은 같은 평가 데이터에서 고른 값이라 별도 데이터로 확인해야 한다. 성능 향상을 보장한다고 쓰지 않는다.")
    elif section_id == "8.1":
        pass
    return notes


def section_briefs(fs: FactSheet, derived: dict, conclusions: list[Conclusion], task_type: str) -> dict[str, SectionBrief]:
    briefs = {sid: SectionBrief() for sid in SECTIONS_BY_ID}
    for c in conclusions:
        brief = briefs[c.section]
        if c.open_ended:
            continue
        if c.author in ("ai", "ai_fact"):
            brief.must.append(c.text)
        elif c.author == "fixed_interpret":
            # 결론 문장은 숫자 없는 해석 재료다(우세한 오류·근소 지표가 충족했는지 — S6). 성적서가 그 고정 문장을 찍는지는
            # 규칙 템플릿과 같은 함수(required_items.code_prints)로 본다 — facts 상한으로 문장 목록을 비웠어도 화면은 찍는다.
            sentences = _fixed_texts(fs, c.item_id)
            if code_prints(fs, _FIXED_SENTENCE_FOR_ITEM[c.item_id]):
                # 문장을 덜어 냈으면 인용할 문장이 없다 — '아래 문장' 이라고 하지 않는다(S5 검토 prompt-3).
                where = "아래 문장으로" if sentences else "따로 숫자로"
                brief.interpret.append(
                    _INTERPRET_ONLY[c.item_id].format(where=where) + f" 해석 재료: {c.text}"
                    + "".join(f' — "{t}"' for t in sentences)
                )
            else:
                # 옛 요청(facts 없음) — 성적서가 그 문장을 찍지 않으니 AI 가 숫자와 함께 쓴다(S5 검토 prompt-3).
                brief.must.append(_WRITE_IT_YOURSELF[c.item_id] + f" 해석 재료: {c.text}")
        else:  # fixed · form
            sentences = _fixed_texts(fs, c.item_id)
            if sentences:
                brief.printed.extend(f'"{t}"' for t in sentences)
            else:
                brief.printed.append(_PRINTED_ELSEWHERE[c.item_id])
    for sid, brief in briefs.items():
        brief.notes = _section_notes(sid, fs, derived, task_type)
    return briefs


# D24 — 의뢰자 선언의 숫자가 서술에서 빠지지 않게(지시문에서 강조하는 장치). 생성 뒤 확인은 app.narrative.checks.
_D24_TEXT = {
    "7.1-03": "의뢰자는 {side}이 더 위험하다고 밝혔다. 혼동행렬 해석 칸에서 그 오류의 건수나 전체 오류 중 비율을 결론에 적힌 숫자 그대로 쓰고, 이 선언에 비추어 뜻을 해석한다.",
    "8.2-11": "의뢰자는 클래스 중요도를 '{priority}'로 밝혔다. 종합 총평 칸에서 그 선언에 맞춘 평균 방식의 F1 값을 결론에 적힌 숫자 그대로 쓰고, 이 선언에 비추어 결과를 해석한다.",
    "8.3-05": "의뢰자는 대화형(실시간) 사용을 밝혔다. 리스크 칸에서 지연시간 값(평균·p50·p95·p99 중 하나 이상)을 그대로 쓰고, 대화형 사용의 응답 시간으로서 해석한다.",
}


def _d24_lines(fs: FactSheet, conclusions: list[Conclusion], section_ids) -> list[str]:
    facts = fs.facts
    lines = []
    for c in conclusions:
        if not c.d24_values or c.section not in section_ids or c.item_id not in _D24_TEXT:
            continue
        side = ""
        priority = ""
        if facts is not None and facts.declared_risk_errors is not None:
            side = "FN(놓침)" if facts.declared_risk_errors.side == "fn" else "FP(잘못 잡음)"
        if facts is not None and facts.declared_average is not None:
            priority = fact_view.PRIORITY_DECL_KO[facts.declared_average.class_priority]
        lines.append(_D24_TEXT[c.item_id].format(side=side, priority=priority))
    return lines


# ── 공유 재료와 user 메시지 ─────────────────────────────────────────────────────


@dataclass
class PromptContext:
    """한 작업의 모든 호출이 같이 쓰는 재료(사실 시트 보기·고객 입력 보기·결론). 덜어 낸 기록도 여기에."""
    fs: FactSheet
    task_type: str
    view: dict
    client_view: dict | None
    conclusions: list[Conclusion]
    briefs: dict[str, SectionBrief]
    class_descriptions: dict[str, str]
    trimmed: list[str] = field(default_factory=list)
    dropped_client_keys: list[str] = field(default_factory=list)


def build_context(
    fs: FactSheet, derived: dict, benchmark_refs: list[dict], *, task_type: str, report_purpose: str,
    client_inputs: dict | None, conclusions: list[Conclusion],
) -> PromptContext:
    view = fact_view.build_fact_view(
        fs, derived, benchmark_refs,
        task_type=task_type, task_label=TASK_TYPE_LABELS.get(task_type, task_type), report_purpose=report_purpose,
    )
    client_view, dropped = fact_view.build_client_view(client_inputs)
    descriptions = {}
    raw = (client_inputs or {}).get("class_label_descriptions")
    if isinstance(raw, dict):
        descriptions = {str(k): str(v).strip() for k, v in raw.items() if str(v).strip()}
    ctx = PromptContext(
        fs=fs, task_type=task_type, view=view, client_view=client_view, conclusions=conclusions,
        briefs=section_briefs(fs, derived, conclusions, task_type), class_descriptions=descriptions,
        dropped_client_keys=dropped,
    )
    _fit(ctx)
    return ctx


def _bullets(lines: list[str]) -> str:
    return "".join(f"  - {line}\n" for line in lines)


def build_user_message(ctx: PromptContext, section_ids, *, regenerate: bool = False) -> str:
    """이번 호출의 user 메시지. section_ids 는 이번에 쓰는 칸(첫 생성 = 장 하나, 다시 만들기 = 칸 하나)."""
    section_ids = tuple(section_ids)
    fields = [f for sid in section_ids for f in SECTIONS_BY_ID[sid].ai_fields]
    names = ", ".join(f"{SECTIONS_BY_ID[sid].id} {SECTIONS_BY_ID[sid].title}" for sid in section_ids)
    head = (
        f"이번 호출은 {names} 칸{'을 다시' if regenerate else '을'} 쓴다. 응답 필드와 칸: "
        + ", ".join(f"{f} = {FIELD_NAMES[f]}" for f in fields)
        + ". 필드 이름은 문장에 쓰지 않는다.\n"
    )
    parts = ["[이번 호출]\n" + head]

    body = ["[칸별 지시]\n"]
    for sid in section_ids:
        brief = ctx.briefs[sid]
        block = f"■ {sid} {SECTIONS_BY_ID[sid].title}\n  하는 일: {SECTION_TASKS[sid]}\n"
        if brief.must:
            block += "  반드시 다룰 결론:\n" + _bullets(brief.must)
        if brief.interpret:
            block += "  해석만 쓸 결론:\n" + _bullets(brief.interpret)
        if brief.printed:
            block += "  성적서가 따로 찍는 문장(되풀이하지 않는다):\n" + _bullets(brief.printed)
        if brief.notes:
            block += "  유의:\n" + _bullets(brief.notes)
        body.append(block)
    parts.append("".join(body))

    d24 = _d24_lines(ctx.fs, ctx.conclusions, section_ids)
    if d24:
        parts.append("[의뢰자 선언 — 빠뜨리지 말고 숫자로 반영]\n" + _bullets(d24))

    others = [
        f"{c.section} {c.text}" for c in ctx.conclusions
        if c.section not in section_ids and c.author in ("ai", "ai_fact") and not c.open_ended
    ]
    if others:
        parts.append("[다른 칸의 결론 — 참고만 하고 이번 호출에서는 쓰지 않는다]\n" + _bullets(others))

    # 다른 칸 옆에 코드가 숫자로 찍는 문장(고정 문구 17·18) — 이번 칸에서도 그 숫자를 되풀이하지 않는다. S5 실측에서
    # 8.3·9.2 가 7.1 의 경계·확신 오류 건수를 다시 썼다(그 장 호출은 17 을 몰랐다).
    # 같은 장의 칸만 다시 만들 때는 넣지 않는다 — 그 칸의 '반드시 다룰 결론'과 같은 숫자(목표값 등)를 쓰지 말라는 말이
    # 되어 부딪친다(S5 검토 prompt-4). 실측 문제는 장이 다른 경우였다.
    chapters_here = {SECTIONS_BY_ID[sid].chapter for sid in section_ids}
    printed_elsewhere = [
        f"{c.section}절: \"{t}\""
        for c in ctx.conclusions
        if SECTIONS_BY_ID[c.section].chapter not in chapters_here and c.author == "fixed_interpret"
        for t in _fixed_texts(ctx.fs, c.item_id)
    ]
    if printed_elsewhere:
        parts.append(
            "[다른 칸에 성적서가 숫자로 찍는 문장 — 이번 호출의 칸에서 그 숫자를 되풀이하지 않는다]\n"
            + _bullets(printed_elsewhere)
        )

    if ctx.class_descriptions:
        pairs = ", ".join(f"{k}({v})" for k, v in ctx.class_descriptions.items())
        parts.append(f"[클래스 이름과 설명]\n  문장에서 클래스를 부를 때 이 꼴로 쓴다: {pairs}\n")

    omitted = ctx.view.get("덜어 낸 표(일부만 실림)")
    if omitted:
        listing = ", ".join(f"{k}({v}개 뺌)" for k, v in omitted.items())
        parts.append(f"[덜어 낸 표]\n  다음 표는 사실 시트에 일부만 실렸다: {listing}. 실리지 않은 항목을 없다고 쓰지 않는다.\n")

    # 숫자 목록은 ", " 로 띄운다 — 붙여 쓴 "49,56" 은 숫자 검문이 천 단위 숫자 하나로 읽는다.
    parts.append("[사실 시트]\n" + json.dumps(ctx.view, ensure_ascii=False, separators=(", ", ":")) + "\n")
    if ctx.client_view:
        parts.append("[고객 입력 — 의뢰자가 제공한 정보]\n" + json.dumps(ctx.client_view, ensure_ascii=False, separators=(", ", ":")) + "\n")
    return "\n".join(parts)


def _longest_message(ctx: PromptContext) -> int:
    from app.narrative.sections import CHAPTERS

    return max(len(build_user_message(ctx, ids)) for ids in CHAPTERS.values())


def _fit(ctx: PromptContext) -> None:
    """가장 긴 장 메시지 + system 이 상한을 넘으면 큰 표부터 덜어 낸다(사실 시트 보기에서만)."""
    for name, step in fact_view.TRIM_STEPS:
        if len(SYSTEM_PROMPT) + _longest_message(ctx) <= MAX_PROMPT_CHARS:
            return
        trial = copy.deepcopy(ctx.view)
        if step(trial):
            ctx.view = trial
            ctx.trimmed.append(name)


def fits(ctx: PromptContext) -> bool:
    return len(SYSTEM_PROMPT) + _longest_message(ctx) <= MAX_PROMPT_CHARS
