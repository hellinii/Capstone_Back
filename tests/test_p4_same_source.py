"""tests/test_p4_same_source.py — 판 4 라운드 S2 ③: 문제 종류 same_source(평가 데이터가 학습과 같은 출처, J8)와 폴백 권고(B23).

- 판 4 요청에만 온다. 판 3 요청(계약 예시·발급본·얼린 입력)에는 없다 — tests/test_p4_contract_ed4.py 가 본다.
- 폴백 권고는 계획 부록 B23 승인 글 그대로다. 모든 종류에 템플릿이 있다는 시험은 test_problem_recommendations 가 본다.
- 권고표는 문제 번호 순서·문제의 우선순위(LOW)를 따르고, 권고 절은 받은 칸 이름(9.1 — 데이터 쪽)으로 가른다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.narrative import generator
from app.narrative.fallback import _PROBLEM_TEMPLATES, problem_recommendation
from app.narrative.schemas import NarrativeRequest, ProblemFact

CONTRACT = Path(__file__).parent / "contract"
SPEC = json.loads((CONTRACT / "fact_sheet_contract_ed4.json").read_text(encoding="utf-8"))

# 계획 부록 B23(2026-10-24 승인) — 글자 그대로.
B23 = (
    "다른 출처 데이터로 재평가",
    "학습 데이터와 다른 출처(다른 기간·장비·기관 등)에서 얻은 평가 데이터로 재평가한다",
    "다른 운영 조건에서도 판정이 유지되는지 확인할 근거 확보",
)


def test_same_source_is_the_last_kind():
    kinds = ProblemFact.model_fields["kind"].annotation.__args__
    assert kinds[-1] == "same_source" and kinds[-2] == "label_caution"  # LOW 맨 끝(label_caution 다음)
    assert len(kinds) == 10


def test_same_source_template_is_b23_word_for_word():
    assert _PROBLEM_TEMPLATES["same_source"] == B23
    shape = SPEC["problems"]["same_source"]["shape"]
    p = ProblemFact(**{**shape, "id": "P2"})
    rec = problem_recommendation(p)
    assert (rec.problem_id, rec.priority) == ("P2", "LOW")
    assert (rec.category, rec.action, rec.expected_impact) == B23


def test_same_source_shape_from_spec_validates():
    shape = SPEC["problems"]["same_source"]["shape"]
    p = ProblemFact(**{**shape, "id": "P9"})
    assert p.kind == "same_source" and p.recommend_section == "9.1" and p.priority == "LOW"
    with pytest.raises(ValidationError):
        ProblemFact(**{**shape, "id": "P9", "kind": "same_sauce"})


async def test_edition4_example_rule_text_answers_same_source_on_data_side():
    raw = json.loads((CONTRACT / "fact_sheet_binary_ed4.json").read_text(encoding="utf-8"))
    req = NarrativeRequest.model_validate(raw)
    assert [p.kind for p in req.fact_sheet.problems] == ["core_near_target", "same_source"]
    gen = await generator.generate_sections(None, req)
    recs = gen.response.recommendations
    assert [(r.problem_id, r.priority) for r in recs] == [("P1", "MEDIUM"), ("P2", "LOW")]
    assert (recs[1].category, recs[1].action, recs[1].expected_impact) == B23
    # 9.1(데이터 쪽)에 같은 출처 문제가, 9.2(모델 쪽)에 근소 합격 문제가 간다.
    assert "P2(" in gen.response.recommendation_narrative.data_quality
    assert "P2(" not in gen.response.recommendation_narrative.model_ops
    assert "P1(" in gen.response.recommendation_narrative.model_ops
