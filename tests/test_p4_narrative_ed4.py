"""tests/test_p4_narrative_ed4.py — 판 4 임시 예시로 판 4 작업 경로를 끝까지(판 4 라운드 S2 ⑨).

입력은 tests/contract/fact_sheet_binary_ed4.json(S2 ① 의 **임시** 판 4 예시 — S3 ⑧ 에서 프론트가 뜬 예시로 바뀐다). 예시에 묶인
기대 글자는 tests/p4_example_ed4.py 한 곳에만 있다 — 예시를 바꾸면 그 파일만 고친다.

- 키 없음: 일곱 칸 규칙 문장이 예시본 5·6절 꼴의 판 4 글(판 4 표기·6절 권고표·B22·B24·통과율 없음)이고 판 4 검문을 통과한다.
- AI 경로: 판 4 system·판 4 자리로 부르고, AI 가 판 4 절 번호('(4.3)'·'6.2절')와 판 4 표기를 쓰면 받는다. 넷째 자리 표기
  (0.9767)를 쓰면 그 칸만 다시 만들고 또 걸리면 규칙 문장 — 같은 글을 판 3 쌍둥이 요청은 받는다(판이 검문까지 간다).
- 9칸 권고가 빠진 문제는 판 4 종류별 폴백(B24)으로 채운다.
"""
from __future__ import annotations

import copy

from app.narrative import generator, prompt
from app.narrative.jobs import parse_request
from app.narrative.sections import SECTION_IDS
from tests.conftest import requested_fields
from tests.p4_example_ed4 import EXAMPLE_TEMPLATES, as_edition, example_raw

OK_TEXTS = {
    "section_7_1": "전체 171건 중 3건(1.8%)이 오분류되었고, 의뢰자가 더 위험하다고 밝힌 거짓 음성은 1건(33.3%)이다.",
    "section_7_2": "평가 데이터의 불균형 비율은 1.672이고 학습·평가 비율 차는 −0.2%p와 +0.2%p이다.",
    "section_7_3": "F1이 가장 높은 결정 임계값은 0.6166이고, 그 임계값에서 Recall은 0.984로 목표 0.970을 유지한다.",
    "section_8_1": "Accuracy(0.982)는 참조 범위 0.85~0.92보다 높다. 이 기준은 자체 설정 기준이다.",
    "section_8_2": "종합 판정은 최종 합격이다. Accuracy 의 구간 0.9497~0.994 가 목표 0.950 을 포함한다(4.3).",
    "section_8_3": "판정의 여유가 작은 지표가 있어 표본이 달라지면 판정이 뒤집힐 수 있다. 이 위험에 대한 권고는 6.2절에서 다룬다.",
    "section_9_1": "평가 데이터가 학습과 같은 출처이므로 다른 출처의 데이터로 재평가를 권고한다.",
    "section_9_2": "핵심 지표의 판정 여유가 작아 운영 중 점검과 재평가를 권고한다.",
}


def _req(raw: dict):
    req, _ = parse_request(copy.deepcopy(raw))
    return req


def _parts(response) -> dict:
    d = response.model_dump(mode="json")
    out = {f"interpretation.{k}": v for k, v in d["interpretation"].items()}
    out |= {f"conclusion.{k}": d["conclusion"][k] for k in ("benchmark", "narrative", "risks")}
    out |= {f"recommendation_narrative.{k}": v for k, v in d["recommendation_narrative"].items()}
    return out


async def test_rule_sentences_of_the_example_without_a_key():
    result = await generator.generate_sections(None, _req(example_raw()))
    assert _parts(result.response) == EXAMPLE_TEMPLATES
    assert result.section_sources["7.3"] == "fixed" and result.response.meta.reason == "no_key"
    recs = {r.problem_id: r for r in result.response.recommendations}
    assert recs["P1"].action.startswith("판정 여유가 작은 채 합격한 핵심 지표를")  # B24
    assert recs["P2"].category == "다른 출처 데이터로 재평가"  # B23(S2a)


async def test_ai_path_accepts_edition4_text(make_section_client):
    client = make_section_client(OK_TEXTS)
    result = await generator.generate_sections(client, _req(example_raw()))
    sent = client.beta.messages.create.call_args_list
    assert all(c.kwargs["system"] == prompt.SYSTEM_PROMPT_ED4 for c in sent)
    for sid in SECTION_IDS:
        checks_ = result.trace["sections"][sid]["checks"]
        assert all(c.get("numbers", []) == [] for c in checks_), (sid, checks_)
    assert {sid: result.section_sources[sid] for sid in ("7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "9")} == dict.fromkeys(
        ("7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "9"), "llm",
    )
    assert result.response.conclusion.risks == OK_TEXTS["section_8_3"]
    assert len(sent) == 3  # 장 셋, 다시 만들기 없음


async def test_fourth_digit_number_is_regenerated_then_templated_only_in_edition4(make_section_client):
    bad = {**OK_TEXTS, "section_8_2": "F1 Score 는 0.9767로 목표를 충족하였다."}
    client = make_section_client(bad)
    result = await generator.generate_sections(client, _req(example_raw()))
    st = result.trace["sections"]["8.2"]
    assert st["http_calls"] == 2 and st["source"] == "fallback"
    assert [c["numbers"] for c in st["checks"]] == [["0.9767"], ["0.9767"]]
    assert result.response.conclusion.narrative == EXAMPLE_TEMPLATES["conclusion.narrative"]
    assert result.response.meta.grounding.violations == ["0.9767"]
    regen = [c.kwargs for c in client.beta.messages.create.call_args_list if requested_fields(c.kwargs) == ["section_8_2"]]
    assert len(regen) == 1 and "이번 호출은 5.1 판정과 불확실성 칸을 다시 쓴다." in regen[0]["messages"][0]["content"]
    # 같은 글을 판 3 쌍둥이 요청은 받는다(판 3 허용 목록은 넷째 자리를 받는다) — 판이 칸 검사까지 간다.
    twin = copy.deepcopy(example_raw())
    twin["fact_sheet"]["problems"] = [p for p in twin["fact_sheet"]["problems"] if p["kind"] != "same_source"]
    client3 = make_section_client(bad)
    result3 = await generator.generate_sections(client3, _req(as_edition(twin, 3)))
    assert result3.section_sources["8.2"] == "llm" and result3.trace["sections"]["8.2"]["http_calls"] == 1


async def test_missing_recommendation_is_filled_with_the_edition4_template(make_section_client):
    only_p2 = [{"problem_id": "P2", "category": "재평가", "action": "다른 출처의 데이터로 재평가한다.",
                "expected_impact": "다른 운영 조건에서의 판정 근거가 생긴다."}]
    client = make_section_client({**OK_TEXTS, "recommendations": only_p2})
    result = await generator.generate_sections(client, _req(example_raw()))
    assert result.trace["recommendations_from_template"] == ["P1"]
    p1 = next(r for r in result.response.recommendations if r.problem_id == "P1")
    assert p1.action.startswith("판정 여유가 작은 채 합격한 핵심 지표를")
