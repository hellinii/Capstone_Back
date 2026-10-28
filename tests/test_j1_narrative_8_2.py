"""tests/test_j1_narrative_8_2.py — 심판 라운드 1단계: 8.2 서술 두 건(서술 설계 v0.14, 판 3·4 모두).

1) 평균 방식 차이(8.2-07 다중 macro vs weighted, 8.2-08 멀티 macro vs micro — D17·D39)는 지표 선택(M22)과 무관하게
   늘 계산하는 클래스별 표의 평균 행(facts.average_gap)이 있으면 쓴다.
2) 평가 데이터의 대표성과 한계(8.2-14): 조건은 '평가 데이터 출처가 제공됨'(고객 입력 test_data 의 '평가 데이터 출처').
   AI 가 해석하되 고객 입력만으로 단정하지 않고 '~로 보인다'처럼 추론으로 쓰고, 규칙 템플릿은 사실 문장만 찍는다
   (심판 설계 6.2 소견-7 — 고객 정보에 기댄 단정형 해석은 위반).

세 겹을 본다 — 결론 목록(select_items), 키 없는 규칙 문장(generate_sections(None, …) 의 8.2 칸), 실제로 보내는 AI
지시문(가짜 클라이언트가 받아 적은 user 메시지). 판 4 는 8.2 칸이 5.1 자리다. 실제 LLM 은 부르지 않는다.
"""
import copy
import json
from pathlib import Path

import pytest

import frozen_s0 as fz
from app.narrative import generator
from app.narrative.required_items import REQUIRED_ITEMS, select_items
from app.narrative.schemas import NarrativeRequest
from app.narrative.sections import place

CONTRACT = Path(__file__).parent / "contract"
TASKS = ("binary", "multiclass", "multilabel")
EDITIONS = (3, 4)
SOURCE_KEY = "평가 데이터 출처"
PURPOSE = "생산 라인 2 의 불량 검출"
SOURCE = "2025년 1분기 고객 로그에서 무작위로 뽑은 표본"

# AI 에 주는 항목 설명에 꼭 있어야 하는 말 — 단정 금지와 추론 표현(심판 6.2 소견-7).
NO_ASSERT = "대표한다·대표하지 못한다고 단정하지 않고"
INFERENCE = "'~로 보인다'처럼 추론으로 쓴다"
# 규칙 템플릿(사실 문장)에 들어가면 안 되는 해석 말.
INTERPRETIVE = ("대표", "보인다", "한계", "해석", "적합", "적절")

_ABSENT = object()  # 키를 아예 두지 않음

# 제공 아님 꼴 — 키 없음, 빈 값, 공백, 글이 아닌 값, '제공되지 않음'(프론트 materialRowText), 모름 + 덧붙이는 글 꼴, '모름' 한 낱말.
NOT_PROVIDED_FORMS = {
    "키 없음": _ABSENT,
    "빈 값": "",
    "공백만": "   ",
    "글이 아님": None,
    "제공되지 않음": "제공되지 않음",
    "제공되지 않음+의뢰자 글": "제공되지 않음 (의뢰자 제공: 사내 로그 일부)",
    "모름": "모름",
    "모름(앞뒤 공백)": "  모름 ",
}


def _raw(task: str, edition: int) -> dict:
    """계약 예시(판 3 은 fact_sheet_{task}, 판 4 는 fact_sheet_{task}_ed4) 원문 사본."""
    name = f"fact_sheet_{task}" + ("_ed4" if edition == 4 else "")
    raw = json.loads((CONTRACT / f"{name}.json").read_text(encoding="utf-8"))
    assert raw.get("report_edition", 3) == edition
    return raw


def _with_source(task: str, edition: int, source, *, purpose: str | None = PURPOSE, test_data: bool = True) -> NarrativeRequest:
    raw = _raw(task, edition)
    inputs = copy.deepcopy(raw.get("client_inputs") or {})
    model = dict(inputs.get("model") or {})
    if purpose is None:
        model.pop("모델 용도", None)
    else:
        model["모델 용도"] = purpose
    inputs["model"] = model
    if not test_data:
        inputs.pop("test_data", None)
    else:
        test = dict(inputs.get("test_data") or {})
        test.pop(SOURCE_KEY, None)
        if source is not _ABSENT:
            test[SOURCE_KEY] = source
        inputs["test_data"] = test
    raw["client_inputs"] = inputs
    req = NarrativeRequest.model_validate(raw)
    assert req.report_edition == edition and req.task_type.value == task
    return req


def _conclusions(req: NarrativeRequest) -> dict:
    out = select_items(req.fact_sheet, req.client_inputs, req.task_type.value, edition=req.report_edition)
    assert out, "결론 목록이 비었다"
    return {c.item_id: c for c in out}


def _fact(source: str) -> str:
    return f"의뢰자가 밝힌 평가 데이터 출처는 '{source}'이다."


# ── 2) 평가 데이터의 대표성과 한계 — 결론 목록 ──────────────────────────────────────────


def test_table_row_for_representativeness():
    by_id = {i.id: i for i in REQUIRED_ITEMS}
    row = by_id["8.2-14"]
    assert (row.section, row.author, row.title) == ("8.2", "ai_fact", "평가 데이터의 대표성과 한계")
    assert row.task_types == set(TASKS)
    assert row.material == "평가 데이터 출처가 제공됨"
    assert row.template is not None  # ai_fact — 템플릿은 따로(사실 문장만)


@pytest.mark.parametrize("edition", EDITIONS)
@pytest.mark.parametrize("task", TASKS)
def test_source_given_brings_the_item_with_inference_wording(task, edition):
    c = _conclusions(_with_source(task, edition, SOURCE))["8.2-14"]
    assert (c.section, c.author) == ("8.2", "ai_fact")
    # AI 몫: 출처를 인용해 모델 용도를 얼마나 대표하는지와 한계를 해석 — 단정하지 않고 추론으로.
    assert f"평가 데이터 출처('{SOURCE}')" in c.text
    assert "모델 용도를 얼마나 대표하는지와 그 한계를 해석한다" in c.text
    assert NO_ASSERT in c.text and INFERENCE in c.text
    # 템플릿: 사실 문장 하나만(해석하지 않는다).
    assert c.template_text == _fact(SOURCE)
    assert not [w for w in INTERPRETIVE if w in c.template_text]


@pytest.mark.parametrize("edition", EDITIONS)
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("form", list(NOT_PROVIDED_FORMS))
def test_source_not_provided_drops_the_item(task, edition, form):
    picked = _conclusions(_with_source(task, edition, NOT_PROVIDED_FORMS[form]))
    assert "8.2-14" not in picked
    # 같은 요청에 출처만 넣으면 나온다(갈래가 조건 하나로만 갈린다).
    assert "8.2-14" in _conclusions(_with_source(task, edition, SOURCE))


@pytest.mark.parametrize("edition", EDITIONS)
@pytest.mark.parametrize("task", TASKS)
def test_no_test_data_group_or_relation_only_drops_the_item(task, edition):
    assert "8.2-14" not in _conclusions(_with_source(task, edition, SOURCE, test_data=False))
    # 학습 데이터와의 관계만 있어도 출처 글이 없으면 해석할 재료가 없다(조건은 '평가 데이터 출처'). 관계는 8.2-04(11a·11b) 몫.
    req = _with_source(task, edition, _ABSENT)
    req.client_inputs["test_data"]["학습 데이터와의 관계"] = "같은 출처"
    assert "8.2-14" not in _conclusions(req)


@pytest.mark.parametrize("edition", EDITIONS)
def test_unknown_inside_a_sentence_is_still_a_source(edition):
    """'모름'은 칸 전체가 그 낱말일 때만 제공 아님 — 글 속의 '모름'은 의뢰자가 적은 내용이라 값으로 둔다."""
    source = "수집 경로는 모름, 사내 로그"
    c = _conclusions(_with_source("binary", edition, source))["8.2-14"]
    assert c.template_text == _fact(source)


@pytest.mark.parametrize("edition", EDITIONS)
def test_source_is_trimmed(edition):
    c = _conclusions(_with_source("multiclass", edition, f"  {SOURCE}\n"))["8.2-14"]
    assert c.template_text == _fact(SOURCE)


@pytest.mark.parametrize("edition", EDITIONS)
def test_without_model_purpose_it_does_not_invent_one(edition):
    """모델 용도가 없으면 '용도를 대표하는지'를 묻지 않는다 — 없는 용도를 지어내지 않게."""
    c = _conclusions(_with_source("multilabel", edition, SOURCE, purpose=None))["8.2-14"]
    assert "모델 용도" not in c.text and "모델이 실제로 쓰일 환경을 얼마나 대표하는지" in c.text
    assert NO_ASSERT in c.text and INFERENCE in c.text
    assert c.template_text == _fact(SOURCE)


# ── 2) 규칙 문장(키 없음) — 8.2 칸(판 4 는 5.1 자리) ────────────────────────────────────


@pytest.mark.parametrize("edition", EDITIONS)
@pytest.mark.parametrize("task", TASKS)
async def test_rule_sentence_prints_only_the_fact(task, edition):
    given = (await generator.generate_sections(None, _with_source(task, edition, SOURCE))).response
    assert given.meta.reason == "no_key"
    narrative = given.conclusion.narrative
    # 8.2 칸 마지막 결론(표 순서 — 모델 용도 사실 문장 바로 뒤).
    assert narrative.endswith(f"의뢰자가 밝힌 모델 용도는 '{PURPOSE}'이다. {_fact(SOURCE)}")
    for form in ("키 없음", "모름", "제공되지 않음+의뢰자 글"):
        dropped = (await generator.generate_sections(None, _with_source(task, edition, NOT_PROVIDED_FORMS[form]))).response
        assert SOURCE_KEY not in dropped.conclusion.narrative, form
        # 출처 문장 하나만 빠지고 나머지 8.2 글은 같다.
        assert dropped.conclusion.narrative == narrative.removesuffix(f" {_fact(SOURCE)}"), form
    # 다른 칸에는 출처 문장이 없다(8.2 칸에만).
    others = [given.conclusion.benchmark, given.conclusion.risks, given.interpretation.confusion_analysis,
              given.interpretation.distribution_analysis, given.interpretation.additional_observations]
    assert all(_fact(SOURCE) not in (t or "") for t in others)


# ── 2) 실제 보내는 AI 지시문 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("edition", EDITIONS)
@pytest.mark.parametrize("task", TASKS)
async def test_prompt_asks_for_inference_in_the_8_2_slot(task, edition):
    req = _with_source(task, edition, SOURCE)
    c = _conclusions(req)["8.2-14"]
    chapters = (await fz.job_prompts(req))["chapters"]
    assert sorted(chapters) == ["7", "8", "9"]
    user8 = chapters["8"]["user"]
    # 8.2 칸(판 4 는 '5.1 판정과 불확실성') 블록의 '반드시 다룰 결론'에 그 글 그대로.
    block = user8.split(f"■ {place('8.2', edition=edition)} ", 1)[1].split("\n■ ", 1)[0].split("\n[", 1)[0]
    must = block.split("반드시 다룰 결론:\n", 1)[1]
    assert f"  - {c.text}\n" in must
    assert NO_ASSERT in must and INFERENCE in must
    # 다른 장의 호출은 '다른 칸의 결론'에 자리(판 3 8.2 / 판 4 5.1)를 붙여 참고로만 받는다.
    for chapter in ("7", "9"):
        user = chapters[chapter]["user"]
        assert f"  - {place('8.2', edition=edition)} {c.text}\n" in user, chapter
    # 판 4 지시문에는 판 3 자리 번호로 이 결론을 싣지 않는다.
    if edition == 4:
        assert all(f"  - 8.2 {c.text}" not in ch["user"] for ch in chapters.values())


@pytest.mark.parametrize("edition", EDITIONS)
@pytest.mark.parametrize("task", TASKS)
async def test_prompt_has_no_representativeness_without_source(task, edition):
    req = _with_source(task, edition, "모름")
    chapters = (await fz.job_prompts(req))["chapters"]
    assert chapters
    for chapter, sent in chapters.items():
        assert "얼마나 대표하는지" not in sent["user"], chapter
        assert INFERENCE not in sent["user"], chapter


# ── 1) 평균 방식 차이 — M22 를 고르지 않아도 ─────────────────────────────────────────

GAP_ITEM = {"multiclass": "8.2-07", "multilabel": "8.2-08"}
GAP_WORDS = {"multiclass": "가중 평균 F1 이 매크로 F1 보다", "multilabel": "마이크로 평균 F1 이 매크로 F1 보다"}


def _gap_req(task: str, edition: int, *, m22: bool) -> NarrativeRequest:
    raw = _raw(task, edition)
    facts = raw["fact_sheet"]["facts"]
    assert facts["average_gap"] is not None, "계약 예시에 평균 방식 차이 값이 있어야 한다"
    selected = [m for m in facts["selected_metrics"] if m != "M22"]
    facts["selected_metrics"] = selected + (["M22"] if m22 else [])
    return NarrativeRequest.model_validate(raw)


@pytest.mark.parametrize("edition", EDITIONS)
@pytest.mark.parametrize("task", list(GAP_ITEM))
def test_average_gap_is_written_without_m22(task, edition):
    req = _gap_req(task, edition, m22=False)
    assert "M22" not in req.fact_sheet.facts.selected_metrics
    picked = _conclusions(req)
    item = GAP_ITEM[task]
    assert item in picked and GAP_WORDS[task] in picked[item].text
    assert picked[item].author == "ai" and picked[item].template_text == picked[item].text
    # M22 를 골라도 같은 글(조건이 아니다).
    assert _conclusions(_gap_req(task, edition, m22=True))[item].text == picked[item].text
    # 다른 유형의 평균 방식 항목은 나오지 않는다.
    other = next(i for t, i in GAP_ITEM.items() if t != task)
    assert other not in picked


@pytest.mark.parametrize("edition", EDITIONS)
@pytest.mark.parametrize("task", list(GAP_ITEM))
def test_average_gap_needs_the_average_rows(task, edition):
    """반대 갈래 — 평균 행 값이 없으면(정답 클래스 둘 미만 등 프론트가 싣지 않음) M22 를 골라도 쓰지 않는다."""
    raw = _raw(task, edition)
    raw["fact_sheet"]["facts"]["average_gap"] = None
    raw["fact_sheet"]["facts"]["selected_metrics"].append("M22")
    assert GAP_ITEM[task] not in _conclusions(NarrativeRequest.model_validate(raw))


@pytest.mark.parametrize("edition", EDITIONS)
def test_binary_never_has_average_gap(edition):
    picked = _conclusions(_with_source("binary", edition, SOURCE))
    assert not {"8.2-07", "8.2-08"} & set(picked)


@pytest.mark.parametrize("edition", EDITIONS)
@pytest.mark.parametrize("task", list(GAP_ITEM))
async def test_average_gap_reaches_rule_sentence_and_prompt_without_m22(task, edition):
    req = _gap_req(task, edition, m22=False)
    c = _conclusions(req)[GAP_ITEM[task]]
    narrative = (await generator.generate_sections(None, req)).response.conclusion.narrative
    assert c.text in narrative
    user8 = (await fz.job_prompts(req))["chapters"]["8"]["user"]
    block = user8.split(f"■ {place('8.2', edition=edition)} ", 1)[1]
    assert f"  - {c.text}\n" in block.split("반드시 다룰 결론:\n", 1)[1]
