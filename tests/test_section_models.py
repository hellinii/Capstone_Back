"""test_section_models.py — 칸별 모델 이름(section_models, 서술 설계 결정 4 — 성적서 화면의 '작성 기록').

규칙: 키는 늘 일곱 칸(SECTION_IDS). 출처 llm 칸만 그 글을 낸 호출의 응답 모델, fallback·fixed 는 None.
'그 글을 낸 호출' — 다시 만든 글이면 다시 만든 호출, 첫 글을 둬 뒀다 실으면(kept_first) 첫 호출의 모델.
호출마다 다른 모델 이름으로 답해 어느 호출의 것인지 가른다. 사실 시트는 계약 이진 예시(문제 3개·D24 7.1).
"""
import json
from pathlib import Path

from app.narrative import generator, jobs
from app.narrative.schemas import NarrativeJobResult, NarrativeRequest
from app.narrative.sections import SECTION_IDS
from tests.conftest import fake_beta_message, requested_fields, section_reply

CONTRACT = Path(__file__).parent / "contract"
D24_OK = "의뢰자가 위험하다고 밝힌 FN 은 43건이다."
CHAPTER_MODEL = {"section_7_1": "model-ch7", "section_8_1": "model-ch8", "section_9_1": "model-ch9"}


def _binary() -> NarrativeRequest:
    return NarrativeRequest.model_validate(json.loads((CONTRACT / "fact_sheet_binary.json").read_text()))


def _replying(make_section_client, base: dict, *, override=None, n=8):
    """장 호출은 장마다 다른 모델로, 칸 하나 다시 만들기는 'model-regen-<칸>' 으로 답한다. override(kwargs)→dict|None."""
    def reply(kwargs):
        fields = requested_fields(kwargs)
        texts = {**base, **((override(kwargs) if override else None) or {})}
        model = CHAPTER_MODEL.get(fields[0]) if len(fields) > 1 else f"model-regen-{fields[0]}"
        return fake_beta_message(section_reply(kwargs, texts), model=model)

    return make_section_client(base, replies=[reply] * n)


async def test_each_llm_section_gets_its_chapters_model(make_section_client):
    fake = _replying(make_section_client, {"section_7_1": D24_OK})
    gen = await generator.generate_sections(fake, _binary())
    assert gen.section_sources == {"7.1": "llm", "7.2": "llm", "7.3": "fixed", "8.1": "llm", "8.2": "llm",
                                   "8.3": "llm", "9": "llm"}
    assert gen.section_models == {
        "7.1": "model-ch7", "7.2": "model-ch7", "7.3": None,
        "8.1": "model-ch8", "8.2": "model-ch8", "8.3": "model-ch8", "9": "model-ch9",
    }
    # 내부 기록에도 같은 값.
    assert {sid: sec["model"] for sid, sec in gen.trace["sections"].items()} == gen.section_models


async def test_regenerated_section_takes_the_regeneration_model(make_section_client):
    def override(kwargs):
        if requested_fields(kwargs)[0] == "section_8_1" and len(requested_fields(kwargs)) > 1:
            return {"section_8_2": "통과율 12.345% 이다."}  # 8장 첫 호출에만 위반
        return None

    fake = _replying(make_section_client, {"section_7_1": D24_OK}, override=override)
    gen = await generator.generate_sections(fake, _binary())
    assert gen.section_sources["8.2"] == "llm" and gen.trace["sections"]["8.2"]["outcome"] == "second"
    assert gen.section_models["8.2"] == "model-regen-section_8_2"
    assert gen.section_models["8.1"] == gen.section_models["8.3"] == "model-ch8"


async def test_section_that_falls_back_has_no_model(make_section_client):
    fake = _replying(make_section_client, {"section_7_1": D24_OK, "section_8_2": "통과율 12.345% 이다."})
    gen = await generator.generate_sections(fake, _binary())
    assert gen.section_sources["8.2"] == "fallback"
    assert gen.section_models["8.2"] is None
    assert gen.trace["sections"]["8.2"]["model"] is None


async def test_kept_first_draft_keeps_the_first_calls_model(make_section_client):
    """D24 만 빠진 첫 글을, 다시 만든 글이 검사에 걸려 그대로 싣는다 — 모델은 첫(7장) 호출의 것."""
    def override(kwargs):
        if requested_fields(kwargs) == ["section_7_1"]:
            return {"section_7_1": "오분류는 12.345% 이다."}
        return None

    fake = _replying(make_section_client, {"section_7_1": "오분류의 방향을 해석한 서술이다."}, override=override)
    gen = await generator.generate_sections(fake, _binary())
    assert gen.trace["sections"]["7.1"]["outcome"] == "kept_first"
    assert gen.section_models["7.1"] == "model-ch7"


async def test_filled_section_9_has_no_model(make_section_client):
    """권고가 빠져 종류별 템플릿으로 채운 9칸은 출처 fallback — 모델이 없다."""
    only_p1 = [{"problem_id": "P1", "category": "점검", "action": "해당 문제를 점검한다.", "expected_impact": "오류가 줄어든다."}]
    fake = _replying(make_section_client, {"section_7_1": D24_OK, "recommendations": only_p1})
    gen = await generator.generate_sections(fake, _binary())
    assert gen.trace["sections"]["9"]["outcome"] == "filled"
    assert gen.section_sources["9"] == "fallback" and gen.section_models["9"] is None


async def test_final_recheck_template_has_no_model(make_section_client, monkeypatch):
    from app.narrative import checks

    monkeypatch.setattr(checks, "check_section", lambda *a, **k: checks.SectionCheck(section=a[0]))
    fake = _replying(make_section_client, {"section_7_1": D24_OK, "section_8_2": "종합 판정은 합격이다."})
    gen = await generator.generate_sections(fake, _binary())
    assert "8.2" in gen.trace["final_recheck"]
    assert gen.section_sources["8.2"] == "fallback" and gen.section_models["8.2"] is None


async def test_no_key_has_all_keys_and_no_models():
    gen = await generator.generate_sections(None, _binary())
    assert tuple(gen.section_models) == SECTION_IDS
    assert set(gen.section_models.values()) == {None}


async def test_customer_result_carries_section_models(make_section_client):
    fake = _replying(make_section_client, {"section_7_1": D24_OK})
    gen = await generator.generate_sections(fake, _binary())
    out = jobs.customer_result(gen)
    assert out["section_models"] == gen.section_models
    assert set(out["section_models"]) == set(out["section_sources"]) == set(SECTION_IDS)
    assert NarrativeJobResult(**out).section_models == gen.section_models
