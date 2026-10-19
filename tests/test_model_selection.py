"""test_model_selection.py — 서술·매핑 모델 이름을 환경변수로 고르는지(2026-10-23 gpt-4.1-nano 종료 대응).

서술은 Claude(app.narrative.llm, 기본 claude-sonnet-5-5), 컬럼 매핑은 OpenAI(기본 gpt-6-luna).
코드 기본값은 운영값이다. 기본값이 종료된 모델로 돌아가거나, 환경변수를 읽지 않거나,
호출에 넘긴 모델과 응답 meta.model 이 어긋나면 이 파일이 실패해야 한다.
"""
import pathlib

import httpx
import openai
import pandas as pd
import pytest

from app.analysis import llm_mapper
from app.analysis.llm_mapper import analyze_columns_with_llm
from app.core.schemas import TaskType
from app.narrative import llm
from app.narrative.schemas import NarrativeRequest
from app.narrative.service import generate_narrative
from tests.test_narrator import _no_special_fact_sheet

RETIRED_MODEL = "gpt-4.1-nano"

_CLEAN_LLM_JSON = {
    "interpretation": {"confusion_analysis": "정확도 0.94로 양호.", "distribution_analysis": ""},
    "conclusion": {"verdict": "PASS", "benchmark": "", "narrative": "", "risks": ""},
    "recommendation_narrative": {"data_quality": "", "model_ops": ""},
    "recommendations": [],
}


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
    monkeypatch.delenv("NARRATIVE_MODEL", raising=False)
    monkeypatch.delenv("MAPPING_MODEL", raising=False)


def test_default_models_are_not_retired():
    assert llm.DEFAULT_NARRATIVE_MODEL == "claude-sonnet-5-5"
    assert llm_mapper.DEFAULT_MAPPING_MODEL == "gpt-6-luna"
    assert llm.narrative_model() == "claude-sonnet-5-5"
    assert llm_mapper.mapping_model() == "gpt-6-luna"


def test_env_overrides_and_blank_falls_back(monkeypatch):
    monkeypatch.setenv("NARRATIVE_MODEL", " claude-x-narr ")
    monkeypatch.setenv("MAPPING_MODEL", "gpt-x-map")
    assert llm.narrative_model() == "claude-x-narr"
    assert llm_mapper.mapping_model() == "gpt-x-map"
    monkeypatch.setenv("NARRATIVE_MODEL", "  ")
    monkeypatch.setenv("MAPPING_MODEL", "")
    assert llm.narrative_model() == llm.DEFAULT_NARRATIVE_MODEL
    assert llm_mapper.mapping_model() == llm_mapper.DEFAULT_MAPPING_MODEL


@pytest.mark.parametrize("env_value, expected", [(None, "claude-sonnet-5-5"), ("claude-x-narr", "claude-x-narr")])
async def test_narrative_call_uses_selected_model(monkeypatch, make_fake_anthropic_client, env_value, expected):
    if env_value is not None:
        monkeypatch.setenv("NARRATIVE_MODEL", env_value)
    client = make_fake_anthropic_client(_CLEAN_LLM_JSON, model=expected)
    req = NarrativeRequest(task_type=TaskType.binary, report_purpose="external", fact_sheet=_no_special_fact_sheet())
    resp = await generate_narrative(client, req)
    assert client.beta.messages.create.call_count == 1
    assert client.beta.messages.create.call_args.kwargs["model"] == expected
    assert resp.meta.source == "llm"
    assert resp.meta.model == expected  # 응답에 적는 모델 = 실제로 답한 모델


async def test_mapping_calls_use_selected_model_including_retry(monkeypatch, make_fake_openai_client):
    """스키마 거부 재시도(D5a)까지 두 호출 모두 고른 모델을 쓴다."""
    monkeypatch.setenv("MAPPING_MODEL", "gpt-x-map")
    rejected = httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    fake = make_fake_openai_client(
        raise_exc=openai.BadRequestError(message="invalid schema", response=rejected, body=None)
    )
    df = pd.DataFrame({"id": [1], "y_true": [0], "y_pred": [1]})
    with pytest.raises(openai.BadRequestError):
        await analyze_columns_with_llm(fake, TaskType.binary, list(df.columns), df)
    calls = fake.chat.completions.create.call_args_list
    assert [c.kwargs["model"] for c in calls] == ["gpt-x-map", "gpt-x-map"]
    assert ["temperature" in c.kwargs for c in calls] == [False, False]  # gpt-4 계열이 아니면 재시도에도 없다


def test_retired_model_not_hardcoded_in_app():
    app_dir = pathlib.Path(__file__).resolve().parents[1] / "app"
    files = sorted(app_dir.rglob("*.py"))
    assert files, "app/*.py 를 하나도 찾지 못했다"
    hits = [str(p) for p in files if RETIRED_MODEL in p.read_text(encoding="utf-8")]
    assert hits == [], f"종료되는 모델 이름이 코드에 남아 있다: {hits}"


@pytest.mark.parametrize("model, sends_temperature", [
    ("gpt-6-luna", False), ("gpt-5.5", False), ("gpt-4.1-mini", True), ("gpt-4.1", True),
])
async def test_mapping_sends_temperature_only_to_models_that_take_it(
    monkeypatch, make_fake_openai_client, model, sends_temperature
):
    """추론형 모델에 temperature 를 보내면 400 이고, 그것이 스키마 거부 재시도를 엉뚱하게 쓴다."""
    monkeypatch.setenv("MAPPING_MODEL", model)
    fake = make_fake_openai_client({"column_mappings": [
        {"column": "id", "role": "sample_id"}, {"column": "y_true", "role": "y_true"}, {"column": "y_pred", "role": "y_pred"},
    ]})
    df = pd.DataFrame({"id": [1, 2], "y_true": [0, 1], "y_pred": [1, 1]})
    await analyze_columns_with_llm(fake, TaskType.binary, list(df.columns), df)
    kw = fake.chat.completions.create.call_args.kwargs
    assert kw["model"] == model
    assert ("temperature" in kw) is sends_temperature
    if sends_temperature:
        assert kw["temperature"] == 0
