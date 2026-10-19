"""test_model_selection.py — 서술·매핑 모델 이름을 환경변수로 고르는지(2026-10-23 gpt-4.1-nano 종료 대응).

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
from app.narrative import service
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
    assert service.DEFAULT_NARRATIVE_MODEL == "gpt-4.1-mini"
    assert llm_mapper.DEFAULT_MAPPING_MODEL == "gpt-4.1-mini"
    assert service.narrative_model() == "gpt-4.1-mini"
    assert llm_mapper.mapping_model() == "gpt-4.1-mini"


def test_env_overrides_and_blank_falls_back(monkeypatch):
    monkeypatch.setenv("NARRATIVE_MODEL", " gpt-x-narr ")
    monkeypatch.setenv("MAPPING_MODEL", "gpt-x-map")
    assert service.narrative_model() == "gpt-x-narr"
    assert llm_mapper.mapping_model() == "gpt-x-map"
    monkeypatch.setenv("NARRATIVE_MODEL", "  ")
    monkeypatch.setenv("MAPPING_MODEL", "")
    assert service.narrative_model() == service.DEFAULT_NARRATIVE_MODEL
    assert llm_mapper.mapping_model() == llm_mapper.DEFAULT_MAPPING_MODEL


@pytest.mark.parametrize("env_value, expected", [(None, "gpt-4.1-mini"), ("gpt-x-narr", "gpt-x-narr")])
async def test_narrative_call_uses_selected_model(monkeypatch, make_fake_openai_client, env_value, expected):
    if env_value is not None:
        monkeypatch.setenv("NARRATIVE_MODEL", env_value)
    client = make_fake_openai_client(_CLEAN_LLM_JSON)
    req = NarrativeRequest(task_type=TaskType.binary, report_purpose="external", fact_sheet=_no_special_fact_sheet())
    resp = await generate_narrative(client, req)
    assert client.chat.completions.create.call_count == 1
    assert client.chat.completions.create.call_args.kwargs["model"] == expected
    assert resp.meta.source == "llm"
    assert resp.meta.model == expected  # 응답에 적는 모델 = 실제로 부른 모델


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
    models = [c.kwargs["model"] for c in fake.chat.completions.create.call_args_list]
    assert models == ["gpt-x-map", "gpt-x-map"]


def test_retired_model_not_hardcoded_in_app():
    app_dir = pathlib.Path(__file__).resolve().parents[1] / "app"
    files = sorted(app_dir.rglob("*.py"))
    assert files, "app/*.py 를 하나도 찾지 못했다"
    hits = [str(p) for p in files if RETIRED_MODEL in p.read_text(encoding="utf-8")]
    assert hits == [], f"종료되는 모델 이름이 코드에 남아 있다: {hits}"
