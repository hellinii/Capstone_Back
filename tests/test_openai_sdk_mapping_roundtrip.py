"""tests/test_openai_sdk_mapping_roundtrip.py — 컬럼 매핑이 **진짜 OpenAI SDK** 를 지나 왕복하는지(심판 라운드 J1 ①).

심판 라운드가 openai SDK 를 1.75.0 → 3.30.0 으로 올렸다(응답 보관을 끈 Responses 도구 반복에 `reasoning.encrypted_content`
가 필요한데 1.75.0 에는 없다). 다른 컬럼 매핑 시험은 MagicMock 클라이언트라 SDK 가 바뀌어도 모른다. 여기서는 진짜
`AsyncOpenAI` 에 httpx 가짜 전송을 끼워, 요청 본문(경로·모델·메시지·구조화 출력)과 응답 읽기, 스키마 거부 때 enum 없는 스키마로
한 번만 다시 부르는 갈래(D5a)를 SDK 를 통과시켜 본다. 네트워크는 쓰지 않는다.
"""
from __future__ import annotations

import json

import httpx
import openai
import pandas as pd
import pytest
from openai import AsyncOpenAI

from app.analysis.llm_mapper import analyze_columns_with_llm
from app.core.schemas import TaskType

COLUMNS = ["id", "y_true", "y_pred", "score_positive"]
DF = pd.DataFrame({"id": ["S1", "S2"], "y_true": [1, 0], "y_pred": [1, 1], "score_positive": [0.9, 0.6]})
ANSWER = {"column_mappings": [
    {"column": "id", "role": "sample_id"}, {"column": "y_true", "role": "y_true"},
    {"column": "y_pred", "role": "y_pred"}, {"column": "score_positive", "role": "score_positive"},
]}


def _completion(content: str) -> dict:
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "gpt-test",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _client(handler) -> AsyncOpenAI:
    return AsyncOpenAI(api_key="sk-test", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_mapping_request_and_response_pass_through_the_sdk(monkeypatch):
    monkeypatch.setenv("MAPPING_MODEL", "gpt-test")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "body": json.loads(request.content)})
        return httpx.Response(200, json=_completion(json.dumps(ANSWER)))

    result = await analyze_columns_with_llm(_client(handler), TaskType.binary, COLUMNS, DF)
    assert len(seen) == 1 and seen[0]["path"] == "/v1/chat/completions"
    body = seen[0]["body"]
    assert body["model"] == "gpt-test"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    column_prop = fmt["json_schema"]["schema"]["properties"]["column_mappings"]["items"]["properties"]["column"]
    assert column_prop.get("enum") == COLUMNS  # 동적 enum 이 SDK 를 지나 그대로 간다
    assert "temperature" not in body  # gpt-4 계열이 아니면 보내지 않는다
    assert [(m.column, m.role.value) for m in result.column_mappings] == [(a["column"], a["role"]) for a in ANSWER["column_mappings"]]


async def test_schema_rejection_retries_once_without_enum_through_the_sdk(monkeypatch):
    monkeypatch.setenv("MAPPING_MODEL", "gpt-test")
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(400, json={"error": {"message": "bad schema", "type": "invalid_request_error"}})
        return httpx.Response(200, json=_completion(json.dumps(ANSWER)))

    await analyze_columns_with_llm(_client(handler), TaskType.binary, COLUMNS, DF)
    assert len(bodies) == 2
    first, second = (b["response_format"]["json_schema"]["schema"]["properties"]["column_mappings"]["items"]["properties"]["column"]
                     for b in bodies)
    assert "enum" in first and "enum" not in second


async def test_timeout_is_not_retried_by_the_mapper(monkeypatch):
    monkeypatch.setenv("MAPPING_MODEL", "gpt-test")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(openai.APITimeoutError):
        await analyze_columns_with_llm(_client(handler), TaskType.binary, COLUMNS, DF)
    assert len(calls) == 1  # max_retries=0 클라이언트 + 매퍼는 스키마 거부만 다시 부른다
