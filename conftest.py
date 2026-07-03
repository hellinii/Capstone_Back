"""공유 pytest 픽스처.

핵심: async OpenAI 클라이언트 목(mock) 팩토리. narrator.generate_narrative /
analyzer.analyze_columns_with_llm 등 LLM 호출부를 실제 API 키·네트워크 없이
결정론적으로 테스트하기 위한 것이다. (grounding 폴백/타임아웃/조립 실패 경로 검증)
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_fake_openai_client(content=None, *, raise_exc=None):
    """AsyncOpenAI 유사 목 클라이언트를 만든다.

    - content: chat.completions.create 가 돌려줄 message.content.
        dict/list 면 json.dumps 로 직렬화하고, str 이면 그대로 둔다
        (비-JSON 문자열을 주면 json.loads 실패 경로를 태울 수 있다).
    - raise_exc: 주어지면 create 호출 시 해당 예외를 raise 한다
        (APITimeoutError 등 LLM 호출 실패 경로 검증용).

    실제 SDK 호출 형태 `await client.chat.completions.create(...)` 및
    `response.choices[0].message.content` 접근을 그대로 모사한다.
    """
    client = MagicMock()

    async def _create(*_args, **_kwargs):
        if raise_exc is not None:
            raise raise_exc
        message = MagicMock()
        message.content = content if isinstance(content, str) else json.dumps(content)
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    client.chat.completions.create = AsyncMock(side_effect=_create)
    return client


@pytest.fixture
def make_fake_openai_client():
    """원하는 응답 JSON 또는 예외로 목 OpenAI 클라이언트를 생성하는 팩토리."""
    return _make_fake_openai_client
