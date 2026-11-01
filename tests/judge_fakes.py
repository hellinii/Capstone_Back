"""tests/judge_fakes.py — 심판 시험용 가짜 OpenAI 클라이언트(시험 파일이 아니라 도우미).

- `responses.create` 의 인자를 지금 SDK 의 `AsyncResponses.create` 서명에 묶는다(모르는 인자 이름이면 TypeError — SDK 를
  바꾸거나 칸 이름을 틀리면 시험이 깨진다).
- 응답은 지금 SDK 의 `Response` 모델로 조립한다(`Response.model_validate`) — 응답 읽기 코드가 진짜 칸 이름을 쓰는지 본다.
- 정해 둔 답(응답·예외·함수)을 차례로 돌려주고, 받은 인자와 `with_options` 인자를 모두 적어 둔다.
"""
from __future__ import annotations

import inspect
import itertools
import json
from typing import Any, Callable

import httpx
import openai
from openai.resources.responses import AsyncResponses
from openai.types.responses import Response

_SIGNATURE = inspect.signature(AsyncResponses.create)
_ids = itertools.count(1)
_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")


def usage(input_tokens=1000, cached=0, output_tokens=200, reasoning=50, cache_write=0) -> dict:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": cache_write},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input_tokens + output_tokens,
    }


def message_item(text: str, *, refusal: bool = False) -> dict:
    part = {"type": "refusal", "refusal": text} if refusal else {"type": "output_text", "text": text, "annotations": []}
    return {"id": f"msg_{next(_ids)}", "type": "message", "role": "assistant", "status": "completed", "content": [part]}


def reasoning_item(encrypted: str = "enc-abc") -> dict:
    return {"id": f"rs_{next(_ids)}", "type": "reasoning", "summary": [], "encrypted_content": encrypted}


def function_call_item(name: str, arguments: dict, call_id: str | None = None) -> dict:
    return {"id": f"fc_{next(_ids)}", "type": "function_call", "name": name, "arguments": json.dumps(arguments),
            "call_id": call_id or f"call_{next(_ids)}", "status": "completed"}


def response(output: list[dict], *, model="gpt-6.1-sol", status="completed", use: dict | None = None) -> Response:
    return Response.model_validate({
        "id": f"resp_{next(_ids)}", "object": "response", "created_at": 1, "model": model, "status": status,
        "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": use if use is not None else usage(),
    })


def json_response(data: Any, **kw) -> Response:
    return response([reasoning_item(), message_item(json.dumps(data, ensure_ascii=False))], **kw)


def rate_limit() -> openai.RateLimitError:
    return openai.RateLimitError("rate limited", response=httpx.Response(429, request=_REQUEST), body=None)


def server_error(status=503) -> openai.InternalServerError:
    return openai.InternalServerError("server", response=httpx.Response(status, request=_REQUEST), body=None)


def bad_request() -> openai.BadRequestError:
    return openai.BadRequestError("bad", response=httpx.Response(400, request=_REQUEST), body=None)


def timeout() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=_REQUEST)


def connection() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQUEST)


class FakeOpenAI:
    """`client.with_options(...).responses.create(...)` 모양만 흉내 낸다."""

    def __init__(self, replies: list[Response | Exception | Callable[[dict], Any]] | None = None):
        self.replies = list(replies or [])
        self.calls: list[dict] = []
        self.options: list[dict] = []
        self.responses = self

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    async def create(self, **kwargs):
        _SIGNATURE.bind(None, **kwargs)  # 지금 SDK 서명에 없는 인자면 TypeError
        self.calls.append(kwargs)
        if not self.replies:
            raise AssertionError("가짜 OpenAI: 준비한 답보다 많이 불렸다")
        reply = self.replies.pop(0)
        if callable(reply) and not isinstance(reply, Response):
            reply = reply(kwargs)
        if isinstance(reply, Exception):
            raise reply
        return reply
