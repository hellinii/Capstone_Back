"""app/judge/llm.py — 심판의 OpenAI 호출 한 곳(외부 경계).

심판 4모듈과 공용 전처리의 모든 호출은 `create_response` 하나를 지난다. 요청 모양·재시도·시간·응답 읽기·호출 기록을
여기서만 정한다. 판정은 각 모듈이 한다.

요청 규칙(심판 설계 4절 "모델 설정", 계획 사-5)
- Responses API(사실 일치 에이전트가 도구를 부르므로 — GPT-6.1 Sol 은 채팅 완성에서 도구를 부를 수 없다). 다른 모듈도 같은 길.
- 응답 보관을 끈다(`store=False`) — 서술 글·사실 값이 OpenAI 쪽에 남지 않게. 대신 도구를 여러 번 부르는 대화에서 앞 추론을
  이어 주려고 `include=["reasoning.encrypted_content"]` 를 받아 다음 턴 입력에 그대로 되돌린다(`next_turn_items`).
- 추론 강도는 4모듈·전처리가 같은 값(기본 medium). 온도·seed 는 보내지 않는다.
- SDK 재시도는 끈다(`with_options(max_retries=0)`). 일시적 오류(429·5xx·연결 끊김)만 남은 시간 안에서 **한 번** 더 부른다
  (HTTP 시도는 최대 2번). 호출 제한 시간 초과는 다시 부르지 않는다 — 시간을 이미 다 썼다.
- 예외를 던지지 않는다. 실패는 종류 문자열(`FAILURES`)로 돌려주고, 발급 보류 여부는 부르는 쪽이 정한다.

호출 기록(`CallRecord`)은 내부 기록·서버 로그에만 남는다. 입력(서술·사실 값)은 남기지 않는다. 비용은 응답 모델의 단가로
(입력 − 캐시 입력)·캐시 입력·출력을 따로 곱해 센다(캐시 쓰기 토큰은 입력 단가 — 따로 정한 값을 받지 못했다). 단가를 모르면 None(추정하지 않는다).

상호작용
- 의존(import): openai(예외 타입), app.judge.config
- 사용처: app.judge 의 전처리·4모듈, scripts/judge_live.py
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import openai

from app.judge import config

logger = logging.getLogger(__name__)

FAILURES = ("no_client", "deadline", "timeout", "transient", "api_error", "refusal", "incomplete", "bad_output", "error")

# 일시적 오류 뒤 한 번 더 부를지 정하는 최소 남은 시간(초). 이보다 적으면 다시 부르지 않고 실패로 돌려준다.
MIN_RETRY_SECONDS = 15.0


@dataclass
class CallRecord:
    """호출 하나(재시도 포함)의 내부 기록."""

    purpose: str
    requested_model: str
    effort: str
    response_model: str | None = None
    status: str | None = None
    attempts: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    tool_calls: int = 0
    cost_usd: float | None = None
    seconds: float = 0.0
    failure: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Outcome:
    """호출 결과. response 가 있으면 성공, 없으면 failure 가 사유다."""

    response: Any | None
    failure: str | None
    record: CallRecord


def cost_usd(model: str | None, input_tokens: int, cached: int, output_tokens: int) -> float | None:
    price = config.price_of(model)
    if price is None:
        return None
    p_in, p_cached, p_out = price
    return ((input_tokens - cached) * p_in + cached * p_cached + output_tokens * p_out) / 1_000_000


def _fill_usage(record: CallRecord, response) -> None:
    usage = getattr(response, "usage", None)
    record.response_model = getattr(response, "model", None)
    record.status = getattr(response, "status", None)
    record.tool_calls = sum(1 for item in (response.output or []) if getattr(item, "type", None) == "function_call")
    if usage is None:
        return
    record.input_tokens = usage.input_tokens or 0
    details = getattr(usage, "input_tokens_details", None)
    record.cached_input_tokens = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    record.cache_write_tokens = (getattr(details, "cache_write_tokens", 0) or 0) if details else 0
    record.output_tokens = usage.output_tokens or 0
    out_details = getattr(usage, "output_tokens_details", None)
    record.reasoning_tokens = (getattr(out_details, "reasoning_tokens", 0) or 0) if out_details else 0
    record.cost_usd = cost_usd(record.response_model, record.input_tokens, record.cached_input_tokens, record.output_tokens)


def _has_refusal(response) -> bool:
    for item in response.output or []:
        if getattr(item, "type", None) == "message":
            for part in item.content or []:
                if getattr(part, "type", None) == "refusal":
                    return True
    return False


def _classify(exc: Exception) -> str:
    if isinstance(exc, openai.APITimeoutError):
        return "timeout"
    if isinstance(exc, (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)):
        return "transient"
    if isinstance(exc, openai.APIStatusError):
        return "transient" if exc.status_code >= 500 else "api_error"
    return "error"


async def create_response(
    client,
    *,
    purpose: str,
    model: str,
    instructions: str,
    input: list | str,
    timeout_s: float,
    deadline: float | None = None,
    tools: list | None = None,
    text_format: dict | None = None,
    effort: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Outcome:
    """Responses API 한 번(일시적 오류면 남은 시간 안에서 한 번 더). deadline 은 clock 기준 절대 시각."""
    effort = effort or config.judge_effort()
    record = CallRecord(purpose=purpose, requested_model=model, effort=effort)
    if client is None:
        record.failure = "no_client"
        return Outcome(None, "no_client", record)
    started = clock()
    params: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input,
        "reasoning": {"effort": effort},
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
    }
    if tools:
        params["tools"] = tools
    if text_format:
        params["text"] = {"format": text_format}

    failure: str | None = None
    response = None
    for attempt in (1, 2):
        remaining = None if deadline is None else deadline - clock()
        if remaining is not None and remaining <= 0:
            failure = "deadline"
            break
        call_timeout = timeout_s if remaining is None else min(timeout_s, remaining)
        record.attempts = attempt
        try:
            response = await client.with_options(max_retries=0, timeout=call_timeout).responses.create(**params)
            failure = None
            break
        except Exception as exc:  # noqa: BLE001 — 종류로 돌려준다
            failure = _classify(exc)
            record.error = f"{type(exc).__name__}: {str(exc)[:200]}"
            left = None if deadline is None else deadline - clock()
            if failure != "transient" or attempt == 2 or (left is not None and left < MIN_RETRY_SECONDS):
                break
    record.seconds = round(clock() - started, 3)

    if response is not None:
        _fill_usage(record, response)
        if _has_refusal(response):
            failure = "refusal"
        elif getattr(response, "status", None) == "incomplete":
            failure = "incomplete"
    record.failure = failure
    logger.info("심판 호출 %s", json.dumps({k: v for k, v in record.as_dict().items() if k != "error"}, ensure_ascii=False))
    return Outcome(response if failure is None else None, failure, record)


def output_text(response) -> str:
    """응답의 assistant 메시지 글(output_text 조각을 이은 것)."""
    parts: list[str] = []
    for item in response.output or []:
        if getattr(item, "type", None) == "message":
            parts.extend(p.text for p in item.content or [] if getattr(p, "type", None) == "output_text")
    return "".join(parts)


def parse_json(outcome: Outcome) -> tuple[dict | None, str | None]:
    """구조화 출력 응답을 dict 로. 실패면 (None, 사유). JSON 이 아니거나 객체가 아니면 bad_output."""
    if outcome.response is None:
        return None, outcome.failure
    try:
        data = json.loads(output_text(outcome.response))
    except ValueError:
        outcome.record.failure = "bad_output"
        return None, "bad_output"
    if not isinstance(data, dict):
        outcome.record.failure = "bad_output"
        return None, "bad_output"
    return data, None


def json_schema_format(name: str, schema: dict) -> dict:
    """Responses 구조화 출력 형식(strict)."""
    return {"type": "json_schema", "name": name, "schema": schema, "strict": True}


def next_turn_items(response) -> list[dict]:
    """다음 턴 입력에 되돌릴 이번 응답의 출력 항목(암호화 추론 포함). 응답 보관을 껐으므로 직접 이어 준다."""
    return [item.model_dump(exclude_none=True, by_alias=True) for item in response.output or []]


def function_calls(response) -> list:
    return [item for item in response.output or [] if getattr(item, "type", None) == "function_call"]
