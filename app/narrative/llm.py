"""app/narrative/llm.py — 서술 생성의 Claude 호출 한 곳(외부 경계).

모든 서술 호출은 `call_narrative_llm` 하나를 지난다. 요청 모양(모델·대체 모델·구조화 출력·effort)과
응답 읽기(거절·잘림을 먼저 보고, text 블록만 JSON 으로 읽기)와 호출 기록(응답 모델·거절 종류·토큰·비용)을
여기서만 정한다. 서술 문장의 검증·조립은 service 가 한다.

요청 규칙(서술 라운드 계획 9번)
- `client.beta.messages.create` 에 `betas=[server-side-fallback-2026-07-01]` + `fallbacks="default"` 를
  늘 보낸다. 거절 종류별로 대체 모델이 정해지고(cyber·frontier_llm 은 대체가 있고 bio·reasoning_extraction·
  general_harms 는 없음) 대체가 없거나 대체도 거절하면 거절이 그대로 온다.
- temperature·top_p·top_k·seed 를 보내지 않는다(Sonnet 5.5 는 기본값이 아닌 값을 400 으로 거절한다).
- 스트리밍을 쓰지 않는다. 비스트리밍이면 중간에 거절돼도 부분 출력 없이 대체 모델의 응답 하나로 온다.
- 응답 스키마에 추론 과정·근거 칸을 두지 않는다(reasoning_extraction 거절은 대체 모델이 없다).

호출 기록은 서버 로그에만 남긴다(입력 — 사실 시트·고객 입력 — 은 남기지 않는다).
비용은 `usage.iterations` 의 항목마다 그 항목 model 의 단가로 센다. 맨 위 usage 는 마지막 시도만 담고,
대체가 일어나면 모델마다 단가가 다르기 때문이다. iterations 가 비면 맨 위 usage 와 응답 model 로 센다.

상호작용
- 의존(import): anthropic(예외 타입), json, logging, os, time
- 사용처: app.narrative.generator(작업 경로), app.narrative.legacy(옛 경로)
"""
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field

import anthropic

logger = logging.getLogger(__name__)

# 코드 기본값 = 운영값. 환경변수는 호출마다 읽는다(배포 뒤 모델만 바꿀 때 코드를 고치지 않게).
DEFAULT_NARRATIVE_MODEL = "claude-sonnet-5-5"
DEFAULT_NARRATIVE_EFFORT = "low"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACKS = "default"

# 생각을 포함한 출력 상한. 비스트리밍 요청이 SDK HTTP 시간 제한 안에 드는 크기다.
MAX_TOKENS = 16000

# 클라이언트 설정(app.main 이 쓴다). 옛 경로는 `호출 시간 제한 × (재시도 + 1)` 이 서버 마감 안에 들어야 한다
# — 서버 마감은 service.NARRATIVE_DEADLINE_SECONDS, 테스트가 이 부등식을 고정한다.
CALL_TIMEOUT_SECONDS = 55.0
MAX_RETRIES = 1

# 100만 토큰당 달러(입력, 출력). 대체 모델 비용도 세려고 대체될 수 있는 모델까지 둔다. 없는 모델은 비용 None.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
}


def narrative_model() -> str:
    """서술 생성 모델(`NARRATIVE_MODEL`, 비었으면 코드 기본값)."""
    return os.getenv("NARRATIVE_MODEL", "").strip() or DEFAULT_NARRATIVE_MODEL


def narrative_effort() -> str:
    """서술 effort(`NARRATIVE_EFFORT`). 비었거나 모르는 값이면 코드 기본값 — 오타가 400 이 되지 않게."""
    value = os.getenv("NARRATIVE_EFFORT", "").strip().lower()
    return value if value in EFFORT_LEVELS else DEFAULT_NARRATIVE_EFFORT


@dataclass
class CallRecord:
    """호출 하나의 내부 기록(고객에게 가지 않는다)."""

    requested_model: str
    effort: str
    fallbacks_sent: bool = True
    response_model: str | None = None
    stop_reason: str | None = None
    refusal_category: str | None = None
    recommended_model: str | None = None
    served_by_fallback: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    seconds: float = 0.0
    error: str | None = None
    iterations: list[dict] = field(default_factory=list)


@dataclass
class LLMResult:
    """호출 결과. data 가 있으면 성공, 없으면 failure 가 강등 사유다."""

    data: dict | None
    failure: str | None
    record: CallRecord


def _iteration_entries(response, requested_model: str) -> list[dict]:
    """usage.iterations 를 (종류, 모델, 입력, 출력) 목록으로. 비면 맨 위 usage 한 줄.

    SDK 에서 message 항목의 model 은 선택 칸이다(fallback_message 는 필수). 비어 있으면 요청한 모델이 처리한
    시도이므로 요청 모델로 채운다 — 그러지 않으면 평범한 성공 호출의 비용이 빈칸이 된다(독립 검토 BE-LLM-01).
    """
    usage = response.usage
    entries = []
    for it in (getattr(usage, "iterations", None) or []):
        model = getattr(it, "model", None)
        if model is None and it.type == "message":
            model = requested_model
        entries.append({
            "type": it.type,
            "model": model,
            "input_tokens": it.input_tokens or 0,
            "output_tokens": it.output_tokens or 0,
        })
    if not entries:
        entries.append({
            "type": "message",
            "model": response.model,
            "input_tokens": usage.input_tokens or 0,
            "output_tokens": usage.output_tokens or 0,
        })
    return entries


def cost_usd(entries: list[dict]) -> float | None:
    """항목마다 그 모델 단가 × 토큰을 더한다. 단가를 모르는 모델이 하나라도 있으면 None(추정하지 않는다)."""
    total = 0.0
    for e in entries:
        price = PRICES_PER_MTOK.get(e["model"] or "")
        if price is None:
            return None
        total += e["input_tokens"] * price[0] / 1_000_000 + e["output_tokens"] * price[1] / 1_000_000
    return total


def _fill_from_response(record: CallRecord, response) -> None:
    record.response_model = response.model
    record.stop_reason = response.stop_reason
    details = response.stop_details
    if details is not None:
        record.refusal_category = getattr(details, "category", None)
        record.recommended_model = getattr(details, "recommended_model", None)
    entries = _iteration_entries(response, record.requested_model)
    record.iterations = entries
    record.served_by_fallback = any(e["type"] == "fallback_message" for e in entries)
    record.input_tokens = sum(e["input_tokens"] for e in entries)
    record.output_tokens = sum(e["output_tokens"] for e in entries)
    record.cost_usd = cost_usd(entries)


def _log(record: CallRecord, failure: str | None) -> None:
    level = logging.INFO if failure is None else logging.WARNING
    logger.log(level, "서술 LLM 호출 기록 %s", json.dumps({"failure": failure, **asdict(record)}, ensure_ascii=False))


async def call_narrative_llm(client, *, system: str, user: str, schema: dict, effort: str | None = None) -> LLMResult:
    """서술 JSON 을 한 번 요청한다. 예외를 던지지 않고 실패는 LLMResult.failure 로 돌려준다.

    failure: refusal(대체도 거절·대체 없음·대체가 한도에 걸림 — 세 경우 모두) · max_tokens(잘림) ·
    model_unavailable(모델 없음) · rate_limited(429) · timeout · api_error · invalid_json.
    """
    model = narrative_model()
    record = CallRecord(requested_model=model, effort=effort or narrative_effort())
    started = time.perf_counter()
    try:
        response = await client.beta.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            betas=[FALLBACK_BETA],
            fallbacks=FALLBACKS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                "effort": record.effort,
            },
        )
    except anthropic.APITimeoutError as exc:
        return _finish(record, started, "timeout", error=repr(exc))
    except anthropic.NotFoundError as exc:
        return _finish(record, started, "model_unavailable", error=repr(exc))
    except anthropic.RateLimitError as exc:
        # 사용 한도(429). 작업 경로는 장별 동시 호출을 차례 호출로 바꿔 한 번 더 부른다(app.narrative.generator).
        return _finish(record, started, "rate_limited", error=repr(exc))
    except anthropic.APIError as exc:
        return _finish(record, started, "api_error", error=repr(exc))

    _fill_from_response(record, response)
    # 거절·잘림을 본문보다 먼저 본다 — 거절이면 content 가 비었거나(출력 전) 쓸 수 없다.
    if response.stop_reason == "refusal":
        return _finish(record, started, "refusal")
    if response.stop_reason == "max_tokens":
        return _finish(record, started, "max_tokens")

    # thinking·fallback 블록은 건너뛰고 text 블록만 읽는다.
    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return _finish(record, started, "invalid_json", error=repr(exc))
    if not isinstance(data, dict):
        return _finish(record, started, "invalid_json", error=f"JSON 최상위가 객체가 아님: {type(data).__name__}")
    record.seconds = round(time.perf_counter() - started, 3)
    _log(record, None)
    return LLMResult(data=data, failure=None, record=record)


def _finish(record: CallRecord, started: float, failure: str, *, error: str | None = None) -> LLMResult:
    record.seconds = round(time.perf_counter() - started, 3)
    record.error = error
    _log(record, failure)
    return LLMResult(data=None, failure=failure, record=record)
