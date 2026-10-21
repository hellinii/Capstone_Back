"""공유 pytest 픽스처.

핵심: 가짜 LLM 클라이언트 팩토리 두 개 — 컬럼 매핑(analyze_columns_with_llm)용 OpenAI 가짜와
서술(generate_narrative)용 Anthropic 가짜. LLM 호출부를 실제 API 키·네트워크 없이
결정론적으로 테스트하기 위한 것이다. (grounding 폴백/타임아웃/조립 실패 경로 검증)
"""
import asyncio
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


# ── Anthropic(서술) 가짜 클라이언트 ─────────────────────────────────────────────
# 서술은 `await client.beta.messages.create(...)` 하나로 Claude 를 부른다(app.narrative.llm).
# 가짜는 **설치된 SDK 의 실제 서명**에 인자를 묶어 본다 — 서명에 없는 인자(temperature·seed 등)나
# 빠진 필수 인자는 실제 SDK 처럼 TypeError 가 난다. 응답은 SDK 의 BetaMessage 로 조립해
# 필드 이름(stop_details·usage.iterations 등)이 실제와 어긋나면 테스트가 깨지게 한다.
import inspect as _inspect

from anthropic import AsyncAnthropic as _AsyncAnthropic
from anthropic.resources.beta.messages import AsyncMessages as _AsyncBetaMessages
from anthropic.types.beta import BetaMessage as _BetaMessage

_BETA_CREATE_SIGNATURE = _inspect.signature(_AsyncBetaMessages.create)
_WITH_OPTIONS_SIGNATURE = _inspect.signature(_AsyncAnthropic.with_options)


def fake_beta_message(
    content=None,
    *,
    blocks=None,
    model="claude-sonnet-5-5",
    stop_reason="end_turn",
    stop_details=None,
    usage=(1000, 500),
    iterations="auto",
):
    """BetaMessage 하나를 만든다.

    - content: dict/list 면 json.dumps 한 text 블록, str 이면 그대로 text 블록, None 이면 블록 없음.
    - blocks: 주면 content 대신 이 블록 목록을 그대로 쓴다(thinking·fallback 블록 섞기).
    - iterations: "auto" 면 응답 model 의 message 한 줄, None·[] 면 비움, 목록이면 그대로.
    """
    if blocks is None:
        if content is None:
            blocks = []
        else:
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            blocks = [{"type": "text", "text": text}]
    if iterations == "auto":
        iterations = [{
            "type": "message", "model": model, "input_tokens": usage[0], "output_tokens": usage[1],
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }]
    return _BetaMessage.model_validate({
        "id": "msg_fake", "type": "message", "role": "assistant", "model": model,
        "content": blocks, "stop_reason": stop_reason, "stop_details": stop_details, "stop_sequence": None,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1], "iterations": iterations},
    })


def _make_fake_anthropic_client(content=None, *, raise_exc=None, responses=None, delay=0.0, **message_kwargs):
    """AsyncAnthropic 유사 가짜. `client.beta.messages.create` 만 있다.

    - content / message_kwargs: fake_beta_message 로 응답 하나를 만든다.
    - responses: 호출마다 차례로 돌려줄 BetaMessage(또는 예외) 목록.
    - raise_exc: 호출 시 이 예외를 던진다.
    - delay: 응답 전에 기다릴 초(서버 마감 테스트).
    """
    client = MagicMock()
    queue = list(responses) if responses is not None else None

    async def _create(*args, **kwargs):
        _BETA_CREATE_SIGNATURE.bind(None, *args, **kwargs)  # 실제 SDK 가 받지 않는 인자면 TypeError
        if delay:
            await asyncio.sleep(delay)
        if raise_exc is not None:
            raise raise_exc
        if queue is not None:
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return fake_beta_message(content, **message_kwargs)

    client.beta.messages.create = AsyncMock(side_effect=_create)

    def _with_options(*args, **kwargs):
        # 작업 번호 경로는 client.with_options(timeout=…, max_retries=0) 로 부른다. 실제 SDK 서명에 묶어 보고
        # 같은 가짜를 돌려준다(호출 기록이 한 곳에 모이게). 넘긴 인자는 with_options.call_args 로 본다.
        _WITH_OPTIONS_SIGNATURE.bind(None, *args, **kwargs)
        return client

    client.with_options = MagicMock(side_effect=_with_options)
    return client


@pytest.fixture
def make_fake_anthropic_client():
    """원하는 응답·예외로 가짜 Anthropic 클라이언트를 만드는 팩토리."""
    return _make_fake_anthropic_client


# ── 작업 경로(장별 호출·칸 다시 만들기)용 가짜 ─────────────────────────────────────
# 작업 경로는 호출마다 칸 스키마(app.narrative.prompt.response_schema)가 다르다 — 7장·8장·9장, 다시 만들 칸 하나.
# 이 가짜는 요청의 스키마를 읽어 **요청한 칸 필드만** 채워 답한다. 기본 글은 숫자가 없어 어느 사실 시트에서도 검문을
# 통과한다(7.3 은 빈 글 = 관찰 없음).
DEFAULT_SECTION_TEXTS = {
    "section_7_1": "혼동행렬에서 오분류가 고르게 나타난다.",
    "section_7_2": "클래스 분포를 해석한 서술이다.",
    "section_7_3": "",
    "section_8_1": "내부 참조 기준과 비교한 서술이다.",
    "section_8_2": "종합 판정의 사유를 적은 서술이다.",
    "section_8_3": "운영상 위험을 적은 서술이다.",
    "section_9_1": "데이터 측면의 권고 서술이다.",
    "section_9_2": "모델 측면의 권고 서술이다.",
}


def requested_fields(kwargs) -> list[str]:
    """호출 인자에서 이번에 요청한 칸 필드 이름(스키마 속성 순서)."""
    return list(kwargs["output_config"]["format"]["schema"]["properties"])


def section_reply(kwargs, texts: dict | None = None) -> dict:
    """요청한 칸 필드만 채운 응답 본문. texts 가 필드를 덮는다(없는 필드는 기본 글, 권고는 문제마다 하나)."""
    schema = kwargs["output_config"]["format"]["schema"]
    texts = texts or {}
    out = {}
    for name, spec in schema["properties"].items():
        if name == "recommendations":
            ids = spec["items"]["properties"]["problem_id"]["enum"]
            out[name] = texts.get(name, [
                {"problem_id": i, "category": "점검", "action": "해당 문제를 점검하고 재평가한다.",
                 "expected_impact": "판정의 근거가 분명해진다."}
                for i in ids
            ])
        else:
            out[name] = texts.get(name, DEFAULT_SECTION_TEXTS[name])
    return out


def _make_section_client(texts: dict | None = None, *, replies=None, delay=0.0, **message_kwargs):
    """칸 스키마를 읽어 답하는 가짜 Anthropic 클라이언트.

    - texts: 모든 호출에 같이 쓰는 칸 글(필드 이름 → 글).
    - replies: 호출마다 차례로 쓸 것 — dict(그 호출의 칸 글을 덮음) · BetaMessage(그대로) · 예외(던짐) ·
      callable(kwargs → 위 셋 중 하나) · None(texts 대로). 다 쓰면 texts 대로.
    - delay: 응답 전에 기다릴 초. message_kwargs 는 fake_beta_message 로(stop_reason·usage 등).
    호출 인자는 client.beta.messages.create.call_args_list 로 본다.
    """
    client = MagicMock()
    queue = list(replies) if replies is not None else []

    async def _create(*args, **kwargs):
        _BETA_CREATE_SIGNATURE.bind(None, *args, **kwargs)
        if delay:
            await asyncio.sleep(delay)
        item = queue.pop(0) if queue else None
        if callable(item) and not isinstance(item, (dict, BaseException)):
            item = item(kwargs)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, _BetaMessage):
            return item
        merged = {**(texts or {}), **(item or {})}
        return fake_beta_message(section_reply(kwargs, merged), **message_kwargs)

    client.beta.messages.create = AsyncMock(side_effect=_create)

    def _with_options(*args, **kwargs):
        _WITH_OPTIONS_SIGNATURE.bind(None, *args, **kwargs)
        return client

    client.with_options = MagicMock(side_effect=_with_options)
    return client


@pytest.fixture
def make_section_client():
    """작업 경로용 — 요청한 칸만 채워 답하는 가짜 Anthropic 클라이언트 팩토리."""
    return _make_section_client


@pytest.fixture(autouse=True)
def _no_real_llm_clients(monkeypatch):
    """모든 테스트에서 실제 LLM 을 부르지 않게 한다.

    앱은 싱글턴이라 다른 테스트가 app.state 에 가짜를 남길 수 있고, 로컬 .env 에 키가 있으면
    lifespan 이 실제 클라이언트를 만든다. 두 키를 지우고 두 클라이언트를 None 으로 시작한다
    (키가 필요한 테스트는 monkeypatch.setenv 로 다시 넣는다).
    """
    from app.main import app  # 처음 import 할 때 load_dotenv() 가 키를 넣으므로 지우기보다 먼저

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    prev = (getattr(app.state, "openai_client", None), getattr(app.state, "anthropic_client", None))
    app.state.openai_client = None
    app.state.anthropic_client = None
    yield
    app.state.openai_client, app.state.anthropic_client = prev


@pytest.fixture(autouse=True)
def _fresh_llm_wallets():
    """시간당 지갑(매핑·서술·서술 작업 생성)은 프로세스 전역이다. 테스트끼리 쓴 양이 쌓여 상한에 닿으면
    무관한 테스트가 규칙 문장으로 강등되므로 테스트마다 비운다."""
    from app.core import llm_budget

    llm_budget.reset()
    yield
    llm_budget.reset()
