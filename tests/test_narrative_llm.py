"""test_narrative_llm.py — 서술 Claude 호출 층(app.narrative.llm)과 옛 경로 서버 마감(service).

가짜 클라이언트(conftest.make_fake_anthropic_client)는 설치된 SDK 의 실제 서명에 인자를 묶어 본다.
그래서 서명에 없는 인자(temperature·seed 등)를 보내면 이 파일의 정상 경로 테스트가 TypeError 로 깨진다.
"""
import contextlib
import logging

import anthropic
import httpx2
import pytest

from app.core import concurrency, llm_budget
from app.core.schemas import TaskType
from app.narrative import llm
from app.narrative import legacy as service
from app.narrative.schemas import NarrativeRequest
from app.narrative.legacy import generate_narrative
from tests.conftest import fake_beta_message
from tests.test_narrator import _no_special_fact_sheet

_CLEAN = {
    "interpretation": {"confusion_analysis": "정확도 0.94로 양호.", "distribution_analysis": ""},
    "conclusion": {"verdict": "PASS", "benchmark": "", "narrative": "", "risks": ""},
    "recommendation_narrative": {"data_quality": "", "model_ops": ""},
}
_REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("NARRATIVE_MODEL", "NARRATIVE_EFFORT"):
        monkeypatch.delenv(key, raising=False)
    llm_budget.reset()
    yield
    llm_budget.reset()


def _request() -> NarrativeRequest:
    return NarrativeRequest(task_type=TaskType.binary, report_purpose="external", fact_sheet=_no_special_fact_sheet())


def _it(kind, model, inp, out):
    return {"type": kind, "model": model, "input_tokens": inp, "output_tokens": out,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ── 요청 모양 ─────────────────────────────────────────────────────────────────

async def test_request_shape(make_fake_anthropic_client):
    client = make_fake_anthropic_client(_CLEAN)
    resp = await generate_narrative(client, _request())
    assert resp.meta.source == "llm", resp.meta
    kw = client.beta.messages.create.call_args.kwargs
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["max_tokens"] == 16000
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["output_config"]["format"]["schema"]["type"] == "object"
    assert kw["output_config"]["effort"] == "low"
    assert isinstance(kw["system"], str) and kw["system"].strip()
    assert [m["role"] for m in kw["messages"]] == ["user"] and kw["messages"][0]["content"].strip()
    # 샘플링·재현 인자와 스트리밍·생각 설정을 보내지 않는다(Sonnet 5.5 는 비기본 sampling 을 400).
    for banned in ("temperature", "top_p", "top_k", "seed", "stream", "thinking", "response_format"):
        assert banned not in kw, banned


@pytest.mark.parametrize("value, expected", [("high", "high"), (" MEDIUM ", "medium"), ("xhigh", "xhigh"), ("max", "max"), ("bogus", "low"), ("", "low")])
async def test_effort_from_env(monkeypatch, make_fake_anthropic_client, value, expected):
    monkeypatch.setenv("NARRATIVE_EFFORT", value)
    client = make_fake_anthropic_client(_CLEAN)
    await generate_narrative(client, _request())
    assert client.beta.messages.create.call_args.kwargs["output_config"]["effort"] == expected


def test_fake_client_rejects_args_the_sdk_does_not_take(make_fake_anthropic_client):
    """가짜가 실제 서명을 지키는지 — 이것이 무너지면 위 '보내지 않는다' 단정만 남는다."""
    import asyncio

    client = make_fake_anthropic_client(_CLEAN)
    with pytest.raises(TypeError):
        asyncio.run(client.beta.messages.create(model="m", max_tokens=1, messages=[], temperature=0))


# ── 거절 세 경우 · 잘림 ───────────────────────────────────────────────────────

_REFUSALS = {
    # 대체 모델도 거절: 원래 모델이 출력 전에 거절 → 대체(Sonnet 5)가 돌았고 그것도 거절.
    "fallback_also_refused": dict(
        model="claude-sonnet-5", stop_details={"type": "refusal", "category": "frontier_llm"},
        iterations=[_it("message", "claude-sonnet-5-5", 900, 0), _it("fallback_message", "claude-sonnet-5", 900, 0)],
    ),
    # 그 거절 종류에는 추천 대체가 없음(bio) — 원래 모델의 거절이 그대로 온다.
    "no_fallback_for_category": dict(
        model="claude-sonnet-5-5", stop_details={"type": "refusal", "category": "bio"},
        iterations=[_it("message", "claude-sonnet-5-5", 900, 0)],
    ),
    # 대체 모델이 사용 한도에 걸려 대체 없이 거절 + recommended_model 힌트.
    "fallback_rate_limited": dict(
        model="claude-sonnet-5-5",
        stop_details={"type": "refusal", "category": "cyber", "recommended_model": "claude-sonnet-5"},
        iterations=[_it("message", "claude-sonnet-5-5", 900, 0)],
    ),
}


@pytest.mark.parametrize("case", list(_REFUSALS))
async def test_refusal_cases_fall_back_without_retry(make_fake_anthropic_client, caplog, case):
    kw = _REFUSALS[case]
    client = make_fake_anthropic_client(None, stop_reason="refusal", **kw)
    with caplog.at_level(logging.WARNING):
        resp = await generate_narrative(client, _request())
    assert resp.meta.source == "fallback" and resp.meta.reason == "refusal"
    # 거절된 요청을 같은 요청으로 다시 부르지 않는다(대체 모델로 약 1시간 고정 라우팅).
    assert client.beta.messages.create.call_count == 1
    log = " ".join(r.getMessage() for r in caplog.records)
    assert kw["stop_details"]["category"] in log
    assert f'"response_model": "{kw["model"]}"' in log
    if "recommended_model" in kw["stop_details"]:
        assert '"recommended_model": "claude-sonnet-5"' in log


async def test_max_tokens_falls_back_without_retry(make_fake_anthropic_client):
    client = make_fake_anthropic_client('{"interpretation": {"confusion_', stop_reason="max_tokens")
    resp = await generate_narrative(client, _request())
    assert resp.meta.reason == "max_tokens"
    assert client.beta.messages.create.call_count == 1


async def test_refusal_is_checked_before_content():
    """거절인데 text 가 붙어 와도(부분 출력) 읽지 않는다."""
    import json

    class _C:
        class beta:
            class messages:
                @staticmethod
                async def create(**_):
                    return fake_beta_message(json.dumps(_CLEAN), stop_reason="refusal",
                                             stop_details={"type": "refusal", "category": None})

    result = await llm.call_narrative_llm(_C(), system="s", user="u", schema={"type": "object"})
    assert result.failure == "refusal" and result.data is None


# ── 블록 읽기 · 대체 모델이 답한 경우 ────────────────────────────────────────

async def test_thinking_and_fallback_blocks_are_skipped_and_fallback_model_recorded(make_fake_anthropic_client):
    import json

    blocks = [
        {"type": "fallback", "from": {"model": "claude-sonnet-5-5"}, "to": {"model": "claude-sonnet-5"},
         "trigger": {"type": "refusal", "category": "frontier_llm"}},
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": json.dumps(_CLEAN, ensure_ascii=False)},
    ]
    iterations = [_it("message", "claude-sonnet-5-5", 900, 0), _it("fallback_message", "claude-sonnet-5", 950, 400)]
    client = make_fake_anthropic_client(blocks=blocks, model="claude-sonnet-5", iterations=iterations)
    resp = await generate_narrative(client, _request())
    assert resp.meta.source == "llm", resp.meta
    assert resp.meta.model == "claude-sonnet-5"  # 실제로 답한 모델


# ── 비용(usage.iterations) ───────────────────────────────────────────────────

def _client_returning(message):
    class _C:
        class beta:
            class messages:
                @staticmethod
                async def create(**_):
                    return message

    return _C()


async def test_cost_sums_each_iteration_at_its_model_price():
    import json

    msg = fake_beta_message(
        json.dumps(_CLEAN), model="claude-sonnet-5", usage=(950, 400),
        iterations=[_it("message", "claude-sonnet-5-5", 1000, 0), _it("fallback_message", "claude-sonnet-5", 2000, 500)],
    )
    result = await llm.call_narrative_llm(_client_returning(msg), system="s", user="u", schema={"type": "object"})
    r = result.record
    assert result.failure is None
    assert r.served_by_fallback is True
    assert (r.input_tokens, r.output_tokens) == (3000, 500)  # 맨 위 usage(950/400)가 아니라 항목 합
    assert r.cost_usd == pytest.approx(1000 * 2 / 1e6 + 2000 * 2 / 1e6 + 500 * 10 / 1e6)


@pytest.mark.parametrize("iterations", [None, []])
async def test_cost_falls_back_to_top_level_usage_when_iterations_empty(iterations):
    import json

    msg = fake_beta_message(json.dumps(_CLEAN), usage=(1200, 300), iterations=iterations)
    result = await llm.call_narrative_llm(_client_returning(msg), system="s", user="u", schema={"type": "object"})
    r = result.record
    assert (r.input_tokens, r.output_tokens, r.served_by_fallback) == (1200, 300, False)
    assert r.cost_usd == pytest.approx(1200 * 2 / 1e6 + 300 * 10 / 1e6)


def test_cost_unknown_model_is_none_not_guessed():
    entries = [{"type": "message", "model": "claude-unknown", "input_tokens": 10, "output_tokens": 10}]
    assert llm.cost_usd(entries) is None


# ── SDK 오류 · JSON 오류 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("exc, reason", [
    (anthropic.APITimeoutError(request=_REQ), "timeout"),
    (anthropic.NotFoundError("no model", response=httpx2.Response(404, request=_REQ), body=None), "model_unavailable"),
    (anthropic.InternalServerError("down", response=httpx2.Response(500, request=_REQ), body=None), "api_error"),
    (anthropic.RateLimitError("slow", response=httpx2.Response(429, request=_REQ), body=None), "rate_limited"),  # 작업 경로는 429 를 차례 호출로 다시 부른다(S5)
    (anthropic.APIConnectionError(request=_REQ), "api_error"),
])
async def test_sdk_errors_map_to_reasons(make_fake_anthropic_client, exc, reason):
    client = make_fake_anthropic_client(raise_exc=exc)
    resp = await generate_narrative(client, _request())
    assert resp.meta.source == "fallback" and resp.meta.reason == reason


@pytest.mark.parametrize("content", ["not json", "[1, 2]", None])
async def test_bad_json_maps_to_invalid_json(make_fake_anthropic_client, content):
    client = make_fake_anthropic_client(content)
    resp = await generate_narrative(client, _request())
    assert resp.meta.reason == "invalid_json"


# ── 옛 경로 서버 마감 ────────────────────────────────────────────────────────

def test_client_timeout_budget_fits_server_deadline_and_frontend():
    worst = llm.CALL_TIMEOUT_SECONDS * (llm.MAX_RETRIES + 1)
    assert worst <= service.NARRATIVE_DEADLINE_SECONDS
    assert service.NARRATIVE_DEADLINE_SECONDS < 160  # 옛 프론트의 서술 호출이 160초에 끊는다


async def test_deadline_while_calling_falls_back_as_timeout(monkeypatch, make_fake_anthropic_client):
    monkeypatch.setattr(service, "NARRATIVE_DEADLINE_SECONDS", 0.05)
    client = make_fake_anthropic_client(_CLEAN, delay=2.0)
    resp = await generate_narrative(client, _request())
    assert resp.meta.source == "fallback" and resp.meta.reason == "timeout"
    assert client.beta.messages.create.call_count == 1


async def test_deadline_while_waiting_for_slot_falls_back_as_busy(monkeypatch, make_fake_anthropic_client):
    monkeypatch.setattr(service, "NARRATIVE_DEADLINE_SECONDS", 0.05)
    client = make_fake_anthropic_client(_CLEAN)
    async with contextlib.AsyncExitStack() as stack:
        for _ in range(concurrency.MAX_CONCURRENT_NARRATIVE_CALLS):
            await stack.enter_async_context(concurrency.narrative_slot())
        resp = await generate_narrative(client, _request())
    assert resp.meta.source == "fallback" and resp.meta.reason == "busy"
    client.beta.messages.create.assert_not_called()


async def test_unexpected_exception_falls_back_as_api_error(make_fake_anthropic_client):
    client = make_fake_anthropic_client(raise_exc=RuntimeError("boom"))
    resp = await generate_narrative(client, _request())
    assert resp.meta.reason == "api_error"


# ── 클라이언트 설정(app.main lifespan) ───────────────────────────────────────

def test_anthropic_client_configured_from_key(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    with TestClient(main.app):
        c = main.app.state.anthropic_client
        assert isinstance(c, anthropic.AsyncAnthropic)
        assert c.timeout == llm.CALL_TIMEOUT_SECONDS
        assert c.max_retries == llm.MAX_RETRIES
        assert main.app.state.openai_client is None  # 키가 따로다


def test_missing_keys_warn_separately(caplog):
    from fastapi.testclient import TestClient

    from app import main

    with caplog.at_level(logging.WARNING), TestClient(main.app):
        assert main.app.state.anthropic_client is None
        assert main.app.state.openai_client is None
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "ANTHROPIC_API_KEY 미설정" in messages
    assert "OPENAI_API_KEY 미설정" in messages


def test_router_uses_anthropic_client(monkeypatch, make_fake_anthropic_client):
    """라우터가 서술에 매핑용 OpenAI 클라이언트가 아니라 Anthropic 클라이언트를 넘기는지(배선)."""
    import json

    from fastapi.testclient import TestClient

    from app.main import app

    fake = make_fake_anthropic_client(_CLEAN)
    app.state.anthropic_client = fake
    app.state.openai_client = object()  # 잘못 쓰면 바로 깨진다
    body = TestClient(app).post("/api/generate-narrative", json=json.loads(_request().model_dump_json())).json()
    assert body["meta"]["source"] == "llm", body["meta"]
    assert fake.beta.messages.create.call_count == 1



# ── 독립 검토(S1) 반영 ──────────────────────────────────────────────────────

async def test_cost_uses_requested_model_when_message_iteration_has_no_model():
    """BE-LLM-01: SDK 에서 message 항목의 model 은 선택 칸이다 — 비면 요청한 모델의 단가로 센다."""
    import json

    it = _it("message", None, 1000, 500)
    msg = fake_beta_message(json.dumps(_CLEAN), usage=(1000, 500), iterations=[it])
    assert msg.usage.iterations[0].model is None
    result = await llm.call_narrative_llm(_client_returning(msg), system="s", user="u", schema={"type": "object"})
    assert result.record.cost_usd == pytest.approx(1000 * 2 / 1e6 + 500 * 10 / 1e6)
    assert result.record.iterations[0]["model"] == "claude-sonnet-5-5"


async def test_request_assembly_failure_falls_back_without_calling(monkeypatch, make_fake_anthropic_client):
    """T5: 프롬프트 조립 실패는 500 이 아니라 규칙 폴백(assembly_error)이고, AI 도 지갑도 쓰지 않는다."""
    def boom(*_a, **_k):
        raise ValueError("조립 실패")

    monkeypatch.setattr(service, "build_user_prompt", boom)
    client = make_fake_anthropic_client(_CLEAN)
    resp = await generate_narrative(client, _request())
    assert resp.meta.reason == "assembly_error"
    client.beta.messages.create.assert_not_called()
    assert llm_budget.narrative.remaining() == llm_budget.narrative.limit


async def test_oversized_prompt_falls_back_before_spending_wallet(monkeypatch, make_fake_anthropic_client):
    """OPS-1: 무인증 경로로 거대한 사실 시트를 보내도 AI 를 부르지 않는다(지갑도 그대로)."""
    monkeypatch.setattr(service, "build_user_prompt", lambda *_a, **_k: "x" * (service.MAX_NARRATIVE_PROMPT_CHARS + 1))
    client = make_fake_anthropic_client(_CLEAN)
    resp = await generate_narrative(client, _request())
    assert resp.meta.source == "fallback" and resp.meta.reason == "input_too_large"
    client.beta.messages.create.assert_not_called()
    assert llm_budget.narrative.remaining() == llm_budget.narrative.limit


def test_normal_prompt_is_far_below_size_cap():
    from app.narrative.baselines import build_benchmark_refs
    from app.narrative.derived import compute_derived
    from app.narrative.legacy import build_system_prompt, build_user_prompt

    fs = _no_special_fact_sheet()
    d = compute_derived(fs)
    size = len(build_system_prompt("external", "binary", fs)) + len(
        build_user_prompt(fs.model_dump(), build_benchmark_refs("binary", fs.metrics, edition=3), d, None))
    assert size * 5 < service.MAX_NARRATIVE_PROMPT_CHARS
