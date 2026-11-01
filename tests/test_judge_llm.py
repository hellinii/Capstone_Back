"""tests/test_judge_llm.py — 심판 OpenAI 호출 한 곳(app/judge/llm.py)·설정(app/judge/config.py)."""
from __future__ import annotations

import pytest

from app.judge import config, llm
from tests import judge_fakes as F


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _call(client, clock=None, deadline=None, **kw):
    return llm.create_response(
        client, purpose="test", model="gpt-6.1-sol", instructions="지시", input=[{"role": "user", "content": "글"}],
        timeout_s=60, deadline=deadline, clock=clock or Clock(), **kw,
    )


# ── 요청 모양 ──


async def test_request_shape_is_bound_to_sdk_signature():
    fake = F.FakeOpenAI([F.json_response({"ok": True})])
    out = await _call(fake, text_format=llm.json_schema_format("x", {"type": "object"}), tools=[{"type": "function", "name": "f"}])
    assert out.failure is None
    sent = fake.calls[0]
    assert sent["store"] is False
    assert sent["include"] == ["reasoning.encrypted_content"]
    assert sent["reasoning"] == {"effort": "medium"}
    assert sent["text"] == {"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}, "strict": True}}
    assert sent["tools"] == [{"type": "function", "name": "f"}]
    assert "temperature" not in sent and "seed" not in sent
    assert fake.options == [{"max_retries": 0, "timeout": 60}]


async def test_fake_rejects_argument_unknown_to_sdk():
    fake = F.FakeOpenAI([F.json_response({})])
    with pytest.raises(TypeError):
        await fake.create(model="m", input="x", no_such_argument=1)


async def test_no_client_is_a_failure_not_an_exception():
    out = await _call(None)
    assert (out.response, out.failure, out.record.attempts) == (None, "no_client", 0)


# ── 재시도: 일시적 오류만 한 번 더 ──


@pytest.mark.parametrize("exc", [F.rate_limit, F.server_error, F.connection])
async def test_transient_error_is_retried_once(exc):
    fake = F.FakeOpenAI([exc(), F.json_response({"ok": 1})])
    out = await _call(fake)
    assert out.failure is None and out.record.attempts == 2 and len(fake.calls) == 2


async def test_transient_error_twice_gives_up_after_two_http_attempts():
    fake = F.FakeOpenAI([F.rate_limit(), F.server_error()])
    out = await _call(fake)
    assert (out.failure, out.record.attempts, len(fake.calls)) == ("transient", 2, 2)


@pytest.mark.parametrize("exc, kind", [(F.timeout, "timeout"), (F.bad_request, "api_error")])
async def test_timeout_and_client_errors_are_not_retried(exc, kind):
    fake = F.FakeOpenAI([exc(), F.json_response({})])
    out = await _call(fake)
    assert (out.failure, len(fake.calls)) == (kind, 1)


async def test_no_retry_when_too_little_time_left():
    clock = Clock(0.0)

    def fail_late(_kwargs):
        clock.t = 100 - llm.MIN_RETRY_SECONDS + 1  # 남은 시간이 재시도 최소보다 적다
        return F.rate_limit()

    fake = F.FakeOpenAI([fail_late, F.json_response({})])
    out = await _call(fake, clock=clock, deadline=100)
    assert (out.failure, len(fake.calls)) == ("transient", 1)


async def test_call_timeout_is_cut_to_remaining_time_and_past_deadline_is_not_called():
    fake = F.FakeOpenAI([F.json_response({})])
    await _call(fake, clock=Clock(70.0), deadline=100.0)
    assert fake.options[0]["timeout"] == 30.0
    fake2 = F.FakeOpenAI([F.json_response({})])
    out = await _call(fake2, clock=Clock(100.0), deadline=100.0)
    assert (out.failure, fake2.calls) == ("deadline", [])


# ── 응답 읽기 ──


async def test_refusal_and_incomplete_are_failures():
    out = await _call(F.FakeOpenAI([F.response([F.message_item("no", refusal=True)])]))
    assert (out.response, out.failure) == (None, "refusal")
    out = await _call(F.FakeOpenAI([F.response([F.message_item("{")], status="incomplete")]))
    assert out.failure == "incomplete"


async def test_parse_json_reads_output_text_and_flags_bad_output():
    out = await _call(F.FakeOpenAI([F.json_response({"a": [1, 2]})]))
    assert llm.parse_json(out) == ({"a": [1, 2]}, None)
    bad = await _call(F.FakeOpenAI([F.response([F.message_item("not json")])]))
    assert llm.parse_json(bad) == (None, "bad_output") and bad.record.failure == "bad_output"
    arr = await _call(F.FakeOpenAI([F.response([F.message_item("[1]")])]))
    assert llm.parse_json(arr) == (None, "bad_output")


async def test_usage_and_cost_are_recorded():
    use = F.usage(input_tokens=50_000, cached=20_000, output_tokens=10_000, reasoning=4_000)
    out = await _call(F.FakeOpenAI([F.response([F.function_call_item("lookup", {"field": "M1"}),
                                                 F.message_item("{}")], use=use)]))
    r = out.record
    assert (r.input_tokens, r.cached_input_tokens, r.output_tokens, r.reasoning_tokens, r.tool_calls) == (50_000, 20_000, 10_000, 4_000, 1)
    # (50000-20000)×2 + 20000×0.1 + 10000×10 = 60000 + 2000 + 100000 → /1e6
    assert r.cost_usd == pytest.approx(0.162)
    assert r.response_model == "gpt-6.1-sol"


def test_cost_unknown_model_is_none_and_dated_snapshot_uses_base_price():
    assert llm.cost_usd("gpt-9", 1, 0, 1) is None
    assert llm.cost_usd("gpt-6.1-sol-mini", 1, 0, 1) is None  # 다른 모델을 같은 단가로 세지 않는다
    assert llm.cost_usd("gpt-6.1-sol-2026-10-01", 1_000_000, 0, 0) == pytest.approx(2.0)
    assert llm.cost_usd("gpt-6-luna", 1_000_000, 1_000_000, 1_000_000) == pytest.approx(0.01 + 0.50)


async def test_next_turn_items_keep_encrypted_reasoning_and_function_calls():
    out = await _call(F.FakeOpenAI([F.response([F.reasoning_item("enc-xyz"), F.function_call_item("calc", {"expr": "1+1"}, "call_9")])]))
    items = llm.next_turn_items(out.response)
    assert [i["type"] for i in items] == ["reasoning", "function_call"]
    assert items[0]["encrypted_content"] == "enc-xyz" and items[1]["call_id"] == "call_9"
    assert [c.name for c in llm.function_calls(out.response)] == ["calc"]


# ── 설정 ──


def test_config_defaults_and_env(monkeypatch):
    for k in ("JUDGE_MODEL", "JUDGE_EXTRACT_MODEL", "JUDGE_EFFORT"):
        monkeypatch.delenv(k, raising=False)
    assert (config.judge_model(), config.extract_model(), config.judge_effort()) == ("gpt-6.1-sol", "gpt-6-luna", "medium")
    monkeypatch.setenv("JUDGE_EFFORT", "HIGH")
    assert config.judge_effort() == "high"
    monkeypatch.setenv("JUDGE_EFFORT", "max")  # Responses 가 모르는 값은 기본값으로
    assert config.judge_effort() == "medium"
