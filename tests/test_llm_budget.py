"""tests/test_llm_budget.py — LLM 호출 시간당 예산 상한.

G-03 (2026-09-07 제품 결정 — "인증은 넣지 않는다. 대신 LLM 호출
예산 상한(시간당 60회) + 초과 시 규칙 폴백 강등").

**429 가 아니라 200 + 폴백이어야 한다.** 프론트 두 호출부는 비 200 응답을 폴백이 아니라
**오류**로 처리한다 — 컬럼 분석은 throw 해서 에러 화면으로 가고, 서술은 빈 서술을 만들어
7·8·9절이 '생성 예정' 안내로 인쇄된다. 즉 예산 초과를 미들웨어(429)로 구현하면 사용자는
'서비스 고장'을 보게 된다. 규칙 폴백은 이미 두 경로에 다 있으므로, 필요한 것은
**새 강등 사유 하나**뿐이다.

서술 라운드 S1 부터 지갑이 둘이다 — 매핑(호출 1건 = 1, 기본 60)과 서술(보고서 1건 = 1, 기본 30).
한쪽을 다 써도 다른 쪽은 계속 호출된다.
"""
import pytest

from app.core import llm_budget


@pytest.fixture(autouse=True)
def _reset_budget():
    llm_budget.reset()
    yield
    llm_budget.reset()


def test_budget_allows_calls_up_to_the_limit():
    for i in range(llm_budget.MAX_LLM_CALLS_PER_HOUR):
        assert llm_budget.try_consume() is True, f"{i + 1}번째 호출이 막혔다"


def test_budget_blocks_after_the_limit():
    for _ in range(llm_budget.MAX_LLM_CALLS_PER_HOUR):
        llm_budget.try_consume()
    assert llm_budget.try_consume() is False


def test_budget_window_rolls_forward():
    """1시간이 지나면 예산이 회복된다(고정 창이 아니라 경과 시간 기준)."""
    now = 1_000_000.0
    for _ in range(llm_budget.MAX_LLM_CALLS_PER_HOUR):
        llm_budget.try_consume(now=now)
    assert llm_budget.try_consume(now=now) is False

    assert llm_budget.try_consume(now=now + 3601) is True


def test_budget_does_not_recover_early():
    now = 1_000_000.0
    for _ in range(llm_budget.MAX_LLM_CALLS_PER_HOUR):
        llm_budget.try_consume(now=now)
    assert llm_budget.try_consume(now=now + 3599) is False


def test_remaining_reports_what_is_left():
    assert llm_budget.remaining() == llm_budget.MAX_LLM_CALLS_PER_HOUR
    llm_budget.try_consume()
    assert llm_budget.remaining() == llm_budget.MAX_LLM_CALLS_PER_HOUR - 1


# ── 두 LLM 경로가 예산 초과 시 200 + 규칙 폴백으로 강등한다 ────────────────

async def test_narrative_degrades_to_rule_fallback_when_budget_is_spent(make_fake_anthropic_client):
    from app.narrative.schemas import NarrativeRequest
    from app.narrative.legacy import generate_narrative
    from tests.narrative_fixtures import minimal_narrative_request

    for _ in range(llm_budget.narrative.limit):
        llm_budget.narrative.try_consume()

    client = make_fake_anthropic_client({"interpretation": {}, "conclusion": {}})
    result = await generate_narrative(client, minimal_narrative_request())

    assert result.meta.source == "fallback"
    assert result.meta.reason == "budget_exceeded"
    # 예산이 소진됐으면 실제 호출이 일어나면 안 된다 — 그것이 상한의 존재 이유다.
    client.beta.messages.create.assert_not_called()


async def test_narrative_still_calls_llm_within_budget(make_fake_anthropic_client):
    from app.narrative.legacy import generate_narrative
    from tests.narrative_fixtures import minimal_narrative_request

    client = make_fake_anthropic_client({"interpretation": {}, "conclusion": {}})
    await generate_narrative(client, minimal_narrative_request())

    client.beta.messages.create.assert_called()


def test_wallets_are_separate_objects_with_their_own_limits():
    assert llm_budget.mapping is not llm_budget.narrative
    assert llm_budget.mapping.limit == llm_budget.MAX_LLM_CALLS_PER_HOUR == 60
    assert llm_budget.narrative.limit == llm_budget.MAX_NARRATIVE_REPORTS_PER_HOUR == 30


async def test_spent_narrative_wallet_does_not_block_mapping(make_fake_openai_client):
    import pandas as pd

    from app.analysis.analysis_service import resolve_column_mapping
    from app.core.schemas import TaskType

    while llm_budget.narrative.try_consume():
        pass
    client = make_fake_openai_client({
        "column_mappings": [{"column": "y_true", "role": "y_true", "sample_values": []}],
    })
    df = pd.DataFrame({"y_true": [1, 0], "y_pred": [1, 0]})
    await resolve_column_mapping(client=client, task_type=TaskType.binary, columns=list(df.columns), df=df)
    client.chat.completions.create.assert_called_once()
    assert llm_budget.mapping.remaining() == llm_budget.mapping.limit - 1


async def test_spent_mapping_wallet_does_not_block_narrative(make_fake_anthropic_client):
    from app.narrative.legacy import generate_narrative
    from tests.narrative_fixtures import minimal_narrative_request

    while llm_budget.mapping.try_consume():
        pass
    client = make_fake_anthropic_client({"interpretation": {}, "conclusion": {}})
    await generate_narrative(client, minimal_narrative_request())
    client.beta.messages.create.assert_called_once()
    assert llm_budget.narrative.remaining() == llm_budget.narrative.limit - 1  # 보고서 1건 = 1


def test_column_analysis_degrades_to_rule_fallback_when_budget_is_spent(make_fake_openai_client):
    """[G-03] 컬럼 분석은 매핑 지갑이 비면 규칙 폴백으로 강등한다."""
    import pandas as pd

    from app.analysis.analysis_service import resolve_column_mapping
    from app.core.schemas import TaskType

    for _ in range(llm_budget.MAX_LLM_CALLS_PER_HOUR):
        llm_budget.try_consume()

    client = make_fake_openai_client({"column_mappings": []})
    df = pd.DataFrame({"id": ["a", "b"], "y_true": [1, 0], "y_pred": [1, 0]})

    import asyncio
    result = asyncio.run(resolve_column_mapping(
        client=client, task_type=TaskType.binary, columns=list(df.columns), df=df,
    ))

    # 규칙 폴백이 실제로 매핑을 돌려준다(빈 응답이 아니다).
    assert result.column_mappings
    client.chat.completions.create.assert_not_called()


def test_column_analysis_calls_llm_within_budget(make_fake_openai_client):
    import asyncio

    import pandas as pd

    from app.analysis.analysis_service import resolve_column_mapping
    from app.core.schemas import TaskType

    client = make_fake_openai_client({
        "column_mappings": [{"column": "y_true", "role": "y_true", "sample_values": []}],
    })
    df = pd.DataFrame({"y_true": [1, 0], "y_pred": [1, 0]})
    asyncio.run(resolve_column_mapping(
        client=client, task_type=TaskType.binary, columns=list(df.columns), df=df,
    ))

    client.chat.completions.create.assert_called()



def test_wallet_limits_read_their_env_names(monkeypatch):
    """T6: 운영에서 상한을 낮출 때 쓰는 환경변수 이름을 실제로 읽는다(import 때 한 번)."""
    import importlib

    monkeypatch.setenv("NARRATIVE_REPORTS_PER_HOUR", "5")
    monkeypatch.setenv("LLM_CALLS_PER_HOUR", "7")
    try:
        reloaded = importlib.reload(llm_budget)
        assert reloaded.narrative.limit == 5 and reloaded.mapping.limit == 7
    finally:
        monkeypatch.delenv("NARRATIVE_REPORTS_PER_HOUR")
        monkeypatch.delenv("LLM_CALLS_PER_HOUR")
        importlib.reload(llm_budget)
