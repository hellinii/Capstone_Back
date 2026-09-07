"""tests/test_llm_budget.py — LLM 호출 시간당 예산 상한.

ISSUES.md G-03 (2026-09-07 ★확정된 제품 결정 9 — "인증은 넣지 않는다. 대신 LLM 호출
예산 상한(시간당 60회) + 초과 시 규칙 폴백 강등").

**429 가 아니라 200 + 폴백이어야 한다.** 프론트 두 호출부는 비 200 응답을 폴백이 아니라
**오류**로 처리한다 — 컬럼 분석은 throw 해서 에러 화면으로 가고, 서술은 빈 서술을 만들어
7·8·9절이 '생성 예정' 안내로 인쇄된다. 즉 예산 초과를 미들웨어(429)로 구현하면 사용자는
'서비스 고장'을 보게 된다. 규칙 폴백은 이미 두 경로에 다 있으므로, 필요한 것은
**새 강등 사유 하나**뿐이다.
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

async def test_narrative_degrades_to_rule_fallback_when_budget_is_spent(make_fake_openai_client):
    from app.narrative.schemas import NarrativeRequest
    from app.narrative.service import generate_narrative
    from tests.narrative_fixtures import minimal_narrative_request

    for _ in range(llm_budget.MAX_LLM_CALLS_PER_HOUR):
        llm_budget.try_consume()

    client = make_fake_openai_client({"interpretation": {}, "conclusion": {}})
    result = await generate_narrative(client, minimal_narrative_request())

    assert result.meta.source == "fallback"
    assert result.meta.reason == "budget_exceeded"
    # 예산이 소진됐으면 실제 호출이 일어나면 안 된다 — 그것이 상한의 존재 이유다.
    client.chat.completions.create.assert_not_called()


async def test_narrative_still_calls_llm_within_budget(make_fake_openai_client):
    from app.narrative.service import generate_narrative
    from tests.narrative_fixtures import minimal_narrative_request

    client = make_fake_openai_client({"interpretation": {}, "conclusion": {}})
    await generate_narrative(client, minimal_narrative_request())

    client.chat.completions.create.assert_called()


def test_budget_is_shared_between_both_llm_endpoints():
    """두 엔드포인트가 같은 지갑을 쓴다 — 한쪽만 막으면 상한이 두 배가 된다."""
    import app.analysis.analysis_service as analysis_service
    import app.narrative.service as narrative_service

    assert analysis_service.llm_budget is narrative_service.llm_budget


def test_column_analysis_degrades_to_rule_fallback_when_budget_is_spent(make_fake_openai_client):
    """[G-03] 컬럼 분석도 같은 지갑을 쓴다 — 한쪽만 막으면 실효 상한이 두 배가 된다."""
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
