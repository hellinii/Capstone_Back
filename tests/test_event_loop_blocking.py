"""[G-04b] async 라우터의 CPU 구간이 이벤트 루프를 막지 않아야 한다.

종전에는 세 업로드 라우터와 서술 생성이 `async def` 안에서 동기 CPU 함수를 직접
호출해, 그 시간 동안 다른 모든 요청(헬스체크 포함)이 멈췄다. 실서버 실측으로
26 MB CSV 가 /health 를 5,032 ms 밀어냈다.

여기서는 CPU 함수를 '느린 함수'로 갈아끼우고, 그 요청이 도는 동안 /health 가
빠르게 응답하는지를 잰다. 오프로드가 없으면 /health 는 느린 함수가 끝날 때까지
기다린다.
"""
import asyncio
import io
import json
import threading
import time

import httpx
import pytest
from httpx import ASGITransport

from app.core import concurrency
from app.main import app
from router_cases import CASES, request_json

# 느린 CPU 작업 1회가 무는 시간. 오프로드 없으면 /health 가 이만큼 밀린다.
_BLOCK_SECONDS = 0.5
# /health 가 이 안에 돌아오면 이벤트 루프가 살아 있다고 본다.
_HEALTH_BUDGET = 0.2

_EVAL_DATA = request_json("binary")
_CSV = CASES["binary"]["csv"].read_bytes()


@pytest.fixture(autouse=True)
def _no_llm_client():
    prev = getattr(app.state, "openai_client", None)
    app.state.openai_client = None
    yield
    app.state.openai_client = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


_started = threading.Event()
_finished = threading.Event()


def _slow(*_args, **_kwargs):
    """느린 CPU 작업 대역. 진입·종료를 알려 판정이 타이밍 예산에 기대지 않게 한다."""
    _started.set()
    time.sleep(_BLOCK_SECONDS)
    _finished.set()
    raise RuntimeError("의도된 중단 — 타이밍만 잰다")


async def _assert_health_stays_responsive(request_coro_factory) -> None:
    """느린 CPU 작업이 **아직 도는 중에** /health 가 처리되는지 본다.

    타이밍 예산으로 판정하지 않는다 — 오프로드가 생기면 스레드 홉이 늘어 양보 횟수
    같은 것에 의존하는 측정은 쉽게 무의미해진다. 대신 결정론적인 성질을 본다:

      · 오프로드 있음 — 작업이 스레드로 빠지므로 이벤트 루프가 곧바로 돌아온다.
        `_started` 를 관측하는 시점에 `_finished` 는 아직 unset 이다.
      · 오프로드 없음 — 작업이 루프를 통째로 물고 있으므로 우리가 제어권을 되찾는
        순간 작업은 이미 끝나 있다. `_started` 와 `_finished` 가 함께 set 이다.
    """
    _started.clear()
    _finished.clear()
    async with _client() as c:
        task = asyncio.create_task(request_coro_factory(c))

        # 느린 구간에 진입할 때까지 기다린다. 오프로드가 없으면 이 대기 중에
        # 이벤트 루프가 통째로 멈추고, 깨어났을 때는 작업이 이미 끝나 있다.
        deadline = time.perf_counter() + 5.0
        while not _started.is_set() and time.perf_counter() < deadline:
            await asyncio.sleep(0.005)
        assert _started.is_set(), "느린 구간에 진입하지 않았다 — 이 측정은 무의미하다"

        blocked = _finished.is_set()

        t0 = time.perf_counter()
        await c.get("/health")
        elapsed = time.perf_counter() - t0
        await task

    assert not blocked, (
        "CPU 작업이 이벤트 루프를 물고 있다 — 제어권을 되찾았을 때 작업이 이미 끝나 있었다."
    )
    assert elapsed < _HEALTH_BUDGET, (
        f"/health 가 {elapsed * 1000:.0f} ms 걸렸다 (예산 {_HEALTH_BUDGET * 1000:.0f} ms)."
    )


async def test_evaluate_does_not_block_event_loop(monkeypatch):
    """[G-04b] /api/evaluate 의 평가 파이프라인 구간."""
    monkeypatch.setattr("app.evaluation.router.run_evaluation_pipeline", _slow)

    async def _post(c):
        try:
            return await c.post(
                "/api/evaluate",
                files={"file": ("d.csv", io.BytesIO(_CSV), "text/csv")},
                data={"data": _EVAL_DATA},
            )
        except RuntimeError:
            return None

    await _assert_health_stays_responsive(_post)


async def test_validate_does_not_block_event_loop(monkeypatch):
    """[G-04b] /api/validate-data 의 검증 파이프라인 구간."""
    monkeypatch.setattr("app.analysis.validation_router.validate_dataset", _slow)

    async def _post(c):
        try:
            return await c.post(
                "/api/validate-data",
                files={"file": ("d.csv", io.BytesIO(_CSV), "text/csv")},
                data={"data": _EVAL_DATA},
            )
        except RuntimeError:
            return None

    await _assert_health_stays_responsive(_post)


async def test_parsing_does_not_block_event_loop(monkeypatch):
    """[G-04b] 파싱(pandas.read_csv)도 CPU 바운드다 — 대장 목록에 없던 지점."""
    monkeypatch.setattr("app.analysis.router.parse_file_content", _slow)

    async def _post(c):
        try:
            return await c.post(
                "/api/analyze-columns",
                files={"file": ("d.csv", io.BytesIO(_CSV), "text/csv")},
                data={"task_type": "binary"},
            )
        except RuntimeError:
            return None

    await _assert_health_stays_responsive(_post)


async def test_narrative_does_not_block_event_loop(monkeypatch):
    """[G-04b] /api/generate-narrative — 파일 업로드조차 필요 없는 순수 JSON 경로."""
    from test_narrative_router import _sample_request

    monkeypatch.setattr("app.narrative.service.compute_derived", _slow)
    payload = json.loads(_sample_request().model_dump_json())

    async def _post(c):
        try:
            return await c.post("/api/generate-narrative", json=payload)
        except RuntimeError:
            return None

    await _assert_health_stays_responsive(_post)


async def test_cpu_work_respects_concurrency_cap(monkeypatch):
    """[G-04b] 오프로드는 동시 실행 상한과 한 쌍이어야 한다.

    상한 없이 오프로드만 넣으면 anyio 기본 40스레드까지 DataFrame 이 동시 상주해
    512 MB 한도에서 OOM 위험이 커진다. 상한을 걷어내면 이 테스트가 깨진다.
    """
    lock = threading.Lock()
    state = {"current": 0, "peak": 0}

    def tracked(*_args, **_kwargs):
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        time.sleep(0.15)
        with lock:
            state["current"] -= 1
        raise RuntimeError("의도된 중단")

    monkeypatch.setattr("app.evaluation.router.run_evaluation_pipeline", tracked)

    async def _post(c):
        try:
            return await c.post(
                "/api/evaluate",
                files={"file": ("d.csv", io.BytesIO(_CSV), "text/csv")},
                data={"data": _EVAL_DATA},
            )
        except RuntimeError:
            return None

    async with _client() as c:
        await asyncio.gather(*[_post(c) for _ in range(6)])

    assert state["peak"] <= concurrency.MAX_CONCURRENT_CPU_TASKS, (
        f"동시 실행 {state['peak']} 건 — 상한 {concurrency.MAX_CONCURRENT_CPU_TASKS} 초과"
    )


def test_concurrency_cap_value_is_pinned():
    """[G-04b] 동시 실행 상한 값 자체를 고정한다.

    위 상한 테스트는 상수를 읽어 비교하므로 '기구'는 지키지만 '값'은 지키지 못한다.
    이 값은 512 MB 한도에서 DataFrame 이 몇 벌까지 동시 상주해도 되는지를 정하는
    튜닝 지점이라, 조용히 올라가면 오프로드가 OOM 위험으로 되돌아간다.
    """
    assert concurrency.MAX_CONCURRENT_CPU_TASKS == 2
