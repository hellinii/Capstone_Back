"""app/core/concurrency.py — CPU 바운드 구간의 스레드풀 오프로드 + 동시 실행 상한

`async def` 라우터 안에서 pandas/sklearn 같은 동기 CPU 작업을 직접 호출하면 그 시간
동안 이벤트 루프 전체가 멈춘다. 실서버(uvicorn 1워커) 실측 —

    26 MB CSV → POST /api/evaluate           : 요청 5.33 s, 그동안 /health max 5,032 ms
    18.8 MB JSON → POST /api/generate-narrative: 요청 15.41 s, 그동안 /health max 15,112 ms
    (유휴 기준선 /health max 32 ms)

`/health` 가 그만큼 밀리면 Render 헬스체크가 타임아웃해 인스턴스가 재시작될 수 있다.

**오프로드만 넣으면 오히려 나빠진다.** 지금은 CPU 구간이 이벤트 루프에서 직렬화돼
사실상 동시 1건인데, 스레드풀로 옮기면 anyio 기본 한도(40)까지 동시 실행돼 DataFrame
이 최대 40개 동시 상주한다. Render free 는 512 MB 다. 그래서 오프로드와 동시 실행
상한은 분리할 수 없는 한 쌍이고, 이 모듈이 둘을 함께 제공한다.

상한 초과 시 거절(429)하지 않고 **대기**시킨다 — 정상 사용자 두 명이 동시에 평가를
돌렸다는 이유로 한 명이 실패하면 안 되기 때문이다. 대기가 길어지면 클라이언트
타임아웃에 걸리는데, 그것은 오프로드 이전에도 마찬가지였다(직렬화돼 있었으므로).

상호작용
- 의존(import): asyncio, weakref, fastapi.concurrency(run_in_threadpool)
- 사용처: app.analysis.router / app.analysis.validation_router / app.evaluation.router /
  app.narrative.service
"""
import asyncio
import weakref
from contextlib import asynccontextmanager
from typing import Any, Callable, TypeVar

from fastapi.concurrency import run_in_threadpool

T = TypeVar("T")

# 동시에 스레드풀에서 돌 수 있는 CPU 작업 수. 종전 동작(직렬화 = 사실상 1)보다 크되
# 512 MB 한도에서 DataFrame 이 여러 벌 상주해도 견딜 수 있는 값으로 잡았다.
MAX_CONCURRENT_CPU_TASKS = 2

# 동시에 진행할 수 있는 외부 LLM 호출 수(G-03). 워커 수를 알 수 없는 환경에서
# 프로세스 로컬 세마포어는 **과허용 방향으로만 틀리는** 유일한 총량 장치다 —
# 워커가 N개면 실효 상한이 N배로 느슨해질 뿐 정상 사용자를 잘못 막지 않는다.
MAX_CONCURRENT_LLM_CALLS = 4

# asyncio.Semaphore 는 처음 대기가 발생할 때 실행 중인 이벤트 루프에 묶인다. 테스트는
# TestClient 마다 새 루프를 만들 수 있으므로 모듈 전역에 하나를 두면 'bound to a
# different event loop' 로 깨진다. 루프별로 하나씩 만들되 루프가 사라지면 함께
# 수거되도록 WeakKeyDictionary 를 쓴다. 프로덕션에는 루프가 하나뿐이다.
_cpu_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)
_llm_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _get_semaphore(registry, limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = registry.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(limit)
        registry[loop] = sem
    return sem


async def run_cpu_bound(func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """동기 CPU 함수를 스레드풀에서 실행한다. 동시 실행 수는 상한까지만 허용한다.

    예외는 그대로 전파되므로 호출부의 기존 try/except 구조를 바꾸지 않는다.
    """
    async with _get_semaphore(_cpu_semaphores, MAX_CONCURRENT_CPU_TASKS):
        return await run_in_threadpool(func, *args, **kwargs)


@asynccontextmanager
async def llm_slot():
    """외부 LLM 호출 하나를 점유한다. 무인증 과금 경로의 총량 장치(G-03).

    거절이 아니라 대기다 — 정상 사용자를 잘못 막지 않는 것이 우선이고, 남용은
    입력 상한·재시도 축소가 함께 맡는다.
    """
    async with _get_semaphore(_llm_semaphores, MAX_CONCURRENT_LLM_CALLS):
        yield
