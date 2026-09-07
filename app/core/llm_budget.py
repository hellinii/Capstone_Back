"""app/core/llm_budget.py — 외부 LLM 호출의 시간당 총량 예산.

ISSUES.md G-03 (2026-09-07 ★확정된 제품 결정 9).

무인증 공개 API 에 과금 엔드포인트가 둘 열려 있다. 3차 라운드가 **요청 하나하나**를
유계로 만들었지만(입력 상한·재시도 축소·동시 호출 상한) **반복 호출은 여전히 과금된다.**
인증을 넣지 않기로 한 이상 남은 방어선은 총량 상한이다.

## 왜 429 가 아니라 규칙 폴백인가

프론트 두 호출부는 비 200 응답을 폴백이 아니라 **오류**로 처리한다 —
컬럼 분석(`useColumnAnalysis`)은 throw 해서 에러 화면으로 가고, 서술(`fetchNarrative`)은
`source:"error"` 의 빈 서술을 만들어 7·8·9절이 '생성 예정' 안내로 인쇄된다. 예산 초과를
미들웨어 429 로 구현하면 사용자는 '서비스 고장'을 본다.

두 경로 모두 **규칙 폴백이 이미 구현돼 있다**(무키·API 실패 경로). 그래서 이 모듈이
하는 일은 새 폴백을 만드는 것이 아니라 **새 강등 사유 하나를 주는 것**이다.

## 왜 전역인가 (IP 단위가 아니라)

프록시 뒤의 실제 client IP 가 **아직 미관측**이다(`FORWARDED_ALLOW_IPS`·`--proxy-headers`
설정이 저장소에 0건). 모든 요청이 같은 IP 로 보이면 IP 리밋은 전역 리밋으로 붕괴하면서
**정상 사용자 1명이 전체를 막는다** — 전역 리밋과 같은 효과를 내면서 그것을 IP 리밋이라
착각하게 만드는 쪽이 더 나쁘다. 관측이 끝나면 그때 IP 단위를 얹는다.

## 한계 (문서화)

프로세스 로컬이다. 워커가 N개면 실효 상한이 N배가 된다 — 워커 수는 Render 대시보드
Start Command 에 있고 저장소에서 확인할 수 없다(`render.yaml` 이 대시보드 수동 생성임을
명시). **과허용 방향으로만 틀린다**는 점에서 동시 호출 상한과 같은 성질이고, 정상
사용자를 잘못 막지 않는다. 정확한 총량이 필요해지면 DB 카운터로 옮겨야 한다.

상호작용
- 의존(import): time, threading
- 사용처: app.analysis.analysis_service, app.narrative.service
"""
import os
import threading
import time
from collections import deque

# 시간당 허용 호출 수. 환경변수로 낮출 수 있게 두되 기본값은 결정문의 60 이다.
MAX_LLM_CALLS_PER_HOUR = int(os.getenv("LLM_CALLS_PER_HOUR", "60"))

_WINDOW_SECONDS = 3600.0

_lock = threading.Lock()
_calls: deque[float] = deque()


def _prune(now: float) -> None:
    """창 밖으로 나간 호출 기록을 버린다(고정 창이 아니라 슬라이딩 창)."""
    cutoff = now - _WINDOW_SECONDS
    while _calls and _calls[0] <= cutoff:
        _calls.popleft()


def try_consume(now: float | None = None) -> bool:
    """예산을 한 번 쓴다. 남아 있으면 True, 소진됐으면 False.

    `now` 는 테스트가 시간을 직접 주기 위한 것이다 — 타이밍에 의존하는 테스트는
    느리고 불안정하다.
    """
    moment = time.monotonic() if now is None else now
    with _lock:
        _prune(moment)
        if len(_calls) >= MAX_LLM_CALLS_PER_HOUR:
            return False
        _calls.append(moment)
        return True


def remaining(now: float | None = None) -> int:
    """남은 예산."""
    moment = time.monotonic() if now is None else now
    with _lock:
        _prune(moment)
        return max(0, MAX_LLM_CALLS_PER_HOUR - len(_calls))


def reset() -> None:
    """예산 기록을 비운다(테스트 전용)."""
    with _lock:
        _calls.clear()
