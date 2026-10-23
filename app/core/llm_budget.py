"""app/core/llm_budget.py — 외부 LLM 호출의 시간당 총량 예산.

G-03 (2026-09-07 제품 결정 — 인증은 넣지 않고, LLM 호출 총량에 시간당 상한을 둔다).

무인증 공개 API 에 과금 엔드포인트가 둘 열려 있다. 3차 라운드가 **요청 하나하나**를
유계로 만들었지만(입력 상한·재시도 축소·동시 호출 상한) **반복 호출은 여전히 과금된다.**
인증을 넣지 않기로 한 이상 남은 방어선은 총량 상한이다.

## 왜 429 가 아니라 규칙 폴백인가

프론트 두 호출부는 비 200 응답을 폴백이 아니라 **오류**로 처리한다 —
컬럼 분석(`useColumnAnalysis`)은 throw 해서 에러 화면으로 가고, 옛 프론트의 서술 호출부는
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

## 지갑 둘 — 매핑과 서술 (서술 라운드 S1)

서술이 Claude 로 옮겨 가며 두 경로의 호출 한 건 값이 크게 달라졌다(매핑은 작은 OpenAI 호출, 서술은
보고서 한 건). 한 지갑을 나눠 쓰면 서술 남용이 매핑까지 막고, 매핑 상한에 맞춘 개수로는 서술 비용을
가늠할 수 없다. 그래서 지갑을 나눈다 — 한쪽을 다 써도 다른 쪽은 계속 호출된다.

- `mapping`: 컬럼 매핑 호출 1건 = 1. `LLM_CALLS_PER_HOUR`(기본 60). 모듈 함수(try_consume·remaining·
  MAX_LLM_CALLS_PER_HOUR)는 종전대로 이 지갑을 가리킨다.
- `narrative`: 서술 **보고서** 1건 = 1(옛 경로는 호출 1번이 보고서 1건). `NARRATIVE_REPORTS_PER_HOUR`(기본 30).
  작업 번호 경로는 작업을 만들 때 보고서 1건을 **예약**한다(app.narrative.jobs). 예약은 돌려주지 않는다 —
  AI 를 부르지 않고 끝나도(요청이 너무 큼 등) 센 채로 둔다. 과소비 방향으로만 틀린다.
- `narrative_jobs`: 서술 **작업 행** 1건 = 1. `NARRATIVE_JOBS_PER_HOUR`(기본 200). AI 비용이 아니라 DB 행을
  막는다 — 키가 없거나 서술 지갑이 비어도 작업은 규칙 문장으로 끝나며 행을 하나씩 남기므로, 무인증 경로를
  되풀이해 부르면 발급 DB(Neon 무료 용량)가 찬다. 넘으면 **429 가 아니라** 행을 만들지 않고 규칙 문장 결과를
  작업 생성 응답에 바로 담는다(app.narrative.jobs) — 누가 상한을 다 써도 다른 사용자는 완성본과 8.1 표를 받는다.

상호작용
- 의존(import): time, threading
- 사용처: app.analysis.analysis_service(mapping), app.narrative.legacy(narrative — 옛 경로), app.narrative.jobs(narrative·narrative_jobs)
"""
import os
import threading
import time
from collections import deque

_WINDOW_SECONDS = 3600.0


class Wallet:
    """한 경로의 시간당 예산(슬라이딩 창). 프로세스 로컬 — 위 '한계' 참조."""

    def __init__(self, name: str, limit: int):
        self.name = name
        self.limit = limit
        self._lock = threading.Lock()
        self._calls: deque[float] = deque()

    def _prune(self, now: float) -> None:
        """창 밖으로 나간 기록을 버린다(고정 창이 아니라 슬라이딩 창)."""
        cutoff = now - _WINDOW_SECONDS
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

    def try_consume(self, now: float | None = None) -> bool:
        """예산을 한 번 쓴다. 남아 있으면 True, 소진됐으면 False.

        `now` 는 테스트가 시간을 직접 주기 위한 것이다 — 타이밍에 의존하는 테스트는 느리고 불안정하다.
        """
        moment = time.monotonic() if now is None else now
        with self._lock:
            self._prune(moment)
            if len(self._calls) >= self.limit:
                return False
            self._calls.append(moment)
            return True

    def remaining(self, now: float | None = None) -> int:
        """남은 예산."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            self._prune(moment)
            return max(0, self.limit - len(self._calls))

    def refund(self) -> None:
        """가장 최근에 쓴 한 번을 돌려준다 — 쓴 뒤의 일이 실패해 그 한 번이 쓰이지 않았을 때(예: 작업 행 저장 실패)."""
        with self._lock:
            if self._calls:
                self._calls.pop()

    def reset(self) -> None:
        """기록을 비운다(테스트 전용)."""
        with self._lock:
            self._calls.clear()


def _env_limit(name: str, default: int) -> int:
    """시간당 상한 환경변수. 비었거나 정수가 아니거나 1 미만이면 기본값 — 오타가 기동 실패(int 변환 예외)나
    상한 0(모든 호출 강등)이 되지 않게."""
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 1 else default


# 매핑: 시간당 허용 호출 수. 환경변수로 낮출 수 있게 두되 기본값은 결정문의 60 이다.
MAX_LLM_CALLS_PER_HOUR = _env_limit("LLM_CALLS_PER_HOUR", 60)
# 서술: 시간당 허용 보고서 수. Sonnet 5.5 보고서 1건은 매핑 1건보다 훨씬 비싸 따로 센다.
MAX_NARRATIVE_REPORTS_PER_HOUR = _env_limit("NARRATIVE_REPORTS_PER_HOUR", 30)
# 서술 작업 생성: 시간당 허용 작업 수(행을 남기는 작업). 정상 사용(보고서 몇 건에 다시 만들기 몇 번)보다 훨씬 크게 잡는다.
MAX_NARRATIVE_JOBS_PER_HOUR = _env_limit("NARRATIVE_JOBS_PER_HOUR", 200)

mapping = Wallet("mapping", MAX_LLM_CALLS_PER_HOUR)
narrative = Wallet("narrative", MAX_NARRATIVE_REPORTS_PER_HOUR)
narrative_jobs = Wallet("narrative_jobs", MAX_NARRATIVE_JOBS_PER_HOUR)


def try_consume(now: float | None = None) -> bool:
    """매핑 지갑을 한 번 쓴다(종전 이름 유지)."""
    return mapping.try_consume(now)


def remaining(now: float | None = None) -> int:
    """매핑 지갑의 남은 예산(종전 이름 유지)."""
    return mapping.remaining(now)


def reset() -> None:
    """지갑을 모두 비운다(테스트 전용)."""
    mapping.reset()
    narrative.reset()
    narrative_jobs.reset()
