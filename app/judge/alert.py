"""app/judge/alert.py — 운영자 알림(계획 13-4·정7).

찾기 쉬운 머리말이 붙은 서버 로그 한 줄 + 환경변수로 켜는 웹훅(`JUDGE_ALERT_WEBHOOK_URL`, 비었으면 로그만).
- 범위: 판 4 발급 보류와 생성 실패(시간 초과·오류·조회 때 유실 판정) — 설계 5.5, 변경 이력 v0.18.
- 작업 하나에 한 번(JobAlert 하나가 한 번만 보낸다). 끝 상태 쓰기를 시도한 뒤 보내되, 쓰기가 실패해도 보낸다(호출부 몫).
- 웹훅은 몇 초 제한·재시도 없음. 실패하거나 멈춰도 예외를 밖으로 내지 않고 작업 상태를 바꾸지 않는다.
- 알림에는 작업 번호·종류·짧은 사유만 싣는다 — 서술 글·근거 인용·고객 입력은 싣지 않는다.

상호작용
- 의존(import): asyncio, json, logging, os, urllib
- 사용처: (J5b) app.narrative.jobs, tests
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# 서버 로그에서 찾는 머리말.
LOG_PREFIX = "[심판 알림]"
WEBHOOK_ENV = "JUDGE_ALERT_WEBHOOK_URL"
# 웹훅 한 번의 제한(초). 넘으면 버린다(재시도 없음).
WEBHOOK_TIMEOUT_SECONDS = 3.0

KINDS = ("held", "failed", "lost")


def webhook_url() -> str | None:
    value = os.getenv(WEBHOOK_ENV, "").strip()
    return value if value.startswith(("https://", "http://")) else None


def _post(url: str, body: bytes) -> int:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT_SECONDS) as resp:  # noqa: S310 — 운영자가 정한 주소
        return resp.status


class JobAlert:
    """작업 하나의 알림. send 를 여러 번 불러도 한 번만 나간다."""

    def __init__(self, job_id: str | None):
        self.job_id = job_id
        self.sent = False

    async def send(self, kind: str, reason: str | None, **extra) -> bool:
        """알린다. 이번 호출이 실제로 보냈으면 True(이미 보냈으면 False). 예외를 내지 않는다."""
        if self.sent:
            return False
        self.sent = True
        payload = {"job_id": self.job_id, "kind": kind, "reason": reason,
                   **{k: v for k, v in extra.items() if isinstance(v, (str, int, float, bool)) or v is None}}
        line = json.dumps(payload, ensure_ascii=False)
        logger.warning("%s %s", LOG_PREFIX, line)
        url = webhook_url()
        if url is None:
            return True
        try:
            await asyncio.wait_for(asyncio.to_thread(_post, url, line.encode("utf-8")), WEBHOOK_TIMEOUT_SECONDS + 1.0)
        except Exception as exc:  # noqa: BLE001 — 알림 실패는 작업에 영향을 주지 않는다
            logger.warning("%s 웹훅 실패(%s) — 로그만 남음", LOG_PREFIX, type(exc).__name__)
        return True
