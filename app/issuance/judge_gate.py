"""app/issuance/judge_gate.py — 판 4 성적서의 서버 발급 검사(심판 라운드 J5b, Q1 나·계획 14-6·14-9a·사-6).

"검증 안 된 보고서는 어떤 경우에도 발급하지 않는다"(심판 설계 5.5)를 화면 밖(옛 화면·직접 호출)에서도 지킨다. 판 4 성적서는
요청에 실린 서술 작업 번호가 "판 4 · 완료 · 심판 통과"여야 받는다 — 첫 발급과 정정 발급 모두.

판 판정(프론트 normalizeReportContent 의 `contentSchema ?? 1` 규칙과 같게, 닫힌 쪽):
- content 가 없으면(메타만) 검사하지 않는다.
- contentSchema 가 없거나 null 이면 판 1 — 검사하지 않는다(CI 발급 스모크 `{"smoke": true}` 가 여기).
- 정수 1·2·3 이면 검사하지 않는다. 그 밖(4, "4", 4.0, 그 밖의 값)은 모두 판 4 로 보고 검사한다.

거절 사유 코드(거절마다 다름): judge_job_missing(번호 없음) · judge_job_not_found(작업 없음) · judge_job_not_edition4(판 4 작업
아님) · judge_record_expired(작업 기록 보관 기간이 지남) · judge_not_passed(진행 중·실패·보류).
서술 글 지문 대조(같은 작업의 서술인지)는 P1(Q1 다, J9)이다 — 지금은 "통과한 다른 작업 번호"를 막지 못한다.

상호작용
- 의존(import): app.narrative.models(NarrativeJob), app.narrative.jobs(record_retention — 보관 기간 한 곳), app.issuance.service(IssuanceError)
- 사용처: app.issuance.router(issue·reissue)
"""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy.orm import Session

from app.issuance.service import IssuanceError
from app.narrative.models import NarrativeJob, _utcnow

MESSAGES = {
    "judge_job_missing": "판 4 성적서는 심판을 통과한 서술 작업 번호가 있어야 발급할 수 있습니다",
    "judge_job_not_found": "서술 작업을 찾을 수 없어 판 4 성적서를 발급할 수 없습니다",
    "judge_job_not_edition4": "판 4 서술 작업이 아니어서 판 4 성적서를 발급할 수 없습니다",
    "judge_record_expired": "서술 작업 기록의 보관 기간이 지나 판 4 성적서를 발급할 수 없습니다",
    "judge_not_passed": "심판을 통과하지 않은 서술이어서 판 4 성적서를 발급할 수 없습니다(발급 보류)",
}


def content_is_edition4(content: dict | None) -> bool:
    if content is None:
        return False
    schema = content.get("contentSchema")
    if schema is None:
        return False
    return not (type(schema) is int and schema in (1, 2, 3))  # bool 은 int 의 하위형 — type 으로 가른다


def reject_reason(db: Session, content: dict | None, job_id: str | None, *, now: datetime | None = None) -> str | None:
    """발급을 막을 사유 코드. 받으면 None."""
    if not content_is_edition4(content):
        return None
    if not job_id or not job_id.strip():
        return "judge_job_missing"
    job = db.get(NarrativeJob, job_id)
    if job is None:
        return "judge_job_not_found"
    internal = json.loads(job.internal_json or "{}")
    if internal.get("report_edition") != 4:
        return "judge_job_not_edition4"
    from app.narrative.jobs import record_retention  # 늦은 import — 발급 모듈 import 때 서술 작업 묶음 전체를 끌지 않게

    if job.created_at < (now or _utcnow()) - record_retention():
        return "judge_record_expired"
    judge = internal.get("judge")
    if job.status != "done" or not isinstance(judge, dict) or judge.get("gate") != "passed":
        return "judge_not_passed"
    return None


def check(db: Session, content: dict | None, job_id: str | None) -> None:
    """판 4 성적서면 작업 번호를 검사한다. 거절이면 IssuanceError(code=사유 코드)."""
    code = reject_reason(db, content, job_id)
    if code is not None:
        raise IssuanceError(code, f"{MESSAGES[code]} ({code})")
