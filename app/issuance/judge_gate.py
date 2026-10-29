"""app/issuance/judge_gate.py — 판 4 성적서의 서버 발급 검사(심판 라운드 J5b, Q1 나·계획 14-6·14-9a·사-6).

"검증 안 된 보고서는 어떤 경우에도 발급하지 않는다"(심판 설계 5.5)를 화면 밖(옛 화면·직접 호출)에서도 지킨다. 판 4 성적서는
요청에 실린 서술 작업 번호가 "판 4 · 완료 · 심판 통과"여야 받는다 — 첫 발급과 정정 발급 모두.

판 판정(프론트 normalizeReportContent 의 `contentSchema ?? 1` 규칙과 같게, 닫힌 쪽):
- content 가 없으면(메타만) 검사하지 않는다.
- contentSchema 가 없거나 null 이면 판 1 — 검사하지 않는다(CI 발급 스모크 `{"smoke": true}` 가 여기).
- 정수 1·2·3 이면 검사하지 않는다. 그 밖(4, "4", 4.0, 그 밖의 값)은 모두 판 4 로 보고 검사한다.

거절 사유 코드(거절마다 다름): judge_job_missing(번호 없음) · judge_job_not_found(작업 없음) · judge_job_not_edition4(판 4 작업
아님) · judge_record_expired(작업 기록 보관 기간이 지남) · judge_not_passed(진행 중·실패·보류) ·
judge_narrative_unrecorded(통과 기록에 서술 글 지문이 없음) · judge_narrative_mismatch(content 의 서술 글이 그 작업의 글과 다름) ·
judge_criteria_unverified(통과 기록이 검증 기록 잠금을 거치지 않음).

검증 기록 잠금 확인(J10 검토, 계획 14-10): 통과 기록의 judge.criteria_lock 이 "ok"(검증한 기준으로 채점)면 받는다. "off"(잠금을 끈
개발·시험 서버에서 채점)는 **지금도 잠금이 꺼져 있을 때만** 받는다 — 운영은 잠금을 끌 길이 없으니(app.judge.criteria_lock) 닫힌
쪽이고, 개발 스모크는 그대로 된다. 그 밖의 값·없음(잠금 전 기록)은 거절.

서술 글 지문 대조(J9, 계획 14-9b·Q1 다): 작업이 통과할 때 남긴 서술 글 지문(internal judge.narrative_sha256)과 content 안 서술 글의
지문(app.narrative.digest — 무엇을 같게 보는지는 그 모듈 머리)을 맞댄다. "통과한 다른 작업 번호"·서술을 바꾼 content 를 막는다.
프론트는 모르는 judge_* 코드도 B48 "서버가 발급을 거절했습니다: {서버 글}"로 보인다(judgeNotice.isJudgeRejection) — 새 화면 글 없음.

상호작용
- 의존(import): app.judge.criteria_lock(enabled — 늦은 import), app.narrative.models(NarrativeJob), app.narrative.digest(서술 글 지문), app.narrative.jobs(record_retention — 보관 기간 한 곳), app.issuance.service(IssuanceError)
- 사용처: app.issuance.router(issue·reissue)
"""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy.orm import Session

from app.issuance.service import IssuanceError
from app.narrative import digest
from app.narrative.models import NarrativeJob, _utcnow

MESSAGES = {  # 고객에게 그대로 보이는 글(프론트 "서버가 발급을 거절했습니다: {글} (코드)") — 사용자 결정 문안(2026-10-29), 내부 낱말 금지
    "judge_job_missing": "자동 검증을 거친 서술인지 확인할 서술 작업 번호가 요청에 없어 발급할 수 없습니다. 화면을 새로 고친 뒤 다시 발급해 주세요.",
    "judge_job_not_found": "이 성적서의 서술을 만든 작업을 찾을 수 없어 발급할 수 없습니다. 서술을 다시 만든 뒤 발급해 주세요.",
    "judge_job_not_edition4": "요청에 담긴 서술 작업이 이 성적서 양식에 맞는 작업이 아니어서 발급할 수 없습니다. 서술을 다시 만든 뒤 발급해 주세요.",
    "judge_record_expired": "서술 작업 기록을 보관하는 기간이 지나 자동 검증 결과를 확인할 수 없습니다. 서술을 다시 만든 뒤 발급해 주세요.",
    "judge_not_passed": "이 서술은 자동 검증을 통과하지 못해 발급이 보류되었습니다. 성적서 화면 위쪽의 안내를 확인해 주세요.",
    "judge_narrative_unrecorded": "자동 검증을 통과한 서술의 내용이 기록되어 있지 않아, 지금 성적서의 서술과 같은지 확인할 수 없습니다. 서술을 다시 만든 뒤 발급해 주세요.",
    "judge_narrative_mismatch": "성적서의 서술이 자동 검증을 통과한 서술과 달라 발급할 수 없습니다. 검증을 통과한 뒤 서술이 바뀌었다면 서술을 다시 만들어 검증을 다시 받아 주세요.",
    "judge_criteria_unverified": "이 서술은 아직 검증을 마치지 않은 점검 기준으로 확인되어 발급할 수 없습니다. 운영자에게 문의해 주세요.",
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
    recorded = judge.get("narrative_sha256")
    if not isinstance(recorded, str) or not recorded:
        return "judge_narrative_unrecorded"
    if digest.of_content(content) != recorded:
        return "judge_narrative_mismatch"
    lock = judge.get("criteria_lock")
    if lock == "ok":
        return None
    from app.judge import criteria_lock  # 늦은 import — 발급 모듈 import 때 심판 기준 묶음을 끌지 않게

    if lock == "off" and not criteria_lock.enabled():
        return None
    return "judge_criteria_unverified"


def check(db: Session, content: dict | None, job_id: str | None) -> None:
    """판 4 성적서면 작업 번호를 검사한다. 거절이면 IssuanceError(code=사유 코드)."""
    code = reject_reason(db, content, job_id)
    if code is not None:
        raise IssuanceError(code, f"{MESSAGES[code]} ({code})")
