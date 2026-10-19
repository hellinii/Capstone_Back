"""app/narrative/models.py — 서술 작업 번호(job) 표 (서술 라운드 S3a).

성적서 서술은 오래 걸릴 수 있어(품질 우선, 최대 10분) 요청 하나로 받지 않고 작업 번호로 나눈다 —
`POST /api/narrative/jobs` 가 번호를 주고, 프론트가 `GET /api/narrative/jobs/{id}` 로 단계와 결과를 본다.
작업 상태는 발급 DB(Neon)의 이 표에 둔다. 서버가 재시작돼도 "어디까지 갔는지"가 남아 유실을 판정할 수 있다.

**열은 이것으로 확정이다.** 이 프로젝트에는 마이그레이션 도구가 없고 `create_all` 은 기존 표에 열을 더하지
못한다(ReportSnapshot 머리말 참조). 앞으로 늘어나는 것은 두 JSON 칸 안에서만 늘린다.

- status        : running | done | failed — 끝 상태는 done·failed 둘. 한 번 끝나면 바뀌지 않는다.
- stage         : queued(대기) · generating(생성 중) · checking(검사 중) · regenerating(다시 만드는 중) ·
                  done(완료) · failed(실패). 프론트가 진행 표시에 쓴다.
- result_json   : 고객용. 만들 때 8.1 벤치마크 표(benchmark_refs)를 넣고, 끝나면 서술 본문과 칸별 출처를
                  더한다. 끝난 뒤 정한 시간이 지나면 지운다(NULL). 사유·토큰·비용은 넣지 않는다.
- internal_json : 내부용(서술 설계 D24 의 "결과 파일"). 실패 종류·강등 사유·호출 기록(응답 모델·거절 종류·
                  토큰·비용)·단계별 시각·슬롯 대기. 결과 글을 지운 뒤에도 남긴다. 입력(사실 시트·고객 입력)은
                  넣지 않는다.
- created_at / updated_at / finished_at : tz 없는 UTC(발급 표와 같은 규칙). updated_at 은 단계가 바뀔 때마다
                  갱신 — 유실 판정의 근거다.

상호작용
- 의존(import): app.core.database(Base)
- 사용처: app.narrative.jobs, app.core.database.init_db(표 등록)
"""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def _utcnow() -> datetime:
    """tz 없는 UTC 현재 시각(발급 표 app.issuance.models._utcnow 와 같은 규칙)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class NarrativeJob(Base):
    """서술 작업 하나. id 는 uuid4 문자열(추측할 수 없는 번호 — 인증이 없는 경로라 순번을 쓰지 않는다)."""
    __tablename__ = "narrative_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    stage: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    internal_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # 정리(결과 글 지우기·오래된 기록 지우기)가 시각으로 찾는다.
    __table_args__ = (
        Index("ix_narrative_jobs_finished_at", "finished_at"),
        Index("ix_narrative_jobs_created_at", "created_at"),
    )
