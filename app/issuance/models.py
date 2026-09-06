"""models.py — 발급 메타 ORM 모델 (설계 문서 §3).

정규화: organization(singleton) 1 : N report(성적서 헤더=채번 단위) 1 : N issuance(발급 차수).
"""
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    """발급 시각 등에 사용하는 현재 시각(서버 시계, tz 없는 UTC 로 저장·직렬화)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Organization(Base):
    """수행기관 — singleton(항상 id=1). 프론트 performer 로 매핑."""
    __tablename__ = "organization"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # singleton, 항상 1
    org_name: Mapped[str] = mapped_column(String, nullable=False)
    department: Mapped[str | None] = mapped_column(String, nullable=True)  # issuer 조합용 ("평가부")
    evaluator: Mapped[str | None] = mapped_column(String, nullable=True)   # performer.evaluator
    contact: Mapped[str | None] = mapped_column(String, nullable=True)
    address: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )


class Report(Base):
    """성적서 헤더 — 채번(report_no) 단위. 같은 run_id 재발급 시 동일 report 재사용."""
    __tablename__ = "report"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_no: Mapped[str] = mapped_column(String, unique=True, nullable=False)  # "RPT-2026-0001"
    year: Mapped[int] = mapped_column(Integer, nullable=False)  # 채번 연도
    seq: Mapped[int] = mapped_column(Integer, nullable=False)   # 연도 내 순번
    # 평가 연결(멱등 키). UNIQUE 로 동시 발급 시 같은 run 중복 채번을 원천 차단(설계 §11.4).
    run_id: Mapped[str | None] = mapped_column(String, nullable=True, unique=True, index=True)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    org_id: Mapped[int] = mapped_column(
        ForeignKey("organization.id"), nullable=False, default=1
    )
    current_version: Mapped[str] = mapped_column(String, nullable=False, default="v1.0")  # 최신 발급 버전
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    organization: Mapped["Organization"] = relationship()
    issuances: Mapped[list["Issuance"]] = relationship(
        back_populates="report",
        order_by="Issuance.id",
        cascade="all, delete-orphan",
    )
    snapshots: Mapped[list["ReportSnapshot"]] = relationship(
        back_populates="report",
        order_by="ReportSnapshot.id",
        cascade="all, delete-orphan",
    )

    # 채번 충돌 이중 방어: 같은 연도에 동일 seq 두 번 커밋 불가.
    __table_args__ = (UniqueConstraint("year", "seq", name="uq_report_year_seq"),)


class ReportSnapshot(Base):
    """발급 시점의 성적서 원본 + 기관 스냅샷 — 발급 차수(version)마다 한 행.

    왜 **신규 테이블**인가 — 기존 report/issuance 테이블에 컬럼을 추가하면 스키마
    마이그레이션 도구가 필요하다. `Base.metadata.create_all` 은 기존 테이블에 컬럼을
    추가하지 못하고(앱은 기동한 뒤 첫 SELECT 에서 죽는다), 이 프로젝트에는 Alembic 이
    없다(ISSUES.md F-11). 반면 **신규 테이블은 create_all 이 정상 생성**하므로,
    마이그레이션 인프라 없이 F-01 을 닫을 수 있다(실증 확인).

    보관 형식은 dialect 중립을 위해 JSON 문자열(Text)이다. SQLite/PostgreSQL 어느
    쪽에서도 같은 DDL 로 동작한다.

    - content_json    : 발급 시 클라이언트가 보낸 성적서 원본(FinalReportData). 선택.
    - content_hash    : content_json 의 SHA-256. 인쇄물 진위 대조용.
    - org_snapshot_json: **발급 시점**의 수행기관 정보. 이것이 없으면 조회 시점의
                        기관 행을 조인하게 되어, 기관명을 바꾸는 순간 이미 발급된
                        모든 성적서의 기관 표기가 소급 변경된다(ISSUES.md G-01).
    """
    __tablename__ = "report_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(
        ForeignKey("report.id"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String, nullable=False)  # "v1.0" — Issuance.version 대응
    content_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    org_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    report: Mapped["Report"] = relationship(back_populates="snapshots")

    # 같은 성적서의 같은 차수에 스냅샷이 두 번 생기지 않도록.
    __table_args__ = (
        UniqueConstraint("report_id", "version", name="uq_snapshot_report_version"),
    )


class Issuance(Base):
    """발급 차수(이력) — 최초 발급 v1.0, 정정 발급 시 v1.1 … 로 누적."""
    __tablename__ = "issuance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(
        ForeignKey("report.id"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String, nullable=False)  # "v1.0", "v1.1" …
    issuer: Mapped[str] = mapped_column(String, nullable=False)   # "한국 AI 인증원 평가부"
    issued_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    note: Mapped[str | None] = mapped_column(String, nullable=True)  # "최초 발급" / "정정 발급: …"
    status: Mapped[str] = mapped_column(String, nullable=False, default="issued")  # issued | superseded

    report: Mapped["Report"] = relationship(back_populates="issuances")

    # 같은 성적서에 동일 버전 두 번 불가 → 동시 재발급이 중복 버전을 커밋하지 못하게 방어.
    __table_args__ = (
        UniqueConstraint("report_id", "version", name="uq_issuance_report_version"),
    )
