"""services/issuance.py — 채번·발급·재발급 트랜잭션 로직 (설계 문서 §4).

라우터에서 분리해 단위테스트를 용이하게 한다. 모든 시각은 서버 시계(UTC, tz 없이 저장).
동시성: SQLite 는 쓰기를 직렬화하고, UNIQUE(year, seq) + UNIQUE(run_id) 로 이중 방어한다.
충돌(IntegrityError) 시 seq 를 재계산해 재시도하고, run_id 충돌은 기존 발급본으로 수렴한다.
"""
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.issuance.models import Issuance, Organization, Report

_MAX_NUMBERING_RETRIES = 5
# 동시 쓰기 충돌 시 재시도 대상: UNIQUE 위반(IntegrityError) + 락 경합(OperationalError).
_RETRYABLE_DB_ERRORS = (IntegrityError, OperationalError)


class IssuanceError(Exception):
    """발급 도메인 오류. 라우터가 code 로 적절한 HTTP 상태로 매핑한다."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def bump_version(version: str) -> str:
    """'v1.0' → 'v1.1' (minor 증가). 형식 해석 불가 시 IssuanceError."""
    try:
        core = version.lstrip("vV")
        major, minor = core.split(".")
        return f"v{int(major)}.{int(minor) + 1}"
    except (ValueError, AttributeError):
        raise IssuanceError("bad_version", f"버전 형식을 해석할 수 없습니다: {version!r}")


def get_organization(db: Session) -> Organization:
    """수행기관(singleton) 조회. 없으면 오류."""
    org = db.get(Organization, 1)
    if org is None:
        raise IssuanceError("no_organization", "기관 정보가 시드되지 않았습니다.")
    return org


def _issuer_or_default(db: Session, issuer: str | None) -> str:
    """issuer 미지정 시 기관 기본값('org_name department') 조합."""
    if issuer:
        return issuer
    org = db.get(Organization, 1)
    if org is None:
        return "미지정 기관"
    parts = [org.org_name, org.department]
    return " ".join(p for p in parts if p)


def _next_seq(db: Session, year: int) -> int:
    """해당 연도 최대 seq + 1 (연도별 순번, 연 경계에서 리셋)."""
    max_seq = db.execute(
        select(func.max(Report.seq)).where(Report.year == year)
    ).scalar_one_or_none()
    return (max_seq or 0) + 1


def get_report(db: Session, report_no: str) -> Report | None:
    """번호로 성적서 헤더 조회(재오픈)."""
    return db.execute(
        select(Report).where(Report.report_no == report_no)
    ).scalar_one_or_none()


def issue_report(
    db: Session,
    *,
    run_id: str,
    model_name: str | None = None,
    model_version: str | None = None,
    note: str | None = None,
    issuer: str | None = None,
    now: datetime | None = None,
) -> Report:
    """발급(채번). 같은 run_id 로 이미 발급된 경우 신규 채번 없이 기존 report 반환(멱등)."""
    existing = db.execute(
        select(Report).where(Report.run_id == run_id)
    ).scalar_one_or_none()
    if existing is not None:
        return existing  # 멱등: 중복 채번 방지

    # 기관(report.org_id FK 대상)이 없으면 모든 INSERT 가 FK 위반으로 실패한다.
    # 이를 채번 충돌로 오인하지 않도록 선검사해 정확한 오류를 낸다.
    get_organization(db)

    now = now or _utcnow()
    year = now.year
    resolved_issuer = _issuer_or_default(db, issuer)

    last_err: Exception | None = None
    for _ in range(_MAX_NUMBERING_RETRIES):
        # 매 시도마다 멱등 재확인 — 동시 발급 레이스에서 방금 커밋된 기존본을 포착.
        existing = db.execute(
            select(Report).where(Report.run_id == run_id)
        ).scalar_one_or_none()
        if existing is not None:
            return existing

        seq = _next_seq(db, year)
        report = Report(
            report_no=f"RPT-{year}-{seq:04d}",
            year=year,
            seq=seq,
            run_id=run_id,
            model_name=model_name,
            model_version=model_version,
            org_id=1,
            current_version="v1.0",
            created_at=now,
        )
        report.issuances.append(
            Issuance(
                version="v1.0",
                issuer=resolved_issuer,
                issued_at=now,
                note=note or "최초 발급",
                status="issued",
            )
        )
        db.add(report)
        try:
            db.commit()
            db.refresh(report)
            return report
        except _RETRYABLE_DB_ERRORS as e:
            # UNIQUE(year, seq)/report_no/run_id 충돌 또는 락 경합 → 롤백 후 재수렴.
            db.rollback()
            last_err = e

    raise IssuanceError(
        "numbering_conflict", "채번 충돌이 반복되어 발급에 실패했습니다."
    ) from last_err


def reissue_report(
    db: Session,
    *,
    report_no: str,
    note: str,
    issuer: str | None = None,
    now: datetime | None = None,
) -> Report:
    """재발급(정정). 같은 번호 유지 + 버전 차수 증가. 이전 발급차수는 superseded 로.

    동시 재발급 방어: UNIQUE(report_id, version) 로 같은 버전 중복 커밋을 막고,
    충돌/락 경합 시 롤백 후 current_version 을 다시 읽어 다음 차수로 재수렴한다.
    """
    now = now or _utcnow()
    resolved_issuer = _issuer_or_default(db, issuer)

    last_err: Exception | None = None
    for _ in range(_MAX_NUMBERING_RETRIES):
        report = get_report(db, report_no)
        if report is None:
            raise IssuanceError("not_found", f"성적서 번호를 찾을 수 없습니다: {report_no}")
        if not report.issuances:
            raise IssuanceError("no_prior_issuance", "발급 이력이 없어 재발급할 수 없습니다.")

        prev = report.issuances[-1]  # order_by=Issuance.id → 최신(현행 issued)
        prev.status = "superseded"
        new_version = bump_version(report.current_version)
        report.current_version = new_version
        report.issuances.append(
            Issuance(
                version=new_version,
                issuer=resolved_issuer,
                issued_at=now,
                note=note,
                status="issued",
            )
        )
        try:
            db.commit()
            db.refresh(report)
            return report
        except _RETRYABLE_DB_ERRORS as e:
            # 다른 재발급이 같은 버전을 먼저 커밋(UNIQUE 위반) 또는 락 경합 → 재수렴.
            db.rollback()
            last_err = e

    raise IssuanceError(
        "reissue_conflict", "재발급 충돌이 반복되어 실패했습니다."
    ) from last_err


def update_organization(db: Session, data: dict) -> Organization:
    """(선택) 기관 정보 수정. 없으면 생성(id=1)."""
    org = db.get(Organization, 1)
    if org is None:
        org = Organization(id=1, **data)
        db.add(org)
    else:
        for key, value in data.items():
            setattr(org, key, value)
    db.commit()
    db.refresh(org)
    return org
