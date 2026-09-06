"""services/issuance.py — 채번·발급·재발급 트랜잭션 로직 (설계 문서 §4).

라우터에서 분리해 단위테스트를 용이하게 한다. 모든 시각은 서버 시계(UTC, tz 없이 저장).
동시성: SQLite 는 쓰기를 직렬화하고, UNIQUE(year, seq) + UNIQUE(run_id) 로 이중 방어한다.
충돌(IntegrityError) 시 seq 를 재계산해 재시도하고, run_id 충돌은 기존 발급본으로 수렴한다.
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.issuance.models import Issuance, Organization, Report, ReportSnapshot

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


# 채번 연도의 기준 시간대. 성적서에 인쇄되는 발급일이 KST 이므로 번호도 KST 를 따른다.
# KST 는 DST 가 없는 고정 오프셋(+09:00)이라 zoneinfo/tzdata 에 의존하지 않는다
# (배포 이미지에 tzdata 가 없어도 동작한다).
KST = timezone(timedelta(hours=9))


def _numbering_year(now: datetime) -> int:
    """채번 연도(KST 기준). now 는 tz 없는 UTC(_utcnow 규약).

    UTC 연도를 쓰면 매년 12/31 15:00~24:00 UTC(= 1/1 00:00~09:00 KST)에 발급한 문서가
    **번호는 전년도, 인쇄된 발급일은 새해**가 되어 연도별 채번 대장과 대조되지 않는다
    (ISSUES.md F-07).
    """
    return now.replace(tzinfo=timezone.utc).astimezone(KST).year


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


def _canonical_json(obj) -> str:
    """해시·저장용 정규 직렬화. 키 순서를 고정해 같은 내용이 같은 해시를 갖게 한다."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _org_snapshot_dict(org: Organization) -> dict:
    """발급 시점에 동결할 기관 필드."""
    return {
        "org_name": org.org_name,
        "department": org.department,
        "evaluator": org.evaluator,
        "contact": org.contact,
        "address": org.address,
    }


def _build_snapshot(org: Organization, version: str, content: dict | None) -> ReportSnapshot:
    """발급 차수 하나에 대응하는 스냅샷 행을 만든다.

    content 가 없어도 행은 만든다 — 기관 스냅샷(G-01)은 내용 전송 여부와 무관하게
    항상 필요하기 때문이다.
    """
    content_json = _canonical_json(content) if content is not None else None
    content_hash = (
        hashlib.sha256(content_json.encode("utf-8")).hexdigest()
        if content_json is not None else None
    )
    return ReportSnapshot(
        version=version,
        content_json=content_json,
        content_hash=content_hash,
        org_snapshot_json=_canonical_json(_org_snapshot_dict(org)),
    )


def get_snapshot(db: Session, report: Report, version: str | None = None) -> ReportSnapshot | None:
    """해당 차수(미지정 시 최신)의 스냅샷. 없으면 None."""
    target = version or report.current_version
    for snap in report.snapshots:
        if snap.version == target:
            return snap
    return None


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
    content: dict | None = None,
    now: datetime | None = None,
) -> Report:
    """발급(채번). 같은 run_id 로 이미 발급된 경우 신규 채번 없이 기존 report 반환(멱등).

    발급 시점의 성적서 원본(content)과 기관 정보를 report_snapshot 에 함께 보관한다
    (ISSUES.md F-01·G-01). content 는 선택이지만 기관 스냅샷은 항상 기록된다.
    """
    existing = db.execute(
        select(Report).where(Report.run_id == run_id)
    ).scalar_one_or_none()
    if existing is not None:
        return existing  # 멱등: 중복 채번 방지

    # 기관(report.org_id FK 대상)이 없으면 모든 INSERT 가 FK 위반으로 실패한다.
    # 이를 채번 충돌로 오인하지 않도록 선검사해 정확한 오류를 낸다.
    org = get_organization(db)

    now = now or _utcnow()
    year = _numbering_year(now)
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
        report.snapshots.append(_build_snapshot(org, "v1.0", content))
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
    run_id: str | None = None,
    issuer: str | None = None,
    content: dict | None = None,
    now: datetime | None = None,
) -> Report:
    """재발급(정정). 같은 번호 유지 + 버전 차수 증가. 이전 발급차수는 superseded 로.

    소지 증명 — run_id 가 주어지면 해당 성적서의 run_id 와 일치해야 한다(ISSUES.md G-02).
    report_no 는 RPT-{year}-{seq:04d} 로 전수 열거가 가능하지만 run_id 는 프론트가
    crypto.randomUUID 로 만들어 추측할 수 없다. 이것이 없으면 번호만 아는 제3자가
    남의 성적서를 superseded 로 만들고 임의 사유의 정정 이력을 남길 수 있다.

    동시 재발급 방어: UNIQUE(report_id, version) 로 같은 버전 중복 커밋을 막고,
    충돌/락 경합 시 롤백 후 current_version 을 다시 읽어 다음 차수로 재수렴한다.
    """
    now = now or _utcnow()
    resolved_issuer = _issuer_or_default(db, issuer)
    org = get_organization(db)

    last_err: Exception | None = None
    for _ in range(_MAX_NUMBERING_RETRIES):
        report = get_report(db, report_no)
        if report is None:
            raise IssuanceError("not_found", f"성적서 번호를 찾을 수 없습니다: {report_no}")
        if not report.issuances:
            raise IssuanceError("no_prior_issuance", "발급 이력이 없어 재발급할 수 없습니다.")
        if run_id is not None and report.run_id != run_id:
            raise IssuanceError(
                "forbidden", "이 성적서를 재발급할 권한이 없습니다(run 식별자 불일치)."
            )

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
        # 이전 차수의 스냅샷은 건드리지 않는다 — 정정 전후를 대조할 수 있어야 한다(F-06).
        report.snapshots.append(_build_snapshot(org, new_version, content))
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
    """기관 정보 수정. 없으면 생성(id=1).

    **HTTP 로 노출되지 않는다.** 무인증 상태에서 `PUT /api/organization` 하나로 이미
    발급된 모든 성적서의 기관 표기가 소급 변경됐기 때문에 엔드포인트를 제거했다
    (ISSUES.md G-01). 시드·운영 스크립트·테스트에서만 호출한다.

    발급된 성적서는 이제 발급 시점 기관을 report_snapshot 에 동결하므로, 이 함수로
    기관을 바꿔도 기존 발급본의 표기는 바뀌지 않는다(신규 발급분부터 반영).
    """
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
