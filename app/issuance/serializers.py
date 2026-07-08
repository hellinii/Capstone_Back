"""app/issuance/serializers.py — 발급 ORM → 응답 스키마 직렬화(presenter)

Report/Organization ORM 객체를 API 응답 Pydantic 모델로 변환한다. DB 접근이 없는 순수
변환 계층이라 라우터에서 분리해 라우터가 HTTP·상태코드에만 집중하게 한다.

상호작용
- 의존(import): app.issuance.models(Organization, Report), app.core.schemas(OrganizationOut,
  IssuanceOut, IssuanceHistoryItem)
- 사용처: app.issuance.router (엔드포인트 응답 조립)
"""
from datetime import timezone

from app.issuance.models import Organization, Report
from app.core.schemas import IssuanceHistoryItem, IssuanceOut, OrganizationOut


def organization_out(org: Organization) -> OrganizationOut:
    return OrganizationOut(
        org_name=org.org_name,
        department=org.department,
        evaluator=org.evaluator,
        contact=org.contact,
        address=org.address,
    )


def _iso_utc(dt) -> str:
    """저장된 naive UTC 시각을 offset 포함 ISO8601 로 방출('...+00:00').

    naive isoformat 은 오프셋이 없어 프론트가 로컬(KST)로 오해하면 하루가 어긋난다
    (KST 오전 발급이 전날로 표기). offset 을 명시해 프론트(Phase D)가 KST 로 정확히
    변환·표기하도록 한다. 표시 포맷은 프론트 책임(단일 출처=백엔드 데이터).
    """
    return dt.replace(tzinfo=timezone.utc).isoformat()


def issuance_out(report: Report) -> IssuanceOut:
    """Report ORM → IssuanceOut (meta.reportId + performer + signature)."""
    latest = report.issuances[-1]
    return IssuanceOut(
        report_no=report.report_no,
        version=report.current_version,
        issuer=latest.issuer,
        issued_at=_iso_utc(latest.issued_at),
        organization=organization_out(report.organization),
        history=[
            IssuanceHistoryItem(
                version=i.version,
                issued_at=_iso_utc(i.issued_at),
                note=i.note,
            )
            for i in report.issuances
        ],
    )
