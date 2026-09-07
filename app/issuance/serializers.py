"""app/issuance/serializers.py — 발급 ORM → 응답 스키마 직렬화(presenter)

Report/Organization ORM 객체를 API 응답 Pydantic 모델로 변환한다. DB 접근이 없는 순수
변환 계층이라 라우터에서 분리해 라우터가 HTTP·상태코드에만 집중하게 한다.

상호작용
- 의존(import): app.issuance.models(Organization, Report), app.core.schemas(OrganizationOut,
  IssuanceOut, IssuanceHistoryItem)
- 사용처: app.issuance.router (엔드포인트 응답 조립)
"""
import json
from datetime import timezone

from app.issuance.models import Organization, Report
from app.issuance.schemas import (
    IssuanceHistoryItem,
    IssuanceOut,
    OrganizationOut,
    ReportContentOut,
)


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


def _organization_at_issue(report: Report) -> OrganizationOut:
    """**발급 시점**의 기관 정보. 스냅샷이 있으면 그것을, 없으면 현재 기관 행을 쓴다.

    종전에는 항상 `report.organization`(FK 조인 = 조회 시점의 singleton 행)을 조립했다.
    그래서 기관명을 한 번 바꾸면 **이미 발급된 모든 성적서의 기관 표기가 소급 변경**됐고,
    같은 응답 안에서 issuer(발급 당시 스냅샷)와 organization(현재 값)이 서로 다른
    시점을 가리켰다(ISSUES.md G-01).

    스냅샷이 없는 구 발급본은 현재 기관으로 폴백한다 — 소급 백필은 하지 않는다.
    서버에 없던 사실을 만들어 넣지 않는다는 원칙(설계 §10 '가짜데이터 0')에 따른 것이고,
    폴백 덕분에 기존 발급본의 표시는 지금과 동일하게 유지된다.
    """
    for snap in reversed(report.snapshots):
        if snap.version == report.current_version and snap.org_snapshot_json:
            data = json.loads(snap.org_snapshot_json)
            return OrganizationOut(**data)
    return organization_out(report.organization)


def diff_report_sections(before: dict | None, after: dict | None) -> list[str] | None:
    """두 성적서 스냅샷 사이에서 값이 달라진 **최상위 절**을 돌려준다 (ISSUES.md F-06).

    재발급 이력에는 "무엇이 바뀌었는지"가 없었다 — 정정본을 받은 사람은 v1.1 이
    나왔다는 사실만 알고 무엇이 달라졌는지 알 수 없었다.

    **새 컬럼을 만들지 않는다.** 필요한 데이터는 이미 서버에 있고(차수별 스냅샷, F-01)
    `create_all` 은 기존 테이블에 컬럼을 추가하지 못하므로(F-11) 읽는 시점에 만든다.

    이전 차수 스냅샷이 없으면(이 라운드 이전 발급본은 소급 백필을 하지 않았다)
    None 을 돌려준다 — 모르는 것을 '변경 없음'으로 말하지 않는다.
    """
    if before is None or after is None:
        return None
    keys = set(before) | set(after)
    return sorted(k for k in keys if before.get(k) != after.get(k))


def _parse_content(content_json: str | None) -> dict | None:
    """스냅샷 본문(JSON 텍스트) → dict. 깨진 값은 '모름'으로 취급한다."""
    if not content_json:
        return None
    try:
        parsed = json.loads(content_json)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _history(report: Report) -> list[IssuanceHistoryItem]:
    """발급 이력 — 각 차수에 **직전 차수 대비 바뀐 절**을 함께 싣는다(ISSUES.md F-06)."""
    by_version = {s.version: _parse_content(s.content_json) for s in report.snapshots}
    items: list[IssuanceHistoryItem] = []
    previous_version: str | None = None

    for issuance in report.issuances:
        changed = (
            None
            if previous_version is None
            else diff_report_sections(
                by_version.get(previous_version), by_version.get(issuance.version)
            )
        )
        items.append(IssuanceHistoryItem(
            version=issuance.version,
            issued_at=_iso_utc(issuance.issued_at),
            note=issuance.note,
            changed_sections=changed,
        ))
        previous_version = issuance.version

    return items


def issuance_out(report: Report) -> IssuanceOut:
    """Report ORM → IssuanceOut (meta.reportId + performer + signature)."""
    latest = report.issuances[-1]
    return IssuanceOut(
        report_no=report.report_no,
        version=report.current_version,
        issuer=latest.issuer,
        issued_at=_iso_utc(latest.issued_at),
        organization=_organization_at_issue(report),
        history=_history(report),
    )


def report_content_out(report: Report, snapshot) -> ReportContentOut:
    """ReportSnapshot ORM → ReportContentOut. 호출부가 content 존재를 이미 확인한다."""
    issued_at = next(
        (i.issued_at for i in report.issuances if i.version == snapshot.version),
        report.issuances[-1].issued_at,
    )
    return ReportContentOut(
        report_no=report.report_no,
        version=snapshot.version,
        issued_at=_iso_utc(issued_at),
        content_hash=snapshot.content_hash or "",
        content=json.loads(snapshot.content_json),
    )
