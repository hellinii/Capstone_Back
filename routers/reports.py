"""routers/reports.py — 발급/조회 API (설계 문서 §5).

- GET  /api/organization                  수행기관(performer) 조회
- POST /api/reports/issue                  발급(채번) — run_id 멱등
- POST /api/reports/{report_no}/reissue    재발급(정정) — 버전업
- GET  /api/reports/{report_no}            발급정보 조회(재오픈)
- PUT  /api/organization                   (선택) 기관 정보 수정
"""
from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import Organization, Report
from schemas import (
    IssuanceHistoryItem,
    IssuanceOut,
    IssueRequest,
    OrganizationIn,
    OrganizationOut,
    ReissueRequest,
)
from services import issuance as issuance_service
from services.issuance import IssuanceError

router = APIRouter(prefix="/api", tags=["Reports"])


def _org_out(org: Organization) -> OrganizationOut:
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


def _issuance_out(report: Report) -> IssuanceOut:
    """Report ORM → IssuanceOut (meta.reportId + performer + signature)."""
    latest = report.issuances[-1]
    return IssuanceOut(
        report_no=report.report_no,
        version=report.current_version,
        issuer=latest.issuer,
        issued_at=_iso_utc(latest.issued_at),
        organization=_org_out(report.organization),
        history=[
            IssuanceHistoryItem(
                version=i.version,
                issued_at=_iso_utc(i.issued_at),
                note=i.note,
            )
            for i in report.issuances
        ],
    )


@router.get("/organization", response_model=OrganizationOut)
def get_organization(db: Session = Depends(get_db)):
    """수행기관(performer) 조회."""
    try:
        org = issuance_service.get_organization(db)
    except IssuanceError as e:
        raise HTTPException(status_code=404, detail=e.message)
    return _org_out(org)


@router.put("/organization", response_model=OrganizationOut)
def update_organization(payload: OrganizationIn, db: Session = Depends(get_db)):
    """(선택) 기관 정보 수정. 부분 업데이트 — 요청에 없는 필드는 기존 값을 유지(NULL 덮어쓰기 방지)."""
    org = issuance_service.update_organization(db, payload.model_dump(exclude_unset=True))
    return _org_out(org)


@router.post("/reports/issue", response_model=IssuanceOut)
def issue(payload: IssueRequest, db: Session = Depends(get_db)):
    """발급(채번). 같은 run_id 재호출 시 기존 발급본 반환(멱등)."""
    try:
        report = issuance_service.issue_report(
            db,
            run_id=payload.run_id,
            model_name=payload.model_name,
            model_version=payload.model_version,
            note=payload.note,
            issuer=payload.issuer,
        )
    except IssuanceError as e:
        raise HTTPException(status_code=409, detail=e.message)
    return _issuance_out(report)


@router.post("/reports/{report_no}/reissue", response_model=IssuanceOut)
def reissue(report_no: str, payload: ReissueRequest, db: Session = Depends(get_db)):
    """재발급(정정). 같은 번호 유지 + 버전 차수 증가."""
    try:
        report = issuance_service.reissue_report(
            db, report_no=report_no, note=payload.note, issuer=payload.issuer
        )
    except IssuanceError as e:
        status = 404 if e.code == "not_found" else 409
        raise HTTPException(status_code=status, detail=e.message)
    return _issuance_out(report)


@router.get("/reports/{report_no}", response_model=IssuanceOut)
def get_report(report_no: str, db: Session = Depends(get_db)):
    """발급정보 조회(재오픈)."""
    report = issuance_service.get_report(db, report_no)
    if report is None:
        raise HTTPException(
            status_code=404, detail=f"성적서 번호를 찾을 수 없습니다: {report_no}"
        )
    return _issuance_out(report)
