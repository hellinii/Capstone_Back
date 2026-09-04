"""routers/reports.py — 발급/조회 API (설계 문서 §5).

- GET  /api/organization                  수행기관(performer) 조회
- POST /api/reports/issue                  발급(채번) — run_id 멱등
- POST /api/reports/{report_no}/reissue    재발급(정정) — 버전업
- GET  /api/reports/{report_no}            발급정보 조회(재오픈)
- PUT  /api/organization                   (선택) 기관 정보 수정
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.issuance.schemas import IssuanceOut, IssueRequest, OrganizationIn, OrganizationOut, ReissueRequest
from app.issuance import service as issuance_service
from app.issuance.service import IssuanceError
from app.issuance.serializers import organization_out, issuance_out

router = APIRouter(prefix="/api", tags=["Reports"])


@router.get("/organization", response_model=OrganizationOut)
def get_organization(db: Session = Depends(get_db)):
    """수행기관(performer) 조회."""
    try:
        org = issuance_service.get_organization(db)
    except IssuanceError as e:
        raise HTTPException(status_code=404, detail=e.message)
    return organization_out(org)


@router.put("/organization", response_model=OrganizationOut)
def update_organization(payload: OrganizationIn, db: Session = Depends(get_db)):
    """(선택) 기관 정보 수정. 부분 업데이트 — 요청에 없는 필드는 기존 값을 유지(NULL 덮어쓰기 방지)."""
    org = issuance_service.update_organization(db, payload.model_dump(exclude_unset=True))
    return organization_out(org)


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
    return issuance_out(report)


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
    return issuance_out(report)


@router.get("/reports/{report_no}", response_model=IssuanceOut)
def get_report(report_no: str, db: Session = Depends(get_db)):
    """발급정보 조회(재오픈)."""
    report = issuance_service.get_report(db, report_no)
    if report is None:
        raise HTTPException(
            status_code=404, detail=f"성적서 번호를 찾을 수 없습니다: {report_no}"
        )
    return issuance_out(report)
