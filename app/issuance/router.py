"""routers/reports.py — 발급/조회 API (설계 문서 §5).

- GET  /api/organization                  수행기관(performer) 조회
- POST /api/reports/issue                  발급(채번) — run_id 멱등
- POST /api/reports/{report_no}/reissue    재발급(정정) — 버전업
- GET  /api/reports/{report_no}            발급정보 조회(재오픈)
- GET  /api/reports/{report_no}/content    발급 시점 성적서 원본 조회(복원·진위 대조)

PUT /api/organization 은 제거했다 — 무인증 상태에서 이 엔드포인트 하나로 이미 발급된
모든 성적서의 기관 표기가 소급 변경됐고(ISSUES.md G-01), 프론트는 이 API 를 한 번도
호출하지 않는다(issuanceApi.ts 전수 확인). 기관 정보는 bootstrap 시드로만 바꾼다.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.issuance.schemas import (
    IssuanceOut,
    IssueRequest,
    OrganizationOut,
    ReissueRequest,
    ReportContentOut,
)
from app.issuance import service as issuance_service
from app.issuance.service import IssuanceError
from app.issuance.serializers import organization_out, issuance_out, report_content_out

router = APIRouter(prefix="/api", tags=["Reports"])


@router.get("/organization", response_model=OrganizationOut)
def get_organization(db: Session = Depends(get_db)):
    """수행기관(performer) 조회."""
    try:
        org = issuance_service.get_organization(db)
    except IssuanceError as e:
        raise HTTPException(status_code=404, detail=e.message)
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
            content=payload.content,
        )
    except IssuanceError as e:
        raise HTTPException(status_code=409, detail=e.message)
    return issuance_out(report)


@router.post("/reports/{report_no}/reissue", response_model=IssuanceOut)
def reissue(report_no: str, payload: ReissueRequest, db: Session = Depends(get_db)):
    """재발급(정정). 같은 번호 유지 + 버전 차수 증가."""
    try:
        report = issuance_service.reissue_report(
            db,
            report_no=report_no,
            note=payload.note,
            run_id=payload.run_id,
            issuer=payload.issuer,
            content=payload.content,
        )
    except IssuanceError as e:
        status = {"not_found": 404, "forbidden": 403}.get(e.code, 409)
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


@router.get("/reports/{report_no}/content", response_model=ReportContentOut)
def get_report_content(
    report_no: str, version: str | None = None, db: Session = Depends(get_db)
):
    """발급 시점 성적서 원본 조회.

    version 미지정 시 최신 차수. 정정 발급이 있었어도 이전 차수를 지정해 받을 수 있어
    무엇이 바뀌었는지 대조할 수 있다(ISSUES.md F-01·F-04·F-06).
    """
    report = issuance_service.get_report(db, report_no)
    if report is None:
        raise HTTPException(
            status_code=404, detail=f"성적서 번호를 찾을 수 없습니다: {report_no}"
        )

    snapshot = issuance_service.get_snapshot(db, report, version)
    if snapshot is None or snapshot.content_json is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"이 성적서({report_no})에는 서버에 보관된 내용이 없습니다. "
                "발급 시 성적서 원본을 함께 전송하지 않았거나, 내용 보관 도입 이전에 발급된 문서입니다."
            ),
        )
    return report_content_out(report, snapshot)
