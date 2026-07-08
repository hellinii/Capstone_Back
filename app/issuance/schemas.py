"""app/issuance/schemas.py — 발급 도메인 스키마(기관·발급 요청/응답)."""

from pydantic import BaseModel, Field, field_validator


class OrganizationOut(BaseModel):
    """수행기관(performer) 조회 응답."""
    org_name:   str        = Field(description="수행기관명")
    department: str | None = Field(default=None, description="부서(issuer 조합용)")
    evaluator:  str | None = Field(default=None, description="평가자(performer.evaluator)")
    contact:    str | None = Field(default=None, description="연락처")
    address:    str | None = Field(default=None, description="주소(선택)")


class OrganizationIn(BaseModel):
    """(선택) 기관 정보 수정 요청."""
    org_name:   str        = Field(description="수행기관명")
    department: str | None = Field(default=None, description="부서")
    evaluator:  str | None = Field(default=None, description="평가자")
    contact:    str | None = Field(default=None, description="연락처")
    address:    str | None = Field(default=None, description="주소")


class IssueRequest(BaseModel):
    """[발급] 채번 요청. 같은 run_id 로 재호출 시 신규 채번 없이 기존 발급본 반환(멱등)."""
    run_id:        str        = Field(min_length=1, description="프론트 워크스페이스 run 식별자(멱등 키)")
    model_name:    str | None = Field(default=None, description="대상 모델명")
    model_version: str | None = Field(default=None, description="대상 모델 버전")
    note:          str | None = Field(default=None, description="발급 비고(미지정 시 '최초 발급')")
    issuer:        str | None = Field(default=None, description="발급자(미지정 시 기관 기본값)")

    @field_validator("run_id")
    @classmethod
    def _run_id_not_blank(cls, v: str) -> str:
        # 공백/빈 문자열은 멱등 키로 부적합 — 서로 다른 평가가 한 번호로 병합되는 것을 차단.
        if not v or not v.strip():
            raise ValueError("run_id 는 비어 있을 수 없습니다.")
        return v


class ReissueRequest(BaseModel):
    """[재발급] 정정 발급 요청. 같은 번호 유지 + 버전 차수 증가(v1.0→v1.1)."""
    note:   str        = Field(description="정정 사유(필수)")
    issuer: str | None = Field(default=None, description="발급자(미지정 시 기관 기본값)")


class IssuanceHistoryItem(BaseModel):
    """발급 이력 1건 → signature.history 요소."""
    version:   str        = Field(description="발급 버전")
    issued_at: str        = Field(description="발급 일시(ISO8601)")
    note:      str | None = Field(default=None, description="비고")


class IssuanceOut(BaseModel):
    """발급 결과 — meta.reportId + performer + signature 를 한 번에 채운다."""
    report_no:    str                       = Field(description="성적서 번호 → meta.reportId")
    version:      str                       = Field(description="최신 발급 버전(current_version)")
    issuer:       str                       = Field(description="발급자 → signature.issuer")
    issued_at:    str                       = Field(description="최신 발급 일시(ISO8601) → signature.signedAt")
    organization: OrganizationOut           = Field(description="수행기관 → performer")
    history:      list[IssuanceHistoryItem] = Field(description="발급 이력 → signature.history")
