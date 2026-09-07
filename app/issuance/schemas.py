"""app/issuance/schemas.py — 발급 도메인 스키마(기관·발급 요청/응답)."""

import json

from pydantic import BaseModel, Field, field_validator

# 발급 요청에 실리는 성적서 원본(FinalReportData)의 크기 상한.
# 통상 수십 KB 다 — ROC/PR 곡선은 60점으로 다운샘플되고 datasetSamples 는 비어 있다.
# 이 API 는 무인증이므로 상한 없이 받으면 임의 크기 JSON 이 DB 에 영구 저장된다
# (ISSUES.md G-03·G-04 와 같은 뿌리). 통상값의 약 20배로 여유 있게 잡는다.
MAX_CONTENT_BYTES = 1_048_576  # 1 MiB


def _validate_content_size(v: dict | None) -> dict | None:
    if v is None:
        return None
    size = len(json.dumps(v, ensure_ascii=False).encode("utf-8"))
    if size > MAX_CONTENT_BYTES:
        raise ValueError(
            f"성적서 내용이 너무 큽니다({size:,} bytes). "
            f"상한은 {MAX_CONTENT_BYTES:,} bytes 입니다."
        )
    return v


class OrganizationOut(BaseModel):
    """수행기관(performer) 조회 응답."""
    org_name:   str        = Field(description="수행기관명")
    department: str | None = Field(default=None, description="부서(issuer 조합용)")
    evaluator:  str | None = Field(default=None, description="평가자(performer.evaluator)")
    contact:    str | None = Field(default=None, description="연락처")
    address:    str | None = Field(default=None, description="주소(선택)")


class IssueRequest(BaseModel):
    """[발급] 채번 요청. 같은 run_id 로 재호출 시 신규 채번 없이 기존 발급본 반환(멱등)."""
    run_id:        str        = Field(min_length=1, description="프론트 워크스페이스 run 식별자(멱등 키)")
    model_name:    str | None = Field(default=None, description="대상 모델명")
    model_version: str | None = Field(default=None, description="대상 모델 버전")
    note:          str | None = Field(default=None, description="발급 비고(미지정 시 '최초 발급')")
    issuer:        str | None = Field(default=None, description="발급자(미지정 시 기관 기본값)")
    content:       dict | None = Field(
        default=None,
        description="발급 시점의 성적서 원본(FinalReportData). 서버가 그대로 보관해 "
                    "번호로 복원·진위 대조할 수 있게 한다. 선택(미전송 시 메타만 저장).",
    )

    @field_validator("run_id")
    @classmethod
    def _run_id_not_blank(cls, v: str) -> str:
        # 공백/빈 문자열은 멱등 키로 부적합 — 서로 다른 평가가 한 번호로 병합되는 것을 차단.
        if not v or not v.strip():
            raise ValueError("run_id 는 비어 있을 수 없습니다.")
        return v

    @field_validator("content")
    @classmethod
    def _content_size(cls, v: dict | None) -> dict | None:
        return _validate_content_size(v)


class ReissueRequest(BaseModel):
    """[재발급] 정정 발급 요청. 같은 번호 유지 + 버전 차수 증가(v1.0→v1.1)."""
    run_id: str        = Field(
        min_length=1,
        description="발급 시 사용한 run 식별자 — 소지 증명. report_no 는 순차 채번이라 "
                    "전수 열거가 가능하지만 run_id 는 randomUUID 라 추측할 수 없다.",
    )
    note:   str        = Field(min_length=1, description="정정 사유(필수)")
    issuer: str | None = Field(default=None, description="발급자(미지정 시 기관 기본값)")
    content: dict | None = Field(
        default=None,
        description="정정된 성적서 원본. 이전 차수의 스냅샷은 보존된다(정정 전후 대조용).",
    )

    @field_validator("run_id")
    @classmethod
    def _run_id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("run_id 는 비어 있을 수 없습니다.")
        return v

    @field_validator("content")
    @classmethod
    def _content_size(cls, v: dict | None) -> dict | None:
        return _validate_content_size(v)

    @field_validator("note")
    @classmethod
    def _note_not_blank(cls, v: str) -> str:
        # 재발급은 이전 차수를 superseded 로 만들고 이력 행을 남기며, 그 이력은
        # SignatureSection 을 통해 성적서에 그대로 인쇄된다. 사유가 공란이면 제3자가
        # 무엇이 왜 정정됐는지 판별할 근거가 사라진다(ISSUES.md F-06).
        # UI 는 ReportLayout.tsx 에서 이미 막고 있었으나 API 는 통과했다.
        if not v.strip():
            raise ValueError("정정 사유(note)는 비어 있을 수 없습니다.")
        return v.strip()


class IssuanceHistoryItem(BaseModel):
    """발급 이력 1건 → signature.history 요소."""
    version:   str        = Field(description="발급 버전")
    issued_at: str        = Field(description="발급 일시(ISO8601)")
    note:      str | None = Field(default=None, description="비고")
    changed_sections: list[str] | None = Field(
        default=None,
        description=(
            "직전 차수 대비 값이 달라진 성적서 최상위 절 목록(ISSUES.md F-06). "
            "최초 발급이거나 이전 차수 스냅샷이 없으면 null — 모르는 것을 '변경 없음'으로 "
            "말하지 않는다(이 라운드 이전 발급본은 소급 백필하지 않았다)."
        ),
    )


class IssuanceOut(BaseModel):
    """발급 결과 — meta.reportId + performer + signature 를 한 번에 채운다."""
    report_no:    str                       = Field(description="성적서 번호 → meta.reportId")
    version:      str                       = Field(description="최신 발급 버전(current_version)")
    issuer:       str                       = Field(description="발급자 → signature.issuer")
    issued_at:    str                       = Field(description="최신 발급 일시(ISO8601) → signature.signedAt")
    organization: OrganizationOut           = Field(description="수행기관 → performer")
    history:      list[IssuanceHistoryItem] = Field(description="발급 이력 → signature.history")


class ReportContentOut(BaseModel):
    """발급 시점 성적서 원본 조회 응답 — 번호만으로 문서를 복원·대조한다."""
    report_no:    str  = Field(description="성적서 번호")
    version:      str  = Field(description="이 내용이 속한 발급 차수")
    issued_at:    str  = Field(description="해당 차수의 발급 일시(ISO8601)")
    content_hash: str  = Field(description="content 의 SHA-256(진위 대조용)")
    content:      dict = Field(description="발급 시점의 성적서 원본(FinalReportData)")
