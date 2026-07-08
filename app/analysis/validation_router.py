"""app/analysis/validation_router.py — 데이터 검증(전처리 dry-run) API 라우터 (얇은 HTTP 계층)

업로드 파일과 매핑 설정(EvaluateRequest JSON)을 받아 파일/요청 파싱과 예외→상태코드 매핑만
담당하고, 실제 검증 파이프라인은 validation_service.validate_dataset 에 위임한다.
(prefix=/api, tags=["Data Validation"], POST /api/validate-data — 메트릭 계산은 하지 않음.)

상호작용
- 의존(import): app.core.schemas(EvaluateRequest, ValidateDataResponse),
  app.core.parsing(parse_file_content), app.analysis.validation_service(validate_dataset)
- 사용처: app.main(validate_router로 등록)
"""
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.analysis.schemas import ValidateDataResponse
from app.evaluation.schemas import EvaluateRequest
from app.core.parsing import parse_file_content
from app.analysis.validation_service import validate_dataset

router = APIRouter(prefix="/api", tags=["Data Validation"])


@router.post(
    "/validate-data",
    response_model=ValidateDataResponse,
    summary="데이터 검증 (전처리 dry-run)",
    description=(
        "업로드된 데이터셋 파일과 매핑 설정을 받아, "
        "전처리(preprocessor) 검증만 수행하여 결과를 반환합니다. "
        "실제 메트릭 계산은 수행하지 않습니다."
    ),
)
async def validate_data(
    file: UploadFile = File(..., description="검증할 데이터셋 파일 (.csv 또는 .json)"),
    data: str = Form(..., description="EvaluateRequest 데이터의 JSON 문자열"),
) -> ValidateDataResponse:
    # 1. 파일 파싱 (HTTP 경계)
    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="빈 파일은 처리할 수 없습니다.")

    filename = file.filename or ""
    try:
        _, df = parse_file_content(file_content, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"파일 파싱 실패: {str(e)}")

    # 2. 요청 데이터 파싱 (HTTP 경계)
    try:
        request_data = EvaluateRequest.model_validate_json(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"설정 데이터 파싱 실패: {str(e)}")

    # 3. 검증 파이프라인은 서비스에 위임
    return validate_dataset(df, request_data)
