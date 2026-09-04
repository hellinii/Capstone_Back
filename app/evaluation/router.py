"""app/evaluation/router.py — 평가(지표 계산) API 라우터 (얇은 HTTP 계층)

업로드 파일과 매핑 설정(EvaluateRequest JSON)을 받아 파싱·검증만 하고, 실제 평가
파이프라인은 evaluation.service.run_evaluation_pipeline 에 위임한다. EvaluationError.code 를
HTTP 상태코드로 매핑한다. (prefix=/api, tags=["Evaluation"], POST /api/evaluate.)

상호작용
- 의존(import): app.core.schemas(EvaluateRequest/Response), app.core.parsing(parse_file_content),
  app.evaluation.service(run_evaluation_pipeline, EvaluationError)
- 사용처: app.main(evaluate_router로 등록)
"""
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.evaluation.schemas import EvaluateRequest, EvaluateResponse
from app.core.parsing import parse_file_content
from app.evaluation.service import run_evaluation_pipeline, EvaluationError

router = APIRouter(prefix="/api", tags=["Evaluation"])


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="평가 연산 실행",
    description=(
        "업로드된 데이터셋 파일과 매핑 설정을 받아, "
        "선택된 ISO/IEC 4213 평가지표를 계산하여 결과를 반환합니다."
    )
)
async def evaluate_dataset(
    file: UploadFile = File(..., description="평가할 데이터셋 파일 (.csv 또는 .json)"),
    data: str = Form(..., description="EvaluateRequest 데이터의 JSON 문자열")
) -> EvaluateResponse:
    # 1. 파일 읽기 (HTTP 경계)
    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="빈 파일은 처리할 수 없습니다.")

    # 2. 파일 파싱
    filename = file.filename or ""
    try:
        _, df = parse_file_content(file_content, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"파일 파싱 실패: {str(e)}")

    # 3. 요청 데이터 파싱 및 검증 (EvaluateRequest)
    try:
        request_data = EvaluateRequest.model_validate_json(data)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"설정 데이터 파싱 실패. 올바른 EvaluateRequest 형식이어야 합니다. 상세 에러: {str(e)}"
        )

    # 4. 평가 파이프라인은 서비스에 위임 (도메인 오류 → 상태코드 매핑)
    try:
        return run_evaluation_pipeline(df, request_data)
    except EvaluationError as e:
        status_code = 500 if e.code == "compute_error" else 400
        raise HTTPException(status_code=status_code, detail=e.message)
