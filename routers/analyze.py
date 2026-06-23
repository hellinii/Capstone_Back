from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request
from schemas import (
    AnalysisResponse, TaskType,
    ConfirmMappingRequest, ConfirmMappingResponse,
)
from analyzer import parse_file_content, analyze_columns_with_llm
from validator import validate_mapping

router = APIRouter(prefix="/api", tags=["Column Analysis"])

ALLOWED_EXTENSIONS = {"csv", "json"}

@router.post(
    "/analyze-columns",
    response_model=AnalysisResponse,
    summary="컬럼 자동 매핑",
    description=(
        "**task_type** (binary / multiclass / multilabel) 을 지정하면, "
        "LLM이 해당 유형의 ISO 4213 컬럼 정의에 맞게 각 컬럼을 매핑합니다.\n\n"
        "지원 파일 형식: **CSV**, **JSON**"
    )
)
async def analyze_columns(
    request: Request,
    task_type: TaskType = Form(..., description="분류 모델 유형: binary / multiclass / multilabel"),
    file: UploadFile = File(..., description="분석할 파일 (.csv 또는 .json)"),
) -> AnalysisResponse:
    # 파일 확장자 검증
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: .{ext or '(없음)'}. CSV 또는 JSON 파일을 업로드해주세요."
        )

    # 파일 내용 읽기
    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="빈 파일은 처리할 수 없습니다.")

    # 파일 파싱 (CSV / JSON 공통 처리)
    try:
        columns, df = parse_file_content(file_content, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"파일 파싱 실패: {str(e)}")

    if not columns:
        raise HTTPException(status_code=422, detail="파일에 컬럼이 없습니다.")

    # app.state에서 openai_client 가져오기
    client = request.app.state.openai_client
    if not client:
        print("⚠️ OPENAI_API_KEY가 설정되어 있지 않아 룰 기반(Rule-based) 폴백 컬럼 매핑을 수행합니다.")
        from analyzer import analyze_columns_fallback
        return analyze_columns_fallback(
            task_type=task_type,
            columns=columns,
            df=df,
        )

    # LLM 호출 → 컬럼 매핑 + 메타데이터 추출
    try:
        result = await analyze_columns_with_llm(
            client=client,
            task_type=task_type,
            columns=columns,
            df=df,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 분석 실패: {str(e)}")

    return result


@router.post(
    "/confirm-mapping",
    response_model=ConfirmMappingResponse,
    summary="매핑 확정 및 유효성 검사",
    description=(
        "사용자가 검토·수정한 컬럼 매핑을 서버에 제출합니다.\n\n"
        "- 필수 역할 누락 시 **오류(errors)** 발생\n"
        "- 선택 역할 미설정 시 **경고(warnings)** 반환\n"
        "- 선택된 TC와 대조하여 **계산 가능성(available_tcs)** 검사"
    )
)
async def confirm_mapping(request: ConfirmMappingRequest) -> ConfirmMappingResponse:
    return validate_mapping(request)
