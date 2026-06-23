from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from schemas import EvaluateRequest, EvaluateResponse
from analyzer import parse_file_content
from evaluator.engine import evaluate as run_evaluation
from evaluator.report import generate_report

router = APIRouter(prefix="/api", tags=["Evaluation"])

@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="평가 연산 실행",
    description=(
        "업로드된 데이터셋 파일과 매핑 설정을 받아, "
        "선택된 ISO/IEC 4213 평가지표(TC)를 계산하여 결과를 반환합니다."
    )
)
async def evaluate_dataset(
    file: UploadFile = File(..., description="평가할 데이터셋 파일 (.csv 또는 .json)"),
    data: str = Form(..., description="EvaluateRequest 데이터의 JSON 문자열")
) -> EvaluateResponse:
    # 1. 파일 내용 읽기
    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="빈 파일은 처리할 수 없습니다.")

    # 2. 파일 파싱
    filename = file.filename or ""
    try:
        _, df = parse_file_content(file_content, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"파일 파싱 실패: {str(e)}")

    # 3. JSON 데이터 파싱 및 검증 (EvaluateRequest)
    try:
        request_data = EvaluateRequest.model_validate_json(data)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"설정 데이터 파싱 실패. 올바른 EvaluateRequest 형식이어야 합니다. 상세 에러: {str(e)}"
        )

    # 4. mappings 형식 변환 (List[Dict] 형태로 전송)
    mappings = [{"column": m.column, "role": m.role.value} for m in request_data.column_mappings]
    
    # positive_class 및 beta 값 가져오기
    positive_class = request_data.metadata.positive_class
    beta = request_data.beta

    # 5. 평가 연산 실행
    try:
        eval_results = run_evaluation(
            df=df,
            mappings=mappings,
            task_type=request_data.task_type.value,
            selected_tcs=request_data.selected_tcs,
            positive_class=positive_class,
            beta=beta
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"평가 연산 실행 오류: {str(e)}")

    # 6. 전처리 에러 확인 및 응답 포매팅
    if "error" in eval_results:
        raise HTTPException(status_code=400, detail=eval_results["error"])

    metadata = eval_results.pop("_metadata", {})
    warnings = metadata.get("warnings", [])
    dropped_rows = metadata.get("dropped_rows", 0)

    # 성공/실패 지표 분리 및 리포트 포매팅
    formatted_results = generate_report(eval_results)

    return EvaluateResponse(
        results=formatted_results,
        warnings=warnings,
        dropped_rows=dropped_rows
    )
