"""app/analysis/validation_service.py — 데이터 검증 파이프라인 오케스트레이션.

파싱된 DataFrame 과 요청(EvaluateRequest)을 받아 검증 단계를 순서대로 조율하고
ValidateDataResponse 를 조립한다. 개별 점검 로직은 validation_checks 에 위임하며,
HTTP(파일 읽기·상태코드) 관심사는 라우터가 담당한다.
(구 validation_router.validate_data 490줄에서 오케스트레이션만 추출, 동작 불변.)

상호작용
- 의존(import): app.core.schemas(EvaluateRequest, ValidateDataResponse, ValidationCheckItem,
  ExecutionSummaryItem), app.analysis.validation_checks(개별 점검 함수 + TASK_CHECKS)
- 사용처: app.analysis.validation_router.validate_data
"""
from app.core.schemas import (
    EvaluateRequest,
    ExecutionSummaryItem,
    ValidateDataResponse,
    ValidationCheckItem,
)
from app.analysis import validation_checks as checks


def _counts(details: list[ValidationCheckItem]) -> tuple[int, int]:
    error_count = sum(1 for v in details if v.status == "error")
    warning_count = sum(1 for v in details if v.status == "warning")
    return error_count, warning_count


def _build_missing_required_response(
    task_type: str, selected_tcs: list[str], details: list[ValidationCheckItem], total_rows: int
) -> ValidateDataResponse:
    """필수 컬럼 누락 시 조기 반환 응답(이하 검증 불가)."""
    error_count, warning_count = _counts(details)
    return ValidateDataResponse(
        task_type=task_type,
        selected_metric_ids=selected_tcs,
        execution_summary=[
            ExecutionSummaryItem(label="Total validated rows", value=f"{total_rows} rows", note="업로드된 전체 행 수"),
            ExecutionSummaryItem(label="Validation result", value=f"Errors {error_count} / Warnings {warning_count}", note="필수 컬럼 누락으로 평가 진행 불가"),
        ],
        validation_details=details,
        error_count=error_count,
        warning_count=warning_count,
    )


def _build_full_response(
    task_type: str, selected_tcs: list[str], details: list[ValidationCheckItem],
    total_rows: int, valid_rows: int, excluded_rows: int,
) -> ValidateDataResponse:
    """전체 검증 완료 응답."""
    error_count, warning_count = _counts(details)
    execution_summary = [
        ExecutionSummaryItem(
            label="Total validated rows",
            value=f"{total_rows} rows",
            note="업로드된 파일의 전체 행 수",
        ),
        ExecutionSummaryItem(
            label="Valid prediction rows",
            value=f"{valid_rows} rows",
            note="결측치 제거 후 평가에 사용될 행 수",
        ),
        ExecutionSummaryItem(
            label="Excluded samples",
            value=f"{excluded_rows} rows",
            note="결측치·오류로 인해 제외된 행 수",
        ),
        ExecutionSummaryItem(
            label="Selected metric count",
            value=f"{len(selected_tcs)} items",
            note=f"선택된 메트릭: {', '.join(selected_tcs)}" if selected_tcs else "메트릭 미선택",
        ),
        ExecutionSummaryItem(
            label="Validation result",
            value=f"Errors {error_count} / Warnings {warning_count}",
            note="아래 상세 테이블 참조",
        ),
    ]
    return ValidateDataResponse(
        task_type=task_type,
        selected_metric_ids=selected_tcs,
        execution_summary=execution_summary,
        validation_details=details,
        error_count=error_count,
        warning_count=warning_count,
    )


def validate_dataset(df, request: EvaluateRequest) -> ValidateDataResponse:
    """검증 파이프라인 조율: 공통 점검 → (필수 컬럼 누락 시 조기 반환) → task별 → latency → 요약."""
    task_type = request.task_type.value
    mappings = [{"column": m.column, "role": m.role.value} for m in request.column_mappings]
    selected_tcs = request.selected_tcs

    total_rows = len(df)
    mapping_dict = {m["role"]: m["column"] for m in mappings}
    prob_cols = [m["column"] for m in mappings if m["role"] == "prob_per_class"]
    required_cols = list(set([m["column"] for m in mappings if m["role"] != "ignore"]))

    details: list[ValidationCheckItem] = []

    # 3-0. 컬럼 상호배타
    details += checks.check_column_conflicts(request.column_mappings, request.task_type)

    # 3-1. 필수 컬럼 존재 (누락 시 조기 반환)
    missing_cols = [col for col in required_cols if col not in df.columns]
    details += checks.check_missing_required(missing_cols)
    if missing_cols:
        return _build_missing_required_response(task_type, selected_tcs, details, total_rows)

    # 이하 검증은 필수 컬럼이 모두 존재하는 상태
    df_work = df[required_cols].copy()
    latency_col = mapping_dict.get("latency")

    # 3-2. 결측치 → df_clean 확정
    details += checks.check_missing_values(df_work, latency_col)
    df_clean = df_work.dropna()
    valid_rows = len(df_clean)
    excluded_rows = total_rows - valid_rows

    # 3-3~3-5. 공통 검사
    details += checks.check_duplicate_id(df_clean, mapping_dict)
    details += checks.check_class_mismatch(df_clean, mapping_dict)
    details += checks.check_excluded_samples(excluded_rows)

    # 3-6. task_type 별 추가 검사 (레지스트리 디스패치)
    task_check = checks.TASK_CHECKS.get(task_type)
    if task_check:
        details += task_check(df_clean, mapping_dict, prob_cols)

    # 3-7. latency
    details += checks.check_latency(df_clean, mapping_dict)

    return _build_full_response(task_type, selected_tcs, details, total_rows, valid_rows, excluded_rows)
