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
from app.analysis.schemas import ExecutionSummaryItem, ValidateDataResponse, ValidationCheckItem
from app.evaluation.schemas import EvaluateRequest
from app.analysis import validation_checks as checks
from app.evaluation.frame import build_evaluation_frame


def _counts(details: list[ValidationCheckItem]) -> tuple[int, int]:
    error_count = sum(1 for v in details if v.status == "error")
    warning_count = sum(1 for v in details if v.status == "warning")
    return error_count, warning_count


def _build_missing_required_response(
    task_type: str, selected_metric_ids: list[str], details: list[ValidationCheckItem], total_rows: int,
    reason: str = "필수 컬럼 누락으로 평가 진행 불가",
) -> ValidateDataResponse:
    """이하 검증이 불가능할 때의 조기 반환 응답(필수 컬럼 누락 · 0행 등)."""
    error_count, warning_count = _counts(details)
    return ValidateDataResponse(
        task_type=task_type,
        selected_metric_ids=selected_metric_ids,
        execution_summary=[
            ExecutionSummaryItem(label="Total validated rows", value=f"{total_rows} rows", note="업로드된 전체 행 수"),
            ExecutionSummaryItem(label="Validation result", value=f"Errors {error_count} / Warnings {warning_count}", note=reason),
        ],
        validation_details=details,
        error_count=error_count,
        warning_count=warning_count,
    )


def _build_full_response(
    task_type: str, selected_metric_ids: list[str], details: list[ValidationCheckItem],
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
            value=f"{len(selected_metric_ids)} items",
            note=f"선택된 메트릭: {', '.join(selected_metric_ids)}" if selected_metric_ids else "메트릭 미선택",
        ),
        ExecutionSummaryItem(
            label="Validation result",
            value=f"Errors {error_count} / Warnings {warning_count}",
            note="아래 상세 테이블 참조",
        ),
    ]
    return ValidateDataResponse(
        task_type=task_type,
        selected_metric_ids=selected_metric_ids,
        execution_summary=execution_summary,
        validation_details=details,
        error_count=error_count,
        warning_count=warning_count,
    )


def validate_dataset(df, request: EvaluateRequest) -> ValidateDataResponse:
    """검증 파이프라인 조율: 공통 점검 → (필수 컬럼 누락 시 조기 반환) → task별 → latency → 요약."""
    task_type = request.task_type.value
    mappings = [{"column": m.column, "role": m.role.value} for m in request.column_mappings]
    selected_metric_ids = request.selected_metric_ids

    total_rows = len(df)
    mapping_dict = {m["role"]: m["column"] for m in mappings}
    prob_cols = [m["column"] for m in mappings if m["role"] == "prob_per_class"]
    required_cols = list(set([m["column"] for m in mappings if m["role"] != "ignore"]))

    details: list[ValidationCheckItem] = []

    # 3-0. 컬럼 상호배타
    details += checks.check_column_conflicts(request.column_mappings, request.task_type)

    # 3-0b. 데이터 행 수 (0행이면 이하 검증이 전부 무의미하므로 조기 반환)
    details += checks.check_row_count(total_rows)
    if total_rows == 0:
        return _build_missing_required_response(
            task_type, selected_metric_ids, details, total_rows,
            reason="데이터 행이 없어 평가 진행 불가",
        )

    # 3-1. 필수 컬럼 존재 (누락 시 조기 반환)
    missing_cols = [col for col in required_cols if col not in df.columns]
    details += checks.check_missing_required(missing_cols)
    if missing_cols:
        return _build_missing_required_response(task_type, selected_metric_ids, details, total_rows)

    # 3-2. 결측치 → df_clean 확정.
    # **평가와 같은 함수로 프레임을 만든다**(ISSUES.md D-01). 종전에는 여기서 따로
    # `df_work.dropna()` 를 했는데, 그 기준은 latency 를 포함하고 멀티레이블 빈 셀을
    # 결측으로 셌다. 평가는 latency 를 제외하고 빈 셀을 살리므로 두 표본 수가 어긋났고,
    # 성적서 6절에 인쇄되는 수와 지표를 만든 수가 달랐다.
    # 후속 검사(3-3~3-7)가 이 프레임을 그대로 받으므로 D-02(과다 축소된 데이터 위의
    # 허위 경고)도 함께 해소된다.
    df_clean, excluded_rows, _notes = build_evaluation_frame(
        df, mappings, task_type, selected_metric_ids
    )
    valid_rows = len(df_clean)
    details += checks.check_missing_values(excluded_rows)

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

    return _build_full_response(task_type, selected_metric_ids, details, total_rows, valid_rows, excluded_rows)
