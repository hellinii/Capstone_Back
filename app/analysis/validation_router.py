"""
routers/validate.py — 데이터 검증 (전처리 dry-run) API

업로드된 파일과 매핑 정보를 받아 preprocess_data()를 실행하여
전처리 검증 결과를 반환합니다.  메트릭 계산은 수행하지 않습니다.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.core.schemas import (
    EvaluateRequest,
    ValidateDataResponse,
    ValidationCheckItem,
    ExecutionSummaryItem,
)
from app.analysis.parsing import parse_file_content
from app.analysis.validator import find_column_conflicts

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
    # ── 1. 파일 파싱 ──────────────────────────────────────────────────────────
    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="빈 파일은 처리할 수 없습니다.")

    filename = file.filename or ""
    try:
        _, df = parse_file_content(file_content, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"파일 파싱 실패: {str(e)}")

    # ── 2. 요청 데이터 파싱 ───────────────────────────────────────────────────
    try:
        request_data = EvaluateRequest.model_validate_json(data)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"설정 데이터 파싱 실패: {str(e)}",
        )

    task_type = request_data.task_type.value
    mappings = [{"column": m.column, "role": m.role.value} for m in request_data.column_mappings]
    selected_tcs = request_data.selected_tcs

    # ── 3. 검증 항목 생성 ─────────────────────────────────────────────────────
    validation_details: list[ValidationCheckItem] = []
    total_rows = len(df)

    # mapping_dict 구성
    mapping_dict = {m["role"]: m["column"] for m in mappings}
    # prob_per_class 여러 컬럼 추출
    prob_cols = [m["column"] for m in mappings if m["role"] == "prob_per_class"]
    # 필수 컬럼 추출 (ignore 제외)
    required_cols = list(set([m["column"] for m in mappings if m["role"] != "ignore"]))

    # ── 3-0. 컬럼 단위 상호배타 검사 (정답=예측 동일 컬럼 등 → 가짜 100% 차단) ──
    #   evaluate 만 막고 validate-data 가 "정상"이라 안내하면 UX 가 모순되므로 여기서도 검사.
    for conflict in find_column_conflicts(request_data.column_mappings, request_data.task_type):
        validation_details.append(ValidationCheckItem(
            name="Same column for true/pred" if conflict.code == "SAME_COLUMN_TRUE_PRED"
                 else "Column mapped to multiple roles",
            result=conflict.message,
            handling="Assign distinct columns",
            status="error",
            group="common",
        ))

    # ── 3-1. 필수 컬럼 존재 확인 ──────────────────────────────────────────────
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation_details.append(ValidationCheckItem(
            name="Missing required column",
            result=f"{', '.join(missing_cols)}",
            handling="Stop evaluation",
            status="error",
            group="common",
        ))
    else:
        validation_details.append(ValidationCheckItem(
            name="Missing required column",
            result="None",
            handling="Stop evaluation",
            status="pass",
            group="common",
        ))

    # 필수 컬럼이 없으면 아래 검증은 불가 → 바로 반환
    if missing_cols:
        error_count = sum(1 for v in validation_details if v.status == "error")
        warning_count = sum(1 for v in validation_details if v.status == "warning")
        return ValidateDataResponse(
            task_type=task_type,
            selected_metric_ids=selected_tcs,
            execution_summary=[
                ExecutionSummaryItem(label="Total validated rows", value=f"{total_rows} rows", note="업로드된 전체 행 수"),
                ExecutionSummaryItem(label="Validation result", value=f"Errors {error_count} / Warnings {warning_count}", note="필수 컬럼 누락으로 평가 진행 불가"),
            ],
            validation_details=validation_details,
            error_count=error_count,
            warning_count=warning_count,
        )

    # 이하 검증은 필수 컬럼이 모두 존재하는 상태
    df_work = df[required_cols].copy()

    # ── 3-2. 결측치(NaN) 검사 ─────────────────────────────────────────────────
    # latency 는 preprocessor 에서 dropna 대상이 아니므로(부가 측정), 여기서도 제외해
    # '제외 행 수'가 실제 평가와 어긋나지 않게 정합화한다(D5d).
    latency_col = mapping_dict.get("latency")
    nan_cols = [c for c in df_work.columns if c != latency_col]
    nan_count = int(df_work[nan_cols].isna().any(axis=1).sum()) if nan_cols else 0
    if nan_count > 0:
        validation_details.append(ValidationCheckItem(
            name="Missing value",
            result=f"{nan_count} rows",
            handling="Exclude affected rows from evaluation",
            status="warning",
            group="common",
        ))
    else:
        validation_details.append(ValidationCheckItem(
            name="Missing value",
            result="None",
            handling="Exclude affected rows from evaluation",
            status="pass",
            group="common",
        ))

    # NaN 제거 후 유효 df
    df_clean = df_work.dropna()
    valid_rows = len(df_clean)
    excluded_rows = total_rows - valid_rows

    # ── 3-3. 중복 ID 검사 ─────────────────────────────────────────────────────
    id_col = mapping_dict.get("sample_id")
    if id_col and id_col in df_clean.columns:
        dup_count = int(df_clean[id_col].duplicated().sum())
        if dup_count > 0:
            validation_details.append(ValidationCheckItem(
                name="Duplicate ID",
                result=f"{dup_count} rows",
                handling="Keep the first row and exclude later duplicates",
                status="warning",
                group="common",
            ))
        else:
            validation_details.append(ValidationCheckItem(
                name="Duplicate ID",
                result="None",
                handling="Keep the first row and exclude later duplicates",
                status="pass",
                group="common",
            ))
    else:
        validation_details.append(ValidationCheckItem(
            name="Duplicate ID",
            result="N/A (no sample_id mapped)",
            handling="Keep the first row and exclude later duplicates",
            status="pass",
            group="common",
        ))

    # ── 3-4. 클래스 불일치 검사 ───────────────────────────────────────────────
    y_true_col = mapping_dict.get("y_true") or mapping_dict.get("true_class")
    y_pred_col = mapping_dict.get("y_pred") or mapping_dict.get("predicted_class")

    if y_true_col and y_pred_col and y_true_col in df_clean.columns and y_pred_col in df_clean.columns:
        true_classes = set(df_clean[y_true_col].astype(str).unique())
        pred_classes = set(df_clean[y_pred_col].astype(str).unique())
        extra_in_pred = pred_classes - true_classes
        if extra_in_pred:
            validation_details.append(ValidationCheckItem(
                name="Class mismatch",
                result=f"Pred has unknown classes: {', '.join(sorted(extra_in_pred))}",
                handling="Exclude affected rows from evaluation",
                status="warning",
                group="common",
            ))
        else:
            validation_details.append(ValidationCheckItem(
                name="Class mismatch",
                result="None",
                handling="Exclude affected rows from evaluation",
                status="pass",
                group="common",
            ))
    else:
        validation_details.append(ValidationCheckItem(
            name="Class mismatch",
            result="N/A",
            handling="Exclude affected rows from evaluation",
            status="pass",
            group="common",
        ))

    # ── 3-5. 제외된 샘플 수 ───────────────────────────────────────────────────
    validation_details.append(ValidationCheckItem(
        name="Excluded samples",
        result=f"{excluded_rows} rows",
        handling="Exclude only rows with missing or invalid values",
        status="warning" if excluded_rows > 0 else "pass",
        group="common",
    ))

    # ── 3-6. Task-type 별 추가 검사 ───────────────────────────────────────────

    if task_type == "binary":
        # score_positive 범위 검사
        score_col = mapping_dict.get("score_positive")
        if score_col and score_col in df_clean.columns:
            try:
                scores = df_clean[score_col].astype(float)
                out_of_range = int(((scores < 0.0) | (scores > 1.0)).sum())
                if out_of_range > 0:
                    validation_details.append(ValidationCheckItem(
                        name="Score range error",
                        result=f"{out_of_range} rows out of [0, 1]",
                        handling="Exclude affected rows from evaluation",
                        status="error",
                        group="binary",
                    ))
                else:
                    validation_details.append(ValidationCheckItem(
                        name="Score range error",
                        result="0 rows",
                        handling="Exclude affected rows from evaluation",
                        status="pass",
                        group="binary",
                    ))
            except (ValueError, TypeError):
                validation_details.append(ValidationCheckItem(
                    name="Score range error",
                    result="Non-numeric values in score column",
                    handling="Stop evaluation",
                    status="error",
                    group="binary",
                ))
        else:
            validation_details.append(ValidationCheckItem(
                name="Score range error",
                result="N/A (no score_positive mapped)",
                handling="Exclude affected rows from evaluation",
                status="pass",
                group="binary",
            ))

        # Binary 클래스 수 검사
        if y_true_col and y_true_col in df_clean.columns:
            n_classes = df_clean[y_true_col].nunique()
            if n_classes != 2:
                validation_details.append(ValidationCheckItem(
                    name="Binary class system error",
                    result=f"Expected 2 classes, found {n_classes}",
                    handling="Exclude affected rows from evaluation",
                    status="error" if n_classes > 2 else "warning",
                    group="binary",
                ))
            else:
                validation_details.append(ValidationCheckItem(
                    name="Binary class system error",
                    result="None",
                    handling="Exclude affected rows from evaluation",
                    status="pass",
                    group="binary",
                ))

    elif task_type == "multiclass":
        # 확률값 개별 범위 검사 (prob_per_class 각 컬럼이 0~1 범위인지)
        if len(prob_cols) > 0:
            prob_in_df = [c for c in prob_cols if c in df_clean.columns]
            if prob_in_df:
                try:
                    prob_values = df_clean[prob_in_df].astype(float)
                    out_of_range = int(((prob_values < 0.0) | (prob_values > 1.0)).any(axis=1).sum())
                    if out_of_range > 0:
                        validation_details.append(ValidationCheckItem(
                            name="Probability range error",
                            result=f"{out_of_range} rows out of [0, 1]",
                            handling="Stop evaluation — values must be in [0.0, 1.0]",
                            status="error",
                            group="multiclass",
                        ))
                    else:
                        validation_details.append(ValidationCheckItem(
                            name="Probability range error",
                            result="0 rows",
                            handling="All probability values within [0.0, 1.0]",
                            status="pass",
                            group="multiclass",
                        ))
                except (ValueError, TypeError):
                    validation_details.append(ValidationCheckItem(
                        name="Probability range error",
                        result="Non-numeric values in probability columns",
                        handling="Stop evaluation",
                        status="error",
                        group="multiclass",
                    ))

        # 확률합 검사
        if len(prob_cols) > 1:
            prob_in_df = [c for c in prob_cols if c in df_clean.columns]
            if len(prob_in_df) > 1:
                try:
                    row_sums = df_clean[prob_in_df].astype(float).sum(axis=1)
                    invalid_sums = int(((row_sums < 0.99) | (row_sums > 1.01)).sum())
                    if invalid_sums > 0:
                        validation_details.append(ValidationCheckItem(
                            name="Probability sum error",
                            result=f"{invalid_sums} rows",
                            handling="Warn and continue",
                            status="warning",
                            group="multiclass",
                        ))
                    else:
                        validation_details.append(ValidationCheckItem(
                            name="Probability sum error",
                            result="0 rows",
                            handling="Warn and continue",
                            status="pass",
                            group="multiclass",
                        ))
                except (ValueError, TypeError):
                    validation_details.append(ValidationCheckItem(
                        name="Probability sum error",
                        result="Non-numeric values in probability columns",
                        handling="Stop evaluation",
                        status="error",
                        group="multiclass",
                    ))

        # Argmax-y_pred 불일치 검사
        if len(prob_cols) > 1 and y_pred_col and y_pred_col in df_clean.columns:
            prob_in_df = [c for c in prob_cols if c in df_clean.columns]
            if len(prob_in_df) > 1:
                try:
                    prob_df = df_clean[prob_in_df].astype(float)
                    argmax_indices = prob_df.values.argmax(axis=1)
                    # 클래스명 추출 (prob_className 형태 가정)
                    class_names = [c.replace("prob_", "") for c in prob_in_df]
                    argmax_labels = [class_names[i] for i in argmax_indices]
                    pred_values = df_clean[y_pred_col].astype(str).values
                    mismatch_count = int(sum(1 for a, p in zip(argmax_labels, pred_values) if a != p))
                    if mismatch_count > 0:
                        validation_details.append(ValidationCheckItem(
                            name="Argmax and y_pred mismatch",
                            result=f"{mismatch_count} rows",
                            handling="Warn and continue",
                            status="warning",
                            group="multiclass",
                        ))
                    else:
                        validation_details.append(ValidationCheckItem(
                            name="Argmax and y_pred mismatch",
                            result="0 rows",
                            handling="Warn and continue",
                            status="pass",
                            group="multiclass",
                        ))
                except Exception:
                    validation_details.append(ValidationCheckItem(
                        name="Argmax and y_pred mismatch",
                        result="Could not verify",
                        handling="Warn and continue",
                        status="warning",
                        group="multiclass",
                    ))

        # Unknown class detected (multiclass)
        if y_true_col and y_pred_col and y_true_col in df_clean.columns and y_pred_col in df_clean.columns:
            true_cls = set(df_clean[y_true_col].astype(str).unique())
            pred_cls = set(df_clean[y_pred_col].astype(str).unique())
            unknown = pred_cls - true_cls
            if unknown:
                validation_details.append(ValidationCheckItem(
                    name="Unknown class detected",
                    result=f"{', '.join(sorted(unknown))}",
                    handling="Exclude affected rows from evaluation",
                    status="warning",
                    group="multiclass",
                ))
            else:
                validation_details.append(ValidationCheckItem(
                    name="Unknown class detected",
                    result="None",
                    handling="Exclude affected rows from evaluation",
                    status="pass",
                    group="multiclass",
                ))

    elif task_type == "multilabel":
        # 멀티레이블 형식 검사
        true_labels_col = mapping_dict.get("true_labels")
        pred_labels_col = mapping_dict.get("pred_labels")
        
        if true_labels_col and true_labels_col in df_clean.columns:
            import ast
            format_errors = 0
            for val in df_clean[true_labels_col].head(100):
                if isinstance(val, str):
                    try:
                        parsed = ast.literal_eval(val)
                        if not isinstance(parsed, list):
                            format_errors += 1
                    except (ValueError, SyntaxError):
                        if '|' not in val and ',' not in val:
                            format_errors += 1
            if format_errors > 0:
                validation_details.append(ValidationCheckItem(
                    name="Label format mismatch",
                    result=f"{format_errors} rows (sampled first 100)",
                    handling="Exclude affected rows from evaluation",
                    status="warning",
                    group="multilabel",
                ))
            else:
                validation_details.append(ValidationCheckItem(
                    name="Label format mismatch",
                    result="None",
                    handling="Exclude affected rows from evaluation",
                    status="pass",
                    group="multilabel",
                ))

    # ── 3-7. 지연시간(Latency) 컬럼 검사 (선택, 매핑된 경우만) ──────────────────────
    # 평가 파이프라인과 일치하도록 결측 제거 후(df_clean) 기준으로 검사한다.
    # latency 는 best-effort 측정이므로 비숫자/음수는 평가를 막지 않고 경고로만 표시한다
    # (비숫자는 평가 시 NaN 으로 처리되어 통계에서만 제외됨).
    latency_col = mapping_dict.get("latency")
    if latency_col and latency_col in df_clean.columns:
        latency_numeric = pd.to_numeric(df_clean[latency_col], errors="coerce")
        # 원래 결측이 아니었는데 숫자 변환에 실패한 값 = 비숫자 문자열
        non_numeric = int((latency_numeric.isna() & df_clean[latency_col].notna()).sum())
        if non_numeric > 0:
            validation_details.append(ValidationCheckItem(
                name="Latency non-numeric values",
                result=f"{non_numeric} rows",
                handling="Treated as unmeasured (excluded from latency stats)",
                status="warning",
                group="latency",
            ))

        valid_latency = latency_numeric.dropna()
        negative = int((valid_latency < 0).sum())
        if negative > 0:
            validation_details.append(ValidationCheckItem(
                name="Latency negative values",
                result=f"{negative} rows",
                handling="Review measurement (kept as-is)",
                status="warning",
                group="latency",
            ))

        if len(valid_latency) > 0:
            validation_details.append(ValidationCheckItem(
                name="Latency statistics (ms)",
                result=(
                    f"mean={valid_latency.mean():.2f}, "
                    f"p95={valid_latency.quantile(0.95):.2f}, "
                    f"max={valid_latency.max():.2f} (n={len(valid_latency)})"
                ),
                handling="Informational",
                status="pass",
                group="latency",
            ))

    # ── 4. 실행 요약 구성 ─────────────────────────────────────────────────────
    error_count = sum(1 for v in validation_details if v.status == "error")
    warning_count = sum(1 for v in validation_details if v.status == "warning")

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
        validation_details=validation_details,
        error_count=error_count,
        warning_count=warning_count,
    )
