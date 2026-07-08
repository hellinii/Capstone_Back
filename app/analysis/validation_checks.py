"""app/analysis/validation_checks.py — 데이터 검증 개별 점검 함수 모음.

/api/validate-data 의 각 점검 항목을 단일 책임 순수 함수로 분리한다. 각 함수는
ValidationCheckItem 리스트를 반환하며, 오케스트레이션(순서·조기반환·요약)은
validation_service 가 담당한다. (구 validation_router.validate_data 490줄에서 추출, 동작 불변.)

상호작용
- 의존(import): pandas, app.core.schemas(ValidationCheckItem), app.analysis.validator(find_column_conflicts)
- 사용처: app.analysis.validation_service.validate_dataset
"""
import ast

import pandas as pd

from app.core.schemas import ValidationCheckItem
from app.analysis.validator import find_column_conflicts


def check_column_conflicts(column_mappings, task_type) -> list[ValidationCheckItem]:
    """3-0. 컬럼 단위 상호배타 검사 (정답=예측 동일 컬럼 등 → 가짜 100% 차단)."""
    items: list[ValidationCheckItem] = []
    for conflict in find_column_conflicts(column_mappings, task_type):
        items.append(ValidationCheckItem(
            name="Same column for true/pred" if conflict.code == "SAME_COLUMN_TRUE_PRED"
                 else "Column mapped to multiple roles",
            result=conflict.message,
            handling="Assign distinct columns",
            status="error",
            group="common",
        ))
    return items


def check_missing_required(missing_cols: list[str]) -> list[ValidationCheckItem]:
    """3-1. 필수 컬럼 존재 확인."""
    if missing_cols:
        return [ValidationCheckItem(
            name="Missing required column",
            result=f"{', '.join(missing_cols)}",
            handling="Stop evaluation",
            status="error",
            group="common",
        )]
    return [ValidationCheckItem(
        name="Missing required column",
        result="None",
        handling="Stop evaluation",
        status="pass",
        group="common",
    )]


def check_missing_values(df_work: pd.DataFrame, latency_col: str | None) -> list[ValidationCheckItem]:
    """3-2. 결측치(NaN) 검사. latency 는 부가 측정이라 제외해 '제외 행 수'를 평가와 정합화(D5d)."""
    nan_cols = [c for c in df_work.columns if c != latency_col]
    nan_count = int(df_work[nan_cols].isna().any(axis=1).sum()) if nan_cols else 0
    if nan_count > 0:
        return [ValidationCheckItem(
            name="Missing value",
            result=f"{nan_count} rows",
            handling="Exclude affected rows from evaluation",
            status="warning",
            group="common",
        )]
    return [ValidationCheckItem(
        name="Missing value",
        result="None",
        handling="Exclude affected rows from evaluation",
        status="pass",
        group="common",
    )]


def check_duplicate_id(df_clean: pd.DataFrame, mapping_dict: dict) -> list[ValidationCheckItem]:
    """3-3. 중복 ID 검사."""
    id_col = mapping_dict.get("sample_id")
    if id_col and id_col in df_clean.columns:
        dup_count = int(df_clean[id_col].duplicated().sum())
        if dup_count > 0:
            return [ValidationCheckItem(
                name="Duplicate ID",
                result=f"{dup_count} rows",
                handling="Keep the first row and exclude later duplicates",
                status="warning",
                group="common",
            )]
        return [ValidationCheckItem(
            name="Duplicate ID",
            result="None",
            handling="Keep the first row and exclude later duplicates",
            status="pass",
            group="common",
        )]
    return [ValidationCheckItem(
        name="Duplicate ID",
        result="N/A (no sample_id mapped)",
        handling="Keep the first row and exclude later duplicates",
        status="pass",
        group="common",
    )]


def check_class_mismatch(df_clean: pd.DataFrame, mapping_dict: dict) -> list[ValidationCheckItem]:
    """3-4. 클래스 불일치 검사 (예측에만 있는 미지 클래스)."""
    y_true_col = mapping_dict.get("y_true") or mapping_dict.get("true_class")
    y_pred_col = mapping_dict.get("y_pred") or mapping_dict.get("predicted_class")

    if y_true_col and y_pred_col and y_true_col in df_clean.columns and y_pred_col in df_clean.columns:
        true_classes = set(df_clean[y_true_col].astype(str).unique())
        pred_classes = set(df_clean[y_pred_col].astype(str).unique())
        extra_in_pred = pred_classes - true_classes
        if extra_in_pred:
            return [ValidationCheckItem(
                name="Class mismatch",
                result=f"Pred has unknown classes: {', '.join(sorted(extra_in_pred))}",
                handling="Exclude affected rows from evaluation",
                status="warning",
                group="common",
            )]
        return [ValidationCheckItem(
            name="Class mismatch",
            result="None",
            handling="Exclude affected rows from evaluation",
            status="pass",
            group="common",
        )]
    return [ValidationCheckItem(
        name="Class mismatch",
        result="N/A",
        handling="Exclude affected rows from evaluation",
        status="pass",
        group="common",
    )]


def check_excluded_samples(excluded_rows: int) -> list[ValidationCheckItem]:
    """3-5. 제외된 샘플 수."""
    return [ValidationCheckItem(
        name="Excluded samples",
        result=f"{excluded_rows} rows",
        handling="Exclude only rows with missing or invalid values",
        status="warning" if excluded_rows > 0 else "pass",
        group="common",
    )]


def check_binary(df_clean: pd.DataFrame, mapping_dict: dict, prob_cols: list[str]) -> list[ValidationCheckItem]:
    """3-6(binary). score_positive 범위 + 이진 클래스 수 검사."""
    items: list[ValidationCheckItem] = []
    y_true_col = mapping_dict.get("y_true") or mapping_dict.get("true_class")

    # score_positive 범위 검사
    score_col = mapping_dict.get("score_positive")
    if score_col and score_col in df_clean.columns:
        try:
            scores = df_clean[score_col].astype(float)
            out_of_range = int(((scores < 0.0) | (scores > 1.0)).sum())
            if out_of_range > 0:
                items.append(ValidationCheckItem(
                    name="Score range error",
                    result=f"{out_of_range} rows out of [0, 1]",
                    handling="Exclude affected rows from evaluation",
                    status="error",
                    group="binary",
                ))
            else:
                items.append(ValidationCheckItem(
                    name="Score range error",
                    result="0 rows",
                    handling="Exclude affected rows from evaluation",
                    status="pass",
                    group="binary",
                ))
        except (ValueError, TypeError):
            items.append(ValidationCheckItem(
                name="Score range error",
                result="Non-numeric values in score column",
                handling="Stop evaluation",
                status="error",
                group="binary",
            ))
    else:
        items.append(ValidationCheckItem(
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
            items.append(ValidationCheckItem(
                name="Binary class system error",
                result=f"Expected 2 classes, found {n_classes}",
                handling="Exclude affected rows from evaluation",
                status="error" if n_classes > 2 else "warning",
                group="binary",
            ))
        else:
            items.append(ValidationCheckItem(
                name="Binary class system error",
                result="None",
                handling="Exclude affected rows from evaluation",
                status="pass",
                group="binary",
            ))
    return items


def check_multiclass(df_clean: pd.DataFrame, mapping_dict: dict, prob_cols: list[str]) -> list[ValidationCheckItem]:
    """3-6(multiclass). 확률 범위/합/argmax 불일치 + 미지 클래스 검사."""
    items: list[ValidationCheckItem] = []
    y_pred_col = mapping_dict.get("y_pred") or mapping_dict.get("predicted_class")
    y_true_col = mapping_dict.get("y_true") or mapping_dict.get("true_class")

    # 확률값 개별 범위 검사 (prob_per_class 각 컬럼이 0~1 범위인지)
    if len(prob_cols) > 0:
        prob_in_df = [c for c in prob_cols if c in df_clean.columns]
        if prob_in_df:
            try:
                prob_values = df_clean[prob_in_df].astype(float)
                out_of_range = int(((prob_values < 0.0) | (prob_values > 1.0)).any(axis=1).sum())
                if out_of_range > 0:
                    items.append(ValidationCheckItem(
                        name="Probability range error",
                        result=f"{out_of_range} rows out of [0, 1]",
                        handling="Stop evaluation — values must be in [0.0, 1.0]",
                        status="error",
                        group="multiclass",
                    ))
                else:
                    items.append(ValidationCheckItem(
                        name="Probability range error",
                        result="0 rows",
                        handling="All probability values within [0.0, 1.0]",
                        status="pass",
                        group="multiclass",
                    ))
            except (ValueError, TypeError):
                items.append(ValidationCheckItem(
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
                    items.append(ValidationCheckItem(
                        name="Probability sum error",
                        result=f"{invalid_sums} rows",
                        handling="Warn and continue",
                        status="warning",
                        group="multiclass",
                    ))
                else:
                    items.append(ValidationCheckItem(
                        name="Probability sum error",
                        result="0 rows",
                        handling="Warn and continue",
                        status="pass",
                        group="multiclass",
                    ))
            except (ValueError, TypeError):
                items.append(ValidationCheckItem(
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
                    items.append(ValidationCheckItem(
                        name="Argmax and y_pred mismatch",
                        result=f"{mismatch_count} rows",
                        handling="Warn and continue",
                        status="warning",
                        group="multiclass",
                    ))
                else:
                    items.append(ValidationCheckItem(
                        name="Argmax and y_pred mismatch",
                        result="0 rows",
                        handling="Warn and continue",
                        status="pass",
                        group="multiclass",
                    ))
            except Exception:
                items.append(ValidationCheckItem(
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
            items.append(ValidationCheckItem(
                name="Unknown class detected",
                result=f"{', '.join(sorted(unknown))}",
                handling="Exclude affected rows from evaluation",
                status="warning",
                group="multiclass",
            ))
        else:
            items.append(ValidationCheckItem(
                name="Unknown class detected",
                result="None",
                handling="Exclude affected rows from evaluation",
                status="pass",
                group="multiclass",
            ))
    return items


def check_multilabel(df_clean: pd.DataFrame, mapping_dict: dict, prob_cols: list[str]) -> list[ValidationCheckItem]:
    """3-6(multilabel). 라벨 형식 검사(앞 100행 샘플)."""
    items: list[ValidationCheckItem] = []
    true_labels_col = mapping_dict.get("true_labels")

    if true_labels_col and true_labels_col in df_clean.columns:
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
            items.append(ValidationCheckItem(
                name="Label format mismatch",
                result=f"{format_errors} rows (sampled first 100)",
                handling="Exclude affected rows from evaluation",
                status="warning",
                group="multilabel",
            ))
        else:
            items.append(ValidationCheckItem(
                name="Label format mismatch",
                result="None",
                handling="Exclude affected rows from evaluation",
                status="pass",
                group="multilabel",
            ))
    return items


def check_latency(df_clean: pd.DataFrame, mapping_dict: dict) -> list[ValidationCheckItem]:
    """3-7. 지연시간(Latency) 컬럼 검사 (선택, 매핑된 경우만). 평가와 일치하게 df_clean 기준."""
    items: list[ValidationCheckItem] = []
    latency_col = mapping_dict.get("latency")
    if latency_col and latency_col in df_clean.columns:
        latency_numeric = pd.to_numeric(df_clean[latency_col], errors="coerce")
        # 원래 결측이 아니었는데 숫자 변환에 실패한 값 = 비숫자 문자열
        non_numeric = int((latency_numeric.isna() & df_clean[latency_col].notna()).sum())
        if non_numeric > 0:
            items.append(ValidationCheckItem(
                name="Latency non-numeric values",
                result=f"{non_numeric} rows",
                handling="Treated as unmeasured (excluded from latency stats)",
                status="warning",
                group="latency",
            ))

        valid_latency = latency_numeric.dropna()
        negative = int((valid_latency < 0).sum())
        if negative > 0:
            items.append(ValidationCheckItem(
                name="Latency negative values",
                result=f"{negative} rows",
                handling="Review measurement (kept as-is)",
                status="warning",
                group="latency",
            ))

        if len(valid_latency) > 0:
            items.append(ValidationCheckItem(
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
    return items


# task_type → task별 추가 검사 함수 (elif 사다리 대체)
TASK_CHECKS = {
    "binary": check_binary,
    "multiclass": check_multiclass,
    "multilabel": check_multilabel,
}
