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

from app.analysis.schemas import ValidationCheckItem
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


def check_row_count(total_rows: int) -> list[ValidationCheckItem]:
    """3-0b. 데이터 행 수 검사. 헤더만 있는 파일(0행)은 평가 자체가 성립하지 않는다.

    종전에는 0행이 error_count=0 으로 검증을 통과해, 사용자가 6단계를 다 지난 뒤
    /api/evaluate 가 400 으로 실패했다. 그 시점에는 어디로 돌아가야 하는지 안내가
    없다(ISSUES.md D-12). 검증 단계에서 error 로 막아 프론트 게이트가 잡게 한다.

    지표별 최소 행 수 하한은 지표마다 다르므로 여기서 정하지 않는다 — 0행만 막는다.
    """
    if total_rows == 0:
        return [ValidationCheckItem(
            name="Empty dataset (no data rows)",
            result="0 rows",
            handling="Stop evaluation",
            status="error",
            group="common",
        )]
    return [ValidationCheckItem(
        name="Empty dataset (no data rows)",
        result=f"{total_rows} rows",
        handling="Stop evaluation",
        status="pass",
        group="common",
    )]


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


def check_missing_values(dropped_rows: int) -> list[ValidationCheckItem]:
    """3-2. 결측치(NaN) 검사.

    자체 계산하지 않고 **평가 프레임이 확정한 제외 행 수를 그대로 보고한다**.
    종전에는 여기서 따로 세느라 같은 응답 안에서 "Missing value: None" 과
    "Excluded samples: 2 rows" 가 동시에 나오는 자기모순이 있었다(ISSUES.md D-01).
    """
    nan_count = dropped_rows
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
                pred_values = df_clean[y_pred_col].astype(str).values

                # 이 검사는 컬럼명이 'prob_<클래스>' 규칙을 따를 때만 클래스를 알아낼 수 있다.
                # 규칙을 따르지 않는 데이터셋(p_cat, score_1 …)에서는 추출한 이름이 실제 클래스와
                # 하나도 맞지 않아 전 행이 불일치로 집계되는 허위 경고가 나온다.
                # 시스템이 컬럼↔클래스 대응을 매핑 단계에서 수집하지 않으므로,
                # 이름으로 클래스를 확정할 수 없으면 잘못된 정보를 주는 대신 검사를 건너뛴다.
                known_classes = set(pred_values)
                if y_true_col and y_true_col in df_clean.columns:
                    known_classes |= set(df_clean[y_true_col].astype(str))
                class_names_resolved = set(class_names) <= known_classes

                if class_names_resolved:
                    argmax_labels = [class_names[i] for i in argmax_indices]
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
        # 형식 오류의 기준은 평가 파서(preprocessor._parse_multilabel_value)와 일치시킨다.
        # 파서는 "A|B" / "A,B" / "['A','B']" / 단일 라벨 "A" 를 모두 정상 처리하므로,
        # 구분자가 없다는 이유만으로 단일 라벨 행을 오류로 세면 안 된다(거의 모든
        # 멀티레이블 데이터셋에 단일 라벨 행이 있어 허위 경고가 항상 인쇄됐다).
        # 실제 오류는 "내용은 있는데 라벨을 하나도 뽑아낼 수 없는" 값뿐이다(예: "|", ",").
        format_errors = 0
        for val in df_clean[true_labels_col].head(100):
            if not isinstance(val, str) or val.strip() == "":
                continue  # 빈 셀 = '해당 라벨 없음' (정상 입력)
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    continue
            except (ValueError, SyntaxError):
                pass
            separator = '|' if '|' in val else ','
            if not [x for x in val.split(separator) if x.strip()]:
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
