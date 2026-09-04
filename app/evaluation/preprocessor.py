"""app/evaluation/preprocessor.py — 지표 계산 전 데이터 정리/검증 전처리

preprocess_data 는 단계별 헬퍼(_guard/_prune/_drop/_coerce/_validate/_check/_extract)를 순서대로
호출하는 얇은 파이프라인이다. 각 헬퍼는 한 가지 정리·검증 책임만 가진다.
(구 preprocess_data 150줄 단일 함수를 단계 함수로 분해, 동작·경고 메시지·순서 불변.)

상호작용
- 의존(import): pandas
- 사용처: app.evaluation.engine.evaluate (지표 계산 직전 호출)
"""
import ast
from typing import Any, Dict, List, Tuple

import pandas as pd


def _guard_identical_true_pred(mapping_dict: dict) -> None:
    """0. 정답=예측 동일 컬럼 방어선 (가짜 100% 차단). 라우터/검증 우회 호출의 최종 백스톱."""
    for true_role, pred_role in [
        ('y_true', 'y_pred'), ('true_class', 'predicted_class'), ('true_labels', 'pred_labels'),
    ]:
        t_col, p_col = mapping_dict.get(true_role), mapping_dict.get(pred_role)
        if t_col and p_col and t_col == p_col:
            raise ValueError(
                f"정답('{true_role}')과 예측('{pred_role}')에 동일한 컬럼 '{t_col}'이 매핑되어 "
                "모든 지표가 100%로 산출됩니다(평가 무의미). 서로 다른 컬럼을 지정해주세요."
            )


def _prune_to_required(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """1. 필수 컬럼 존재 확인 및 필요한 컬럼만 남기기."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"데이터셋에 매핑된 필수 컬럼이 없습니다: {missing_cols}")
    return df[required_cols].copy()


def _fill_multilabel_missing(df: pd.DataFrame, mapping_dict: dict) -> pd.DataFrame:
    """멀티레이블 컬럼 결측치는 '해당 레이블 없음'의 유효 데이터 → dropna 유실 방지 위해 '' 대체."""
    for role in ['true_labels', 'pred_labels']:
        col = mapping_dict.get(role)
        if col and col in df.columns:
            df[col] = df[col].fillna('')
    return df


def _drop_missing_rows(df: pd.DataFrame, mapping_dict: dict, logs: Dict[str, Any]) -> pd.DataFrame:
    """2. 결측치(NaN) 처리. latency 는 부가 측정이라 dropna 대상 제외(평가 샘플 수 보존)."""
    latency_col = mapping_dict.get('latency')
    dropna_subset = (
        [c for c in df.columns if c != latency_col] if latency_col in df.columns else None
    )
    initial_len = len(df)
    df.dropna(subset=dropna_subset, inplace=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        logs["dropped_rows"] = dropped
        logs["warnings"].append(f"{dropped}개 행이 결측치(NaN)로 인해 제외되었습니다 (전체 {initial_len}개 중 {len(df)}개 평가).")
    if len(df) == 0:
        raise ValueError("결측치를 제외하고 나니 평가할 데이터가 하나도 남지 않았습니다.")
    return df


def _coerce_label_types(df: pd.DataFrame, mapping_dict: dict) -> pd.DataFrame:
    """3. y_true 기준으로 y_pred dtype 강제 캐스팅."""
    y_true_col = mapping_dict.get('y_true') or mapping_dict.get('true_class')
    y_pred_col = mapping_dict.get('y_pred') or mapping_dict.get('predicted_class')
    if y_true_col and y_pred_col:
        true_type = df[y_true_col].dtype
        pred_series = df[y_pred_col]

        # 정수형으로 캐스팅할 때 소수값은 astype 이 예외 없이 잘라낸다(0.6 → 0).
        # 확률 컬럼을 y_pred 로 잘못 매핑한 경우가 대표적이며, 그대로 두면 모든 예측이
        # 0 으로 뭉개진 채 겉보기 정상인 성적서가 발급된다. 조용한 절단 대신 차단한다.
        if pd.api.types.is_integer_dtype(true_type) and pd.api.types.is_float_dtype(pred_series):
            fractional = pred_series.dropna() % 1 != 0
            if fractional.any():
                sample = pred_series[fractional].iloc[0]
                raise ValueError(
                    f"예측 라벨 '{y_pred_col}'에 소수값({sample})이 있어 정답 라벨 '{y_true_col}'의 "
                    f"정수 타입으로 변환하면 값이 잘립니다. 확률 컬럼을 예측(y_pred) 컬럼으로 "
                    f"매핑하지 않았는지 확인해주세요."
                )

        try:
            df[y_pred_col] = pred_series.astype(true_type)
        except Exception:
            raise ValueError(f"예측 라벨 '{y_pred_col}'을 정답 라벨 '{y_true_col}'의 타입({true_type})으로 강제 변환할 수 없습니다.")
    return df


def _parse_multilabel_value(item):
    """멀티레이블 셀 → 리스트 (ast.literal_eval + '|' 또는 ',' 구분자)."""
    if isinstance(item, list):
        return item
    if isinstance(item, str):
        try:
            parsed = ast.literal_eval(item)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        if '|' in item:
            return [x.strip() for x in item.split('|') if x.strip()]
        return [x.strip() for x in item.split(',') if x.strip()]
    return [item]


def _parse_multilabel_columns(df: pd.DataFrame, mapping_dict: dict) -> pd.DataFrame:
    """3. 멀티레이블 true/pred 컬럼을 리스트로 파싱."""
    for role in ['true_labels', 'pred_labels']:
        col = mapping_dict.get(role)
        if col and col in df.columns:
            df[col] = df[col].apply(_parse_multilabel_value)
    return df


def _validate_score_columns(df: pd.DataFrame, mapping_dict: dict, prob_cols: List[str]) -> pd.DataFrame:
    """4. 확률/점수 컬럼 float 강제 변환 + [0,1] 범위 유효성 검사."""
    score_roles = ['score_positive', 'score', 'score_per_label']
    score_cols = [mapping_dict[r] for r in score_roles if r in mapping_dict]
    score_cols.extend(prob_cols)

    for col in score_cols:
        if col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except Exception:
                raise ValueError(f"확률 컬럼 '{col}'에 숫자로 변환할 수 없는 문자가 포함되어 있습니다.")

            invalid = df[(df[col] < 0.0) | (df[col] > 1.0)]
            if not invalid.empty:
                # 첫 번째 에러가 발생한 위치를 구체적으로 안내
                first_idx = invalid.index[0]
                first_val = invalid.loc[first_idx, col]
                raise ValueError(f"'{col}' 컬럼의 {first_idx}번째 행: 값 {first_val} (허용범위 0.0~1.0 초과). Logit 등이 아닌 0~1 사이의 확률값으로 변환 후 다시 업로드해 주세요.")
    return df


def _coerce_latency(df: pd.DataFrame, mapping_dict: dict, logs: Dict[str, Any]) -> None:
    """4-1. 지연시간 컬럼 숫자 변환(best-effort). 비숫자→NaN(통계만 제외), 음수 경고."""
    latency_col = mapping_dict.get('latency')
    if latency_col and latency_col in df.columns:
        coerced = pd.to_numeric(df[latency_col], errors="coerce")
        bad = int((coerced.isna() & df[latency_col].notna()).sum())
        if bad > 0:
            logs["warnings"].append(f"지연시간(latency) 컬럼에 숫자가 아닌 값 {bad}개가 있어 해당 행의 지연시간은 측정에서 제외됩니다.")
        df[latency_col] = coerced
        if (df[latency_col] < 0).any():
            neg = int((df[latency_col] < 0).sum())
            logs["warnings"].append(f"지연시간(latency) 컬럼에 음수 값 {neg}개가 있습니다(측정 오류 가능).")


def _check_prob_sum(df: pd.DataFrame, task_type: str, prob_cols: List[str], logs: Dict[str, Any]) -> None:
    """5. 확률합 검증 (Multiclass 한정)."""
    if task_type == 'multiclass' and len(prob_cols) > 1:
        row_sums = df[prob_cols].sum(axis=1)
        invalid_sums = row_sums[(row_sums < 0.99) | (row_sums > 1.01)]
        if not invalid_sums.empty:
            logs["warnings"].append(f"Multiclass 확률합 경고: {len(invalid_sums)}개 행에서 확률의 합이 1.0(±0.01) 범위를 벗어났습니다. 결과의 신뢰도가 낮을 수 있습니다.")


def _extract_class_distribution(df: pd.DataFrame, mapping_dict: dict, task_type: str, logs: Dict[str, Any]) -> None:
    """6. 클래스 분포 추출."""
    class_dist = {}
    y_true_col_for_dist = mapping_dict.get('y_true') or mapping_dict.get('true_class') or mapping_dict.get('true_labels')
    if y_true_col_for_dist and y_true_col_for_dist in df.columns:
        if task_type == 'multilabel':
            # df[y_true_col_for_dist]는 이미 _parse_multilabel_value 를 거쳐 리스트 형태임
            for labels in df[y_true_col_for_dist].dropna():
                if isinstance(labels, list):
                    for label in labels:
                        class_dist[label] = class_dist.get(label, 0) + 1
        else:
            class_dist = df[y_true_col_for_dist].value_counts().to_dict()
    logs["class_distribution"] = class_dist


def preprocess_data(df: pd.DataFrame, mappings: List[Dict[str, str]], task_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """평가(Metric) 계산 전 데이터를 정리·검증하는 전처리 파이프라인(단계 헬퍼 순차 호출)."""
    if df.empty:
        raise ValueError("데이터셋이 비어 있습니다.")

    logs = {"dropped_rows": 0, "warnings": [], "errors": []}
    mapping_dict = {m['role']: m['column'] for m in mappings}

    _guard_identical_true_pred(mapping_dict)

    prob_cols = [m['column'] for m in mappings if m['role'] == 'prob_per_class']
    required_cols = list(set([m['column'] for m in mappings if m['role'] != 'ignore']))

    df = _prune_to_required(df, required_cols)
    df = _fill_multilabel_missing(df, mapping_dict)
    df = _drop_missing_rows(df, mapping_dict, logs)
    df = _coerce_label_types(df, mapping_dict)
    df = _parse_multilabel_columns(df, mapping_dict)
    df = _validate_score_columns(df, mapping_dict, prob_cols)
    _coerce_latency(df, mapping_dict, logs)
    _check_prob_sum(df, task_type, prob_cols, logs)
    _extract_class_distribution(df, mapping_dict, task_type, logs)

    return df, logs
