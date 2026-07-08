import pandas as pd
import numpy as np
import ast
from typing import Dict, Any, List, Tuple

def preprocess_data(df: pd.DataFrame, mappings: List[Dict[str, str]], task_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    평가(Metric) 계산 전 데이터를 정리하고 검증하는 전처리 함수.
    """
    if df.empty:
        raise ValueError("데이터셋이 비어 있습니다.")
        
    logs = {
        "dropped_rows": 0,
        "warnings": [],
        "errors": []
    }
    
    # List[Dict] 형태의 mappings를 딕셔너리로 변환
    mapping_dict = {m['role']: m['column'] for m in mappings}

    # ── 0. 정답=예측 동일 컬럼 방어선 (가짜 100% 차단) ─────────────────────────────
    # 라우터/검증을 우회한 호출도 절대 accuracy 1.0 을 내지 않도록 하는 최종 백스톱.
    for true_role, pred_role in [
        ('y_true', 'y_pred'), ('true_class', 'predicted_class'), ('true_labels', 'pred_labels'),
    ]:
        t_col, p_col = mapping_dict.get(true_role), mapping_dict.get(pred_role)
        if t_col and p_col and t_col == p_col:
            raise ValueError(
                f"정답('{true_role}')과 예측('{pred_role}')에 동일한 컬럼 '{t_col}'이 매핑되어 "
                "모든 지표가 100%로 산출됩니다(평가 무의미). 서로 다른 컬럼을 지정해주세요."
            )

    # 멀티클래스의 경우 prob_per_class 가 여러 컬럼에 매핑될 수 있으므로 배열 추출
    prob_cols = [m['column'] for m in mappings if m['role'] == 'prob_per_class']
    
    # 전체 필수/선택 컬럼명 목록 추출
    required_cols = list(set([m['column'] for m in mappings if m['role'] != 'ignore']))
    
    # ── 1. 필수 컬럼 존재 확인 및 Pruning ──────────────────────────────────────────
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"데이터셋에 매핑된 필수 컬럼이 없습니다: {missing_cols}")
        
    df = df[required_cols].copy()
    
    # [버그 수정] 멀티레이블 컬럼의 결측치(NaN)는 "해당하는 레이블 없음"의 유효한 데이터이므로,
    # dropna로 인해 행 전체가 유실되지 않도록 빈 문자열('')로 미리 대체합니다.
    for role in ['true_labels', 'pred_labels']:
        col = mapping_dict.get(role)
        if col and col in df.columns:
            df[col] = df[col].fillna('')
            
    # ── 2. 결측치(NaN) 처리 ────────────────────────────────────────────────────────
    # latency 는 선택적 부가 측정이므로, 그 컬럼의 결측이 해당 행의 분류 평가를 무효화하지
    # 않도록 dropna 대상에서 제외한다(평가 샘플 수 보존). latency 미매핑이면 기존과 동일.
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

    # ── 3. 데이터 타입 정규화 (Type Casting) ───────────────────────────────────────
    # y_true (or true_class) 기준 y_pred 맞추기
    y_true_col = mapping_dict.get('y_true') or mapping_dict.get('true_class')
    y_pred_col = mapping_dict.get('y_pred') or mapping_dict.get('predicted_class')
    
    if y_true_col and y_pred_col:
        true_type = df[y_true_col].dtype
        try:
            df[y_pred_col] = df[y_pred_col].astype(true_type)
        except Exception:
            raise ValueError(f"예측 라벨 '{y_pred_col}'을 정답 라벨 '{y_true_col}'의 타입({true_type})으로 강제 변환할 수 없습니다.")
            
    # 멀티레이블 리스트 파싱 (ast.literal_eval + '|' 또는 ',' 구분자)
    def parse_multilabel(item):
        if isinstance(item, list): return item
        if isinstance(item, str):
            try:
                parsed = ast.literal_eval(item)
                if isinstance(parsed, list): return parsed
            except:
                pass
            if '|' in item:
                return [x.strip() for x in item.split('|') if x.strip()]
            return [x.strip() for x in item.split(',') if x.strip()]
        return [item]

    for role in ['true_labels', 'pred_labels']:
        col = mapping_dict.get(role)
        if col and col in df.columns:
            df[col] = df[col].apply(parse_multilabel)

    # ── 4. 확률값(Score) 타입 강제 변환 및 범위 유효성 검사 ──────────────────────────
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

    # ── 4-1. 지연시간(Latency) 컬럼 숫자 변환 (선택, best-effort) ────────────────────
    # latency 는 부가 측정이므로 비숫자/결측이 해당 행의 분류 평가를 막아서는 안 된다.
    # 숫자로 변환 못 하는 값은 NaN 으로 강제(coerce)하여 통계에서만 제외한다(평가는 계속).
    if latency_col and latency_col in df.columns:
        coerced = pd.to_numeric(df[latency_col], errors="coerce")
        bad = int((coerced.isna() & df[latency_col].notna()).sum())
        if bad > 0:
            logs["warnings"].append(f"지연시간(latency) 컬럼에 숫자가 아닌 값 {bad}개가 있어 해당 행의 지연시간은 측정에서 제외됩니다.")
        df[latency_col] = coerced
        if (df[latency_col] < 0).any():
            neg = int((df[latency_col] < 0).sum())
            logs["warnings"].append(f"지연시간(latency) 컬럼에 음수 값 {neg}개가 있습니다(측정 오류 가능).")

    # ── 5. 확률합 검증 (Multiclass 한정) ─────────────────────────────────────────
    if task_type == 'multiclass' and len(prob_cols) > 1:
        row_sums = df[prob_cols].sum(axis=1)
        invalid_sums = row_sums[(row_sums < 0.99) | (row_sums > 1.01)]
        if not invalid_sums.empty:
            logs["warnings"].append(f"Multiclass 확률합 경고: {len(invalid_sums)}개 행에서 확률의 합이 1.0(±0.01) 범위를 벗어났습니다. 결과의 신뢰도가 낮을 수 있습니다.")

    # ── 6. 클래스 분포(Class Distribution) 추출 ─────────────────────────────────
    class_dist = {}
    y_true_col_for_dist = mapping_dict.get('y_true') or mapping_dict.get('true_class') or mapping_dict.get('true_labels')
    if y_true_col_for_dist and y_true_col_for_dist in df.columns:
        if task_type == 'multilabel':
            # df[y_true_col_for_dist]는 이미 parse_multilabel을 거쳐 리스트 형태임
            for labels in df[y_true_col_for_dist].dropna():
                if isinstance(labels, list):
                    for label in labels:
                        class_dist[label] = class_dist.get(label, 0) + 1
        else:
            class_dist = df[y_true_col_for_dist].value_counts().to_dict()
    logs["class_distribution"] = class_dist

    return df, logs

