import math
import pandas as pd
import pytest
from pathlib import Path
from app.evaluation.engine import evaluate
from app.evaluation.preprocessor import preprocess_data

# 7. CSV 경로를 pathlib.Path를 이용해 동적 절대 경로로 지정 (tests/data 하위, 소문자)
BASE_DIR = Path(__file__).parent
BINARY_CSV = BASE_DIR / "data" / "binary" / "binary_test_data_200.csv"
MULTICLASS_CSV = BASE_DIR / "data" / "multiclass" / "multiclass_200.csv"
MULTILABEL_CSV = BASE_DIR / "data" / "multilabel" / "multilabel_200.csv"

# 3. 지원 지표 목록을 모듈 상수로 분리하여 유지보수성 향상
BINARY_SUPPORTED_METRICS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M19", "M20", "M21", "M22", "M23"]
MULTICLASS_SUPPORTED_METRICS = ["M1", "M2", "M3", "M4", "M5", "M6", "M11", "M12", "M13", "M14", "M21", "M22", "M23"]
MULTILABEL_SUPPORTED_METRICS = ["M1", "M2", "M3", "M4", "M5", "M15", "M16", "M17", "M18", "M21", "M22", "M23"]

@pytest.fixture
def binary_df():
    return pd.read_csv(BINARY_CSV)

@pytest.fixture
def multiclass_df():
    return pd.read_csv(MULTICLASS_CSV)

@pytest.fixture
def multilabel_df():
    return pd.read_csv(MULTILABEL_CSV)


def assert_valid_results(results, selected_metric_ids, task_type):
    """공통 결과 검증 함수 (1, 2, 4번 개선사항 반영)"""
    print(f"\n[{task_type.upper()} 평가 결과 검증]")
    
    for metric_id in selected_metric_ids:
        assert metric_id in results, f"{metric_id} 결과가 누락됨"
        val = results[metric_id]
        
        # 1. 결과값 타입(None) 검증
        assert val is not None, f"{metric_id} 결과가 None입니다."
        
        if isinstance(val, float):
            # 1. NaN 검증
            assert not math.isnan(val), f"{metric_id} 결과가 NaN입니다."
            
            # 2. 메트릭 범위 검증 (대부분의 정확도/오차 지표는 0~1)
            # 단, M6(KL Divergence), M19(Log Loss), M23(Imbalance Ratio)는 1.0을 초과할 수 있으므로 제외
            if metric_id not in ["M6", "M19", "M23"]:
                assert 0.0 <= val <= 1.0, f"{metric_id} 값({val})이 0.0~1.0 범위를 벗어남"
            print(f"  [OK] {metric_id}: {val:.4f}")
            
        elif isinstance(val, dict):
            # Dict 형태 반환 시 내부 에러 텍스트 유무 검증
            assert "error" not in val, f"{metric_id} 에러 발생: {val['error']}"
            print(f"  [OK] {metric_id}: {str(val)[:50]}... (정상 Dict)")
        else:
            pytest.fail(f"{metric_id} 결과 타입 오류: {type(val)}")
            
    # 4. _metadata 구조 및 내용 검증
    assert "_metadata" in results
    meta = results["_metadata"]
    assert "dropped_rows" in meta
    assert isinstance(meta["warnings"], list)
    print(f"  [INFO] Metadata: {meta['dropped_rows']} rows dropped, {len(meta['warnings'])} warnings")


# ── 1. 전처리(Preprocessor) 테스트 ─────────────────────────────────────────────

def test_preprocessor_pruning_and_nan(binary_df):
    print("\n[Preprocessor - Pruning & NaN 검증]")
    mappings = [
        {"role": "y_true", "column": "y_true"},
        {"role": "y_pred", "column": "y_pred"},
        {"role": "score_positive", "column": "score_positive"}
    ]
    df_clean, logs = preprocess_data(binary_df.copy(), mappings, "binary")
    assert "noise_col1" not in df_clean.columns
    assert "score_positive" in df_clean.columns
    print("  [OK] Pruning 정상 동작 (noise_col 삭제됨)")
    
    df_with_nan = binary_df.copy()
    df_with_nan.loc[0, "y_true"] = float('nan')
    df_nan_clean, logs_nan = preprocess_data(df_with_nan, mappings, "binary")
    assert logs_nan["dropped_rows"] >= 1
    assert len(df_nan_clean) < len(binary_df)
    print("  [OK] 결측치 제거 및 dropped_rows 로그 정상 동작")

def test_preprocessor_range_validation(binary_df):
    print("\n[Preprocessor - 확률값 초과 검증]")
    mappings = [
        {"role": "y_true", "column": "y_true"},
        {"role": "y_pred", "column": "y_pred"},
        {"role": "score_positive", "column": "score_positive"}
    ]
    df_invalid_score = binary_df.copy()
    df_invalid_score.loc[0, "score_positive"] = 1.5
    with pytest.raises(ValueError, match="허용범위 0.0~1.0 초과"):
        preprocess_data(df_invalid_score, mappings, "binary")
    print("  [OK] 허용범위 초과 데이터 차단 정상 동작")

def test_multiclass_prob_sum_warning(multiclass_df):
    print("\n[Preprocessor - 다중 클래스 확률합 검증]")
    mappings = [
        {"role": "y_true", "column": "y_true"},
        {"role": "y_pred", "column": "y_pred"},
        {"role": "prob_per_class", "column": "prob_cat"},
        {"role": "prob_per_class", "column": "prob_dog"},
        {"role": "prob_per_class", "column": "prob_bird"},
    ]
    df_warn = multiclass_df.copy()
    # 5. 합 1.2 명시적 보장
    df_warn.loc[0, "prob_cat"] = 1.0
    df_warn.loc[0, "prob_dog"] = 0.2
    df_warn.loc[0, "prob_bird"] = 0.0
    
    df_clean, logs = preprocess_data(df_warn, mappings, "multiclass")
    assert any("확률의 합이 1.0" in w for w in logs["warnings"])
    print("  [OK] 확률합 1.0 초과 시 Warning 로그 생성 확인")


# ── 2. 엔진(Engine) 동적 실행 및 메트릭 테스트 ──────────────────────────────────

def test_engine_binary_evaluation(binary_df):
    mappings = [
        {"role": "y_true", "column": "y_true"},
        {"role": "y_pred", "column": "y_pred"},
        {"role": "score_positive", "column": "score_positive"}
    ]
    results = evaluate(binary_df, mappings, "binary", BINARY_SUPPORTED_METRICS)
    assert_valid_results(results, BINARY_SUPPORTED_METRICS, "binary")
    
    # 지원하지 않는 M15는 에러를 반환해야 함
    results_invalid = evaluate(binary_df, mappings, "binary", ["M15"])
    assert "M15" in results_invalid
    assert "error" in results_invalid["M15"]

def test_engine_binary_evaluation_with_positive_class(binary_df):
    mappings = [
        {"role": "y_true", "column": "y_true"},
        {"role": "y_pred", "column": "y_pred"},
        {"role": "score_positive", "column": "score_positive"}
    ]
    # positive_class = "1"
    results = evaluate(binary_df, mappings, "binary", BINARY_SUPPORTED_METRICS, positive_class="1")
    assert_valid_results(results, BINARY_SUPPORTED_METRICS, "binary")

    # positive_class = "0" (reverse positive class)
    results_reverse = evaluate(binary_df, mappings, "binary", BINARY_SUPPORTED_METRICS, positive_class="0")
    assert_valid_results(results_reverse, BINARY_SUPPORTED_METRICS, "binary")

def test_engine_multiclass_evaluation(multiclass_df):
    mappings = [
        {"role": "y_true", "column": "y_true"},
        {"role": "y_pred", "column": "y_pred"},
        {"role": "prob_per_class", "column": "prob_cat"},
        {"role": "prob_per_class", "column": "prob_dog"},
        {"role": "prob_per_class", "column": "prob_bird"},
    ]
    results = evaluate(multiclass_df, mappings, "multiclass", MULTICLASS_SUPPORTED_METRICS)
    assert_valid_results(results, MULTICLASS_SUPPORTED_METRICS, "multiclass")

def test_engine_multilabel_evaluation(multilabel_df):
    mappings = [
        {"role": "true_labels", "column": "true_labels"},
        {"role": "pred_labels", "column": "pred_labels"},
        {"role": "score_per_label", "column": "score_sports"},
        {"role": "score_per_label", "column": "score_news"},
        {"role": "score_per_label", "column": "score_finance"},
        {"role": "score_per_label", "column": "score_tech"},
    ]
    results = evaluate(multilabel_df, mappings, "multilabel", MULTILABEL_SUPPORTED_METRICS)
    assert_valid_results(results, MULTILABEL_SUPPORTED_METRICS, "multilabel")


# ── 3. 엣지 케이스 (Edge Cases) 테스트 ────────────────────────────────────────

def test_edge_case_empty_dataframe():
    """6. 빈 DataFrame 입력 시 에러 처리"""
    print("\n[Edge Case - 빈 데이터프레임]")
    df_empty = pd.DataFrame()
    mappings = [{"role": "y_true", "column": "y_true"}]
    # evaluate 함수는 에러를 {"error": "..."} 형태로 반환함
    results = evaluate(df_empty, mappings, "binary", ["M1"])
    assert "error" in results
    assert "비어 있" in results["error"]
    print("  [OK] 빈 데이터프레임 거부 정상 동작")

def test_edge_case_missing_mapping(binary_df):
    """6. 매핑 누락"""
    print("\n[Edge Case - 필수 매핑 누락]")
    # y_true가 누락됨
    mappings = [{"role": "y_pred", "column": "y_pred"}]
    results = evaluate(binary_df, mappings, "binary", ["M1"])
    # 엔진 설계상, 매핑 누락 등은 개별 지표 계산 중 잡혀서 지표 내부에 에러로 반환됨
    assert "M1" in results
    assert "error" in results["M1"]
    assert "매핑" in results["M1"]["error"] or "필수" in results["M1"]["error"]
    print("  [OK] 필수 매핑 누락 시 개별 지표 에러 반환 정상 동작")
