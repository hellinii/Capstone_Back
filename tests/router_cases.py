"""tests/router_cases.py — 라우터 characterization 테스트용 공용 입력 정의.

각 task_type 별 샘플 데이터셋 경로 + 컬럼 매핑 + 선택 TC 를 한곳에 모아,
validate-data/evaluate 골든 테스트가 동일한 입력을 재사용하게 한다.
(pytest 수집 대상이 아닌 헬퍼 모듈 — test_ 함수 없음.)
"""
from pathlib import Path

from app.core.schemas import ColumnMapping, ColumnRole, DataMetadata, TaskType
from app.evaluation.schemas import EvaluateRequest

DATA = Path(__file__).parent / "data"

# task_type → (매핑[(컬럼, 역할)], 선택 TC 목록). 선택 TC 는 test_evaluator 의 지원 목록과 동일.
CASES = {
    "binary": {
        "csv": DATA / "binary" / "binary_test_data_200.csv",
        "csv_errors": DATA / "binary" / "binary_test_data_with_errors.csv",
        "json": DATA / "binary" / "binary_test_data_200.json",
        "mappings": [
            ("id", "sample_id"),
            ("y_true", "y_true"),
            ("y_pred", "y_pred"),
            ("score_positive", "score_positive"),
            ("noise_col1", "ignore"),
            ("noise_col2", "ignore"),
        ],
        "selected_tcs": ["TC1", "TC2", "TC3", "TC4", "TC5", "TC6", "TC7", "TC8",
                         "TC9", "TC10", "TC19", "TC20", "TC21", "TC22", "TC23"],
    },
    "multiclass": {
        "csv": DATA / "multiclass" / "multiclass_200.csv",
        "csv_errors": DATA / "multiclass" / "multiclass_200_errors.csv",
        "json": DATA / "multiclass" / "multiclass_200.json",
        "mappings": [
            ("id", "sample_id"),
            ("y_true", "y_true"),
            ("y_pred", "y_pred"),
            ("prob_cat", "prob_per_class"),
            ("prob_dog", "prob_per_class"),
            ("prob_bird", "prob_per_class"),
        ],
        "selected_tcs": ["TC1", "TC2", "TC3", "TC4", "TC5", "TC6",
                         "TC11", "TC12", "TC13", "TC14", "TC21", "TC22", "TC23"],
    },
    "multilabel": {
        "csv": DATA / "multilabel" / "multilabel_200.csv",
        "csv_errors": DATA / "multilabel" / "multilabel_200_errors.csv",
        "json": DATA / "multilabel" / "multilabel_200.json",
        "mappings": [
            ("id", "sample_id"),
            ("true_labels", "true_labels"),
            ("pred_labels", "pred_labels"),
            ("score_sports", "score_per_label"),
            ("score_news", "score_per_label"),
            ("score_finance", "score_per_label"),
            ("score_tech", "score_per_label"),
        ],
        "selected_tcs": ["TC1", "TC2", "TC3", "TC4", "TC5",
                         "TC15", "TC16", "TC17", "TC18", "TC21", "TC22", "TC23"],
    },
}


# task 별 metadata 힌트(분석 단계 산출물 대체). binary 는 score 기반 지표를 위해 양/음성 클래스 지정.
_METADATA = {
    "binary": DataMetadata(positive_class="1", negative_class="0"),
    "multiclass": DataMetadata(),
    "multilabel": DataMetadata(),
}


def request_json(task: str) -> str:
    """해당 task 의 EvaluateRequest JSON 문자열(엔드포인트 data 폼필드용)."""
    c = CASES[task]
    req = EvaluateRequest(
        task_type=TaskType(task),
        column_mappings=[
            ColumnMapping(column=col, role=ColumnRole(role)) for col, role in c["mappings"]
        ],
        selected_tcs=c["selected_tcs"],
        metadata=_METADATA[task],
    )
    return req.model_dump_json()
