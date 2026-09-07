"""API 계약 고정 — 프론트(Vercel)가 의존하는 엔드포인트 집합의 불변을 기계적으로 보증.

경로만 비교하면 **메서드 추가·삭제가 드러나지 않는다**(ISSUES.md H-04). 실제로 이번
라운드에서 무인가 `PUT /api/organization` 을 제거했는데, 경로 집합 비교로는 같은 경로에
GET 이 남아 있어 아무 테스트도 깨지지 않았다 — 계약 축소가 조용히 통과한 것이다.
그래서 (메서드, 경로) 쌍으로 비교한다.

계약을 의도적으로 바꿀 때만 EXPECTED_OPERATIONS 를 갱신할 것.
"""
from app.main import app

# (HTTP 메서드, 경로) — 프론트와의 계약 표면 전체.
EXPECTED_OPERATIONS = {
    ("post", "/api/analyze-columns"),
    ("post", "/api/confirm-mapping"),
    ("post", "/api/evaluate"),
    ("post", "/api/generate-narrative"),
    ("get", "/api/organization"),
    ("post", "/api/reports/issue"),
    ("get", "/api/reports/{report_no}"),
    ("get", "/api/reports/{report_no}/content"),
    ("post", "/api/reports/{report_no}/reissue"),
    ("post", "/api/validate-data"),
    ("get", "/health"),
}


def _actual_operations() -> set[tuple[str, str]]:
    spec = app.openapi()["paths"]
    return {
        (method, path)
        for path, ops in spec.items()
        for method in ops
        if method in {"get", "post", "put", "patch", "delete"}
    }


def test_api_operations_unchanged():
    assert _actual_operations() == EXPECTED_OPERATIONS


def test_organization_is_read_only():
    """[G-01] 기관 정보를 HTTP 로 수정하는 경로가 없어야 한다.

    무인증 상태에서 PUT 한 번이면 이미 발급된 모든 성적서의 기관 표기가 소급 변경됐다.
    기관 정보는 bootstrap 시드로만 바꾼다.
    """
    write_methods = {
        (m, p) for (m, p) in _actual_operations()
        if p == "/api/organization" and m != "get"
    }
    assert write_methods == set(), f"기관 정보에 쓰기 경로가 있다: {write_methods}"


def test_every_operation_declares_a_response_model():
    """응답 스키마가 선언되지 않은 오퍼레이션이 없어야 한다(계약의 최소 조건)."""
    spec = app.openapi()["paths"]
    missing = [
        f"{m.upper()} {p}"
        for p, ops in spec.items()
        for m, op in ops.items()
        if m in {"get", "post", "put", "patch", "delete"}
        and "200" in op.get("responses", {})
        and "content" not in op["responses"]["200"]
    ]
    assert missing == [], f"200 응답 스키마가 없는 오퍼레이션: {missing}"


# ── 응답 필드 계약 (ISSUES.md H-04) ────────────────────────────────────────
#
# (메서드, 경로) 비교는 **엔드포인트가 사라지는 것**만 잡는다. 응답에서 필드 하나가
# 빠지는 변경은 그대로 통과하고, 프론트는 `undefined` 를 받아 옵셔널 체이닝으로
# 조용히 0/빈 배열로 처리한다(뿌리 ③). 프론트가 실제로 **읽는** 필드를 여기 고정한다.
#
# 이번 라운드에만 응답 필드가 다섯 늘었다(environment · derived_prediction ·
# column_notes · warnings · available_metric_ids). 하나가 조용히 사라지면 성적서의
# 어떤 칸이 비는지 코드만 봐서는 알 수 없다.

_SCHEMAS = lambda: app.openapi()["components"]["schemas"]


def _fields(schema_name: str) -> set[str]:
    return set(_SCHEMAS()[schema_name].get("properties", {}))


def test_evaluate_response_fields():
    """성적서 6절 표본 수·3절 진단·4절 평가 환경이 전부 여기서 온다."""
    assert {
        "results", "warnings", "dropped_rows", "n_samples",
        "class_distribution", "environment",
    } <= _fields("EvaluateResponse")


def test_evaluation_environment_fields():
    """4절 '주요 라이브러리'·'평가 수행 일시'의 출처(F-09)."""
    assert {"libraries", "evaluated_at"} <= _fields("EvaluationEnvironment")


def test_evaluate_request_carries_the_decision_threshold():
    """확률 전용 경로의 입력(결정 1). 이름이 `threshold` 로 되돌아가면 프론트의
    성적서 합격 목표값(`metricDetails.targetValue` 파생)과 섞인다.

    이 모델은 multipart 의 `data` 폼필드로 실려 오므로 OpenAPI 스키마에 나타나지
    않는다 — 모델에서 직접 확인한다.
    """
    from app.evaluation.schemas import EvaluateRequest

    fields = set(EvaluateRequest.model_fields)
    assert {"task_type", "column_mappings", "selected_metric_ids", "metadata",
            "beta", "decision_threshold"} == fields
    assert "threshold" not in fields


def test_confirm_mapping_response_fields():
    """6단계 안내 배선(B-04·A-12)이 읽는 필드."""
    assert {
        "is_valid", "errors", "warnings",
        "available_metric_ids", "unavailable_metric_ids",
    } <= _fields("ConfirmMappingResponse")


def test_analysis_response_fields():
    """컬럼 대조 안내(B-03)가 여기 실린다."""
    assert {"task_type", "column_mappings", "metadata", "column_notes"} <= _fields("AnalysisResponse")


def test_validate_data_response_fields():
    """6절 검증표와 프론트 진행 게이트(E-04)가 읽는 필드."""
    assert {
        "task_type", "selected_metric_ids", "execution_summary",
        "validation_details", "error_count", "warning_count",
    } <= _fields("ValidateDataResponse")


def test_validation_check_item_fields():
    """`handling` 은 6단계 화면이 인쇄한다(D-03 이 문구를 고친 자리)."""
    assert {"name", "result", "handling", "status", "group"} <= _fields("ValidationCheckItem")
