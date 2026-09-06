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
