"""API 경로 계약 고정 — 리팩토링 전 구간 동안 /api/* 불변을 기계적으로 보증.

프론트(Vercel)가 의존하는 엔드포인트 집합이 디렉토리 리팩토링 중
사라지거나 바뀌면 이 테스트가 실패한다.
경로를 의도적으로 추가/변경할 때만 EXPECTED_PATHS 스냅샷을 갱신할 것.
"""
from app.main import app

EXPECTED_PATHS = {
    "/api/analyze-columns",
    "/api/confirm-mapping",
    "/api/evaluate",
    "/api/generate-narrative",
    "/api/organization",
    "/api/reports/issue",
    "/api/reports/{report_no}",
    "/api/reports/{report_no}/reissue",
    "/api/validate-data",
    "/health",
}


def test_api_paths_unchanged():
    assert set(app.openapi()["paths"].keys()) == EXPECTED_PATHS
