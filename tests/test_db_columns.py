"""test_db_columns.py — 판 4 라운드 S0: DB 표와 열 이름을 고정한다.

이 프로젝트에는 마이그레이션 도구가 없고 create_all 은 기존 표에 열을 더하지 못한다(app.core.database.init_db).
운영 DB(Neon)에 이미 있는 표에 열을 더하면 배포 뒤 첫 쿼리에서야 깨진다. 판 4 정보는 열이 아니라 content(JSON) 안에
둔다(계획 2.5-3) — 그래서 표 이름과 열 이름 집합을 글자 그대로 고정한다. 바꾸려면 이 목록을 손으로 고치고 이유를 남긴다.
"""
import app.main  # noqa: F401 — 앱이 등록하는 모든 모델을 불러온다(새 표가 생겨도 잡히게)
from app.core.database import Base

EXPECTED_COLUMNS = {
    "organization": {"id", "org_name", "department", "evaluator", "contact", "address", "updated_at"},
    "report": {
        "id", "report_no", "year", "seq", "run_id", "model_name", "model_version", "org_id", "current_version",
        "created_at",
    },
    "report_snapshot": {"id", "report_id", "version", "content_json", "content_hash", "org_snapshot_json", "created_at"},
    "issuance": {"id", "report_id", "version", "issuer", "issued_at", "note", "status"},
    "narrative_jobs": {
        "id", "status", "stage", "result_json", "internal_json", "created_at", "updated_at", "finished_at",
    },
}


def _actual() -> dict[str, set[str]]:
    import app.issuance.models  # noqa: F401
    import app.narrative.models  # noqa: F401

    return {name: {c.name for c in table.columns} for name, table in Base.metadata.tables.items()}


def test_table_names_are_frozen():
    assert set(_actual()) == set(EXPECTED_COLUMNS)


def test_column_names_are_frozen():
    actual = _actual()
    for table, columns in EXPECTED_COLUMNS.items():
        assert actual[table] == columns, (table, sorted(actual[table] ^ columns))
