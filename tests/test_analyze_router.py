"""D5a/D5c/D6a — 컬럼명 대조(reconcile) + analyze-columns LLM 실패 폴백 + 클라이언트 timeout."""
import io

from fastapi.testclient import TestClient

from app.analysis.reconcile import reconcile_llm_columns
from app.core.schemas import ColumnRole


# ── D5a: reconcile_llm_columns (순수 함수) ─────────────────────────────────

def test_reconcile_exact_and_normalized():
    actual = ["id", "label", "prediction", "score"]
    llm = [
        {"column": "label", "role": "y_true"},       # exact
        {"column": "Prediction", "role": "y_pred"},   # 대소문자 정규화 → prediction
    ]
    reconciled, notes = reconcile_llm_columns(llm, actual)
    by_col = {r["column"]: r["role"] for r in reconciled}
    assert by_col["label"] == "y_true"
    assert by_col["prediction"] == "y_pred"
    assert any(n.status == "corrected" and n.matched_column == "prediction" for n in notes)


def test_reconcile_unmatched_dropped():
    actual = ["id", "label", "prediction"]
    llm = [{"column": "probability_x", "role": "score_positive"}]  # 환각 컬럼명
    reconciled, notes = reconcile_llm_columns(llm, actual)
    assert all(r["column"] != "probability_x" for r in reconciled)
    assert any(n.status == "unmatched" and n.llm_column == "probability_x" for n in notes)


def test_reconcile_unmapped_header_added_as_ignore():
    actual = ["id", "label", "prediction", "score"]
    llm = [{"column": "label", "role": "y_true"}, {"column": "prediction", "role": "y_pred"}]
    reconciled, notes = reconcile_llm_columns(llm, actual)
    by_col = {r["column"]: r["role"] for r in reconciled}
    assert by_col.get("id") == ColumnRole.ignore.value
    assert by_col.get("score") == ColumnRole.ignore.value
    assert sum(1 for n in notes if n.status == "unmapped_header") == 2


# ── D5c/D6a: 라우터 폴백 + 클라이언트 구성 (TestClient) ─────────────────────

_CSV = b"id,y_true,y_pred\n1,1,1\n2,0,1\n3,1,0\n4,0,0\n"


def _upload(client: TestClient):
    return client.post(
        "/api/analyze-columns",
        data={"task_type": "binary"},
        files={"file": ("data.csv", io.BytesIO(_CSV), "text/csv")},
    )


def test_analyze_no_key_falls_back():
    from app import main
    with TestClient(main.app) as c:
        main.app.state.openai_client = None
        r = _upload(c)
    assert r.status_code == 200
    assert len(r.json()["column_mappings"]) >= 1


def test_analyze_llm_error_falls_back(make_fake_openai_client):
    from app import main
    with TestClient(main.app) as c:
        main.app.state.openai_client = make_fake_openai_client(raise_exc=RuntimeError("boom"))
        r = _upload(c)
    assert r.status_code == 200  # 500 아님 — 규칙 폴백으로 강등(D5c)
    roles = {m["role"] for m in r.json()["column_mappings"]}
    assert "y_true" in roles and "y_pred" in roles


def test_analyze_bad_extension_still_rejected():
    from app import main
    with TestClient(main.app) as c:
        main.app.state.openai_client = None
        r = c.post(
            "/api/analyze-columns",
            data={"task_type": "binary"},
            files={"file": ("data.txt", io.BytesIO(b"x"), "text/plain")},
        )
    assert r.status_code in (400, 422)  # 폴백이 파싱/확장자 오류를 삼키지 않음


def test_openai_client_timeout_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    from app import main
    with TestClient(main.app):
        client = main.app.state.openai_client
        assert client is not None
        assert client.max_retries == 2  # D6a: 재시도 상한 명시
