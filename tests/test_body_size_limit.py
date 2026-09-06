"""[G-03] 순수 JSON 본문 크기 상한.

필드 단위 상한(metrics 64개 · 클래스 256개 …)만으로는 부족하다는 것이 실측으로
드러났다. 18.8 MB JSON 을 /api/generate-narrative 로 보내면 상한에 걸려 422 로
거절되지만, **거절 판정을 내리기까지 pydantic 이 그 18.8 MB 를 전부 파싱한다.**
그 3.2 초 동안 이벤트 루프가 멈춘다(오프로드보다 앞 단계라 G-04b 로도 못 막는다).

업로드(멀티파트)는 라우터 안에서 처리 전에 막히지만(G-04a), 순수 JSON 경로는
핸들러에 도달하기 전에 이미 검증이 끝나므로 미들웨어에서 재야 한다.
"""
import json

from fastapi.testclient import TestClient

from app.core.upload import MAX_UPLOAD_BYTES
from app.main import MAX_JSON_BODY_BYTES, app
from test_narrative_router import _sample_request

client = TestClient(app)


def test_oversized_json_body_is_rejected_before_parsing():
    """[G-03] 상한을 넘는 JSON 본문은 413 으로 즉시 거절된다."""
    n = 2500
    payload = {
        "task_type": "binary",
        "report_purpose": "internal",
        "fact_sheet": {
            "n_samples": 200, "dropped_rows": 0, "verdict": "PASS", "score": 90.0,
            "confusion": {"labels": [str(i) for i in range(n)],
                          "matrix": [[1] * n for _ in range(n)]},
        },
    }
    raw = json.dumps(payload)
    assert len(raw.encode()) > MAX_JSON_BODY_BYTES
    r = client.post("/api/generate-narrative", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 413, f"{r.status_code} {r.text[:200]}"


def test_normal_json_request_still_passes():
    """[G-03] 정상 크기 서술 요청은 그대로 통과한다."""
    r = client.post("/api/generate-narrative",
                    json=json.loads(_sample_request().model_dump_json()))
    assert r.status_code == 200, r.text[:200]


def test_multipart_upload_is_not_capped_by_the_json_limit():
    """[G-03] 멀티파트는 JSON 상한이 아니라 업로드 상한(20 MiB)을 따라야 한다.

    두 상한을 하나로 합치면 정상 데이터셋 업로드가 JSON 상한에 걸려 죽는다.
    멀티파트는 라우터의 공용 가드(G-04a)가 청크 단위로 읽으며 막는다.
    """
    import io

    from router_cases import request_json

    app.state.openai_client = None
    row = b"1,1,1\n"
    # JSON 상한보다는 크고 업로드 상한보다는 작은 크기
    size = MAX_JSON_BODY_BYTES + 1_000_000
    assert size < MAX_UPLOAD_BYTES
    csv = b"id,y_true,y_pred\n" + row * (size // len(row))
    r = client.post(
        "/api/evaluate",
        files={"file": ("d.csv", io.BytesIO(csv), "text/csv")},
        data={"data": request_json("binary")},
    )
    assert r.status_code != 413, "정상 크기 업로드가 JSON 상한에 걸렸다"


def test_request_without_content_length_is_not_rejected():
    """[G-03] Content-Length 가 없으면(chunked) 미들웨어는 판단하지 않는다.

    이 경로는 필드 단위 상한이 계속 막는다 — 미들웨어는 앞단 방어일 뿐이다.
    """
    r = client.post("/api/generate-narrative",
                    json=json.loads(_sample_request().model_dump_json()))
    assert r.status_code == 200
