"""[G-03][E-18] OpenAI 과금 엔드포인트 2개의 입력 상한 · 재시도 예산 · 동시 호출 상한.

/api/analyze-columns 와 /api/generate-narrative 는 무인증이고 프로젝트 OpenAI 계정에
과금한다. 종전 실측 —

- analyze-columns: 컬럼 2,000개 x 셀 200자 → user prompt 2,102,678자(~525k 토큰).
  행은 df.head(30) 으로 제한되지만 **컬럼 수와 셀 길이에는 상한이 전혀 없었다.**
- generate-narrative: metrics/per_class 각 20,000개 → 5,369,661자(~1.34M 토큰).
  파일 업로드조차 필요 없는 순수 JSON 경로다.
- llm_mapper 의 blanket 재시도가 SDK max_retries=2 와 곱해져 익명 요청 1건의
  과금 호출이 최대 6회, 최악 벽시계가 약 270초였다(E-18).
"""
import json

import httpx
import openai
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.analysis.llm_mapper import analyze_columns_with_llm
from app.analysis.prompt_builder import build_user_prompt, MAX_PROMPT_CHARS
from app.core import concurrency
from app.core.schemas import TaskType
from app.main import app
from app.narrative import schemas as narrative_schemas
from test_narrative_router import _sample_request

client = TestClient(app)


@pytest.fixture(autouse=True)
def _force_fallback():
    prev = getattr(app.state, "openai_client", None)
    app.state.openai_client = None
    yield
    app.state.openai_client = prev


def _payload(**overrides) -> dict:
    body = json.loads(_sample_request().model_dump_json())
    body["fact_sheet"].update(overrides)
    return body


# ── 서술 경로 입력 상한 ────────────────────────────────────────────────────

def test_narrative_rejects_oversized_metrics():
    """[G-03] metrics 배열에 상한이 있어야 한다 (종전: 무제한 → 200)."""
    huge = [
        {"metric_id": f"M{i}", "display_name": "x", "value": 0.5, "status": "pass"}
        for i in range(narrative_schemas.MAX_METRIC_FACTS + 1)
    ]
    r = client.post("/api/generate-narrative", json=_payload(metrics=huge))
    assert r.status_code == 422, r.status_code


def test_narrative_rejects_oversized_per_class():
    """[G-03] per_class 배열에 상한이 있어야 한다."""
    huge = [{"label": str(i)} for i in range(narrative_schemas.MAX_CLASS_ENTRIES + 1)]
    r = client.post("/api/generate-narrative", json=_payload(per_class=huge))
    assert r.status_code == 422, r.status_code


def test_narrative_rejects_oversized_confusion_matrix():
    """[G-03] 혼동행렬 차원에 상한이 있어야 한다.

    build_number_whitelist 가 행x열 이중 루프라 차원에 대해 제곱으로 커진다.
    실측: 2500x2500 입력이 이벤트 루프를 15.1 초 막았다.
    """
    n = narrative_schemas.MAX_CLASS_ENTRIES + 1
    conf = {"labels": [str(i) for i in range(n)], "matrix": [[0] * n for _ in range(n)]}
    r = client.post("/api/generate-narrative", json=_payload(confusion=conf))
    assert r.status_code == 422, r.status_code


def test_narrative_rejects_ragged_confusion_matrix():
    """[G-03] 행별 길이도 검사해야 한다 — 바깥 길이만 재면 한 행에 다 밀어넣을 수 있다."""
    n = narrative_schemas.MAX_CLASS_ENTRIES + 1
    conf = {"labels": ["0", "1"], "matrix": [[0] * n, [0] * n]}
    r = client.post("/api/generate-narrative", json=_payload(confusion=conf))
    assert r.status_code == 422, r.status_code


def test_narrative_rejects_oversized_distribution():
    """[G-03] class_distribution 키 개수에도 상한이 있어야 한다."""
    n = narrative_schemas.MAX_CLASS_ENTRIES + 1
    dist = {"class_distribution": {str(i): 1 for i in range(n)}, "imbalance_ratio": 1.0}
    r = client.post("/api/generate-narrative", json=_payload(distribution=dist))
    assert r.status_code == 422, r.status_code


def test_narrative_accepts_normal_sized_request():
    """[G-03] 정상 크기 요청은 그대로 통과해야 한다 — 상한이 정상 사용을 막으면 안 된다."""
    r = client.post("/api/generate-narrative", json=json.loads(_sample_request().model_dump_json()))
    assert r.status_code == 200, r.text[:200]


# ── 컬럼 분석 프롬프트 증폭 ────────────────────────────────────────────────

def test_analyze_user_prompt_is_bounded():
    """[G-03] 컬럼 수·셀 길이가 커도 프롬프트 길이가 상한 안에 머물러야 한다.

    종전: 컬럼 2,000개 x 셀 200자 → 2,102,678자(~525k 토큰), 상한 없음.
    """
    n_cols = 2000
    cell = "x" * 200
    df = pd.DataFrame({f"col{i}": [cell] * 3 for i in range(n_cols)})
    prompt = build_user_prompt(list(df.columns), df)
    assert len(prompt) <= MAX_PROMPT_CHARS, f"프롬프트 {len(prompt):,}자 > 상한 {MAX_PROMPT_CHARS:,}자"


def test_analyze_user_prompt_keeps_normal_dataset_intact():
    """[G-03] 정상 데이터셋은 절단되지 않아야 한다 — 상한이 매핑 품질을 깎으면 안 된다."""
    df = pd.DataFrame({"id": [1, 2], "y_true": [0, 1], "y_pred": [1, 1], "score": [0.1, 0.9]})
    prompt = build_user_prompt(list(df.columns), df)
    for col in df.columns:
        assert f"{col}:" in prompt


# ── 재시도 예산 (E-18) ─────────────────────────────────────────────────────

async def test_analyze_does_not_retry_on_timeout(make_fake_openai_client):
    """[E-18] 타임아웃은 재시도 예산을 한 번 더 쓰면 안 된다 (종전: 2회 호출).

    SDK max_retries=2 와 곱해져 익명 요청 1건의 최악 대기가 약 270초였다.
    """
    fake = make_fake_openai_client(raise_exc=openai.APITimeoutError(request=None))
    df = pd.DataFrame({"id": [1], "y_true": [0], "y_pred": [1]})
    with pytest.raises(openai.APITimeoutError):
        await analyze_columns_with_llm(fake, TaskType.binary, list(df.columns), df)
    assert fake.chat.completions.create.call_count == 1, (
        f"타임아웃에 재시도했다 — {fake.chat.completions.create.call_count}회 호출"
    )


async def test_analyze_still_retries_on_schema_rejection(make_fake_openai_client):
    """[E-18] 동적 enum strict 스키마 거부에 대한 1회 재시도(D5a)는 유지되어야 한다."""
    rejected = httpx.Response(
        400, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )
    fake = make_fake_openai_client(
        raise_exc=openai.BadRequestError(message="invalid schema", response=rejected, body=None)
    )
    df = pd.DataFrame({"id": [1], "y_true": [0], "y_pred": [1]})
    with pytest.raises(openai.BadRequestError):
        await analyze_columns_with_llm(fake, TaskType.binary, list(df.columns), df)
    assert fake.chat.completions.create.call_count == 2, (
        f"스키마 거부 재시도가 사라졌다 — {fake.chat.completions.create.call_count}회 호출"
    )


def test_llm_concurrency_cap_value_is_pinned():
    """[G-03] 동시 LLM 호출 상한 — 워커 수를 몰라도 안전한 유일한 총량 장치다.

    워커가 N개면 실효 상한이 N배로 느슨해질 뿐 정상 사용자를 잘못 막지 않는다
    (과허용 방향으로만 틀린다).
    """
    assert concurrency.MAX_CONCURRENT_LLM_CALLS == 4
