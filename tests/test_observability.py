"""[G-06] 조용한 강등이 로그에 남아야 한다.

종전 상태: app/ 전체에 print 6건 · logging 1지점(narrative 조립 실패)뿐이었고,
품질을 떨어뜨리는 세 갈래 강등이 **아무 흔적도 남기지 않았다**.

  · LLM 컬럼 매핑 실패 → 규칙 폴백      (analysis_service)
  · LLM 서술 호출 실패 → 규칙 폴백      (narrative/service)
  · grounding 위반    → 규칙 폴백      (narrative/service)

같은 라운드의 G-03/G-04 실측에서 18.8 MB 남용 요청이 200 + 규칙 폴백으로 끝나고
서버에 아무 것도 남지 않는 것을 관측했다. 남용이 탐지 불가능하다는 뜻이다.
로깅은 탐지를 만드는 것이 아니라 탐지의 전제를 만든다.
"""
import json
import logging

import pandas as pd
import pytest

from app.analysis.analysis_service import resolve_column_mapping
from app.core.schemas import TaskType
from app.narrative.service import generate_narrative
from test_narrative_router import _sample_request

_DF = pd.DataFrame({"id": [1, 2], "y_true": [0, 1], "y_pred": [1, 1]})


async def test_llm_mapping_failure_is_logged(make_fake_openai_client, caplog):
    """[G-06] LLM 컬럼 매핑이 실패해 규칙 폴백으로 내려간 사실이 로그에 남는다."""
    fake = make_fake_openai_client(raise_exc=RuntimeError("boom"))
    with caplog.at_level(logging.WARNING):
        await resolve_column_mapping(fake, TaskType.binary, list(_DF.columns), _DF)
    assert any(r.levelno >= logging.WARNING for r in caplog.records), "로그가 하나도 없다"
    assert any("boom" in r.getMessage() for r in caplog.records), (
        f"실패 원인이 기록되지 않았다: {[r.getMessage() for r in caplog.records]}"
    )


async def test_no_key_fallback_is_logged(caplog):
    """[G-06] 키 없음으로 규칙 폴백을 타는 것도 흔적이 남아야 한다."""
    with caplog.at_level(logging.WARNING):
        await resolve_column_mapping(None, TaskType.binary, list(_DF.columns), _DF)
    assert any(r.levelno >= logging.WARNING for r in caplog.records), "로그가 하나도 없다"


async def test_narrative_api_error_fallback_is_logged(make_fake_openai_client, caplog):
    """[G-06] LLM 서술 호출 실패 → 폴백이 로그에 남는다."""
    fake = make_fake_openai_client(raise_exc=RuntimeError("upstream down"))
    with caplog.at_level(logging.WARNING):
        resp = await generate_narrative(fake, _sample_request())
    assert resp.meta.reason == "api_error"
    assert any("api_error" in r.getMessage() for r in caplog.records), (
        f"강등 사유가 기록되지 않았다: {[r.getMessage() for r in caplog.records]}"
    )


async def test_grounding_violation_fallback_is_logged(make_fake_openai_client, caplog):
    """[G-06] grounding 위반으로 폴백할 때 **어떤 숫자가 걸렸는지** 남아야 한다.

    위반 토큰을 남기지 않으면 '환각이 있었다'는 사실만 알고 무엇이었는지 모른다.
    """
    llm_json = {
        "interpretation": {"confusion_analysis": "정확도는 99.9% 였다.", "distribution_analysis": ""},
        "conclusion": {"benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": [],
    }
    fake = make_fake_openai_client(llm_json)
    with caplog.at_level(logging.WARNING):
        resp = await generate_narrative(fake, _sample_request())
    assert resp.meta.reason == "grounding_failed"
    messages = [r.getMessage() for r in caplog.records]
    assert any("grounding" in m for m in messages), f"강등 사유가 없다: {messages}"
    assert any("99.9" in m for m in messages), f"위반 토큰이 기록되지 않았다: {messages}"


def test_no_print_left_in_production_code():
    """[G-06] 프로덕션 경로에 print 가 남아 있으면 안 된다.

    print 는 레벨도 없고 필터링도 안 되며 caplog 로 검증할 수도 없다.
    (scripts/ 의 수동 스모크는 대상이 아니다 — app/ 만 본다.)
    """
    import pathlib
    import re

    offenders = []
    for path in pathlib.Path("app").rglob("*.py"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.match(r"\s*print\(", line):
                offenders.append(f"{path}:{i}")
    assert offenders == [], f"print 잔존: {offenders}"


def test_parse_failure_records_cause(caplog):
    """[G-06] 파싱 실패를 상태코드로만 바꾸지 말고 원인을 기록해야 한다."""
    import io

    from fastapi.testclient import TestClient

    from app.main import app
    from router_cases import request_json

    app.state.openai_client = None
    with caplog.at_level(logging.WARNING):
        r = TestClient(app).post(
            "/api/evaluate",
            files={"file": ("broken.csv", io.BytesIO(b'a,b\n"unterminated\n'), "text/csv")},
            data={"data": request_json("binary")},
        )
    assert r.status_code == 422
    assert any("파싱 실패" in rec.getMessage() for rec in caplog.records), (
        f"원인이 기록되지 않았다: {[rec.getMessage() for rec in caplog.records]}"
    )
