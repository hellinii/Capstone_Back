"""tests/test_judge_dev_fixtures.py — 심판 개발용 판 4 서술 요청 3건(심판 라운드 J0, 계획 17-6)을 지킨다.

`tests/data/judge_dev/` 의 세 파일은 프론트 앱이 실제로 보내는 판 4 서술 요청 본문이다(랜덤 예측 CSV — 이 폴더 README).
심판 개발(시험·프롬프트·파일럿)은 이 입력과 합성 문장만 쓴다. 검증용 6건(판 3 발급본과 같은 입력)은 저장소에 두지 않는다.

- 바이트 고정(sha256): 요청 모양이 바뀌면(심판 라운드 J2 의 읽기 재료 등) 프론트 도구로 다시 뜨고 이유와 함께 고친다.
- 판 4 요청이고 FactSheet 로 읽힌다. 랜덤 예측이다(이진 Accuracy 0.505).
- dev_binary 는 평가 데이터 출처 두 줄을 모두 싣는다(고객 입력만으로 단정하는 문장을 보려고 — 계획 Q2 ②).
- 키 없이(LLM 호출 없음) 작업 경로가 끝까지 돌아 규칙 문장 서술이 나온다 — 비용 없이 심판 코드를 짜는 바탕.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, configure_sqlite
from app.main import app
from app.narrative import jobs
from app.narrative.schemas import NarrativeRequest

DEV = Path(__file__).parent / "data" / "judge_dev"
SHA256 = {
    "dev_binary.request.json": "37ffed1cc1c4400226b90346b00ec402b7b7ebb2ecaa09aa9ee393ddecce3c90",
    "dev_multiclass.request.json": "05daa190d67c9a48dec61ffa966b96610ec709748c4bddbe8da1fcde1db58214",
    "dev_multilabel.request.json": "50631a22cd789cc8f2a787c1b5f0004129e837c818ef36e037630ecf4dd79807",
}
TASKS = {"dev_binary.request.json": "binary", "dev_multiclass.request.json": "multiclass",
         "dev_multilabel.request.json": "multilabel"}


def _raw(name: str) -> dict:
    return json.loads((DEV / name).read_text(encoding="utf-8"))


def test_folder_holds_exactly_the_pinned_requests():
    files = sorted(p.name for p in DEV.glob("*.json"))
    assert files == sorted(SHA256)
    actual = {n: hashlib.sha256((DEV / n).read_bytes()).hexdigest() for n in files}
    assert actual == SHA256


@pytest.mark.parametrize("name", sorted(SHA256))
def test_edition4_request_that_reads_as_fact_sheet(name):
    raw = _raw(name)
    req = NarrativeRequest.model_validate(raw)
    assert (req.report_edition, req.task_type.value) == (4, TASKS[name])
    assert len(req.fact_sheet.metrics) >= 5


def test_predictions_are_random():
    acc = {m["metric_id"]: m["value_text"] for m in _raw("dev_binary.request.json")["fact_sheet"]["metrics"]}
    assert acc["M1"] == "0.505"
    multi = {m["metric_id"]: m["value_text"] for m in _raw("dev_multiclass.request.json")["fact_sheet"]["metrics"]}
    assert multi["M1"] == "0.335"  # 세 클래스 — 우연 수준


def test_dev_binary_states_both_evaluation_data_lines():
    test_data = _raw("dev_binary.request.json")["client_inputs"]["test_data"]
    assert test_data["평가 데이터 출처"] != "제공되지 않음"
    assert test_data["학습 데이터와의 관계"].startswith("다른 출처")
    # 나머지 둘은 하나 이상 비어 있다(대표성 항목 조건이 갈린다)
    for name in ("dev_multiclass.request.json", "dev_multilabel.request.json"):
        assert "제공되지 않음" in _raw(name)["client_inputs"]["test_data"].values(), name


@pytest.fixture
def api(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}", connect_args={"check_same_thread": False})
    configure_sqlite(engine)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    app.dependency_overrides[jobs.get_session_factory] = lambda: factory
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(jobs.get_session_factory, None)
        engine.dispose()


@pytest.mark.parametrize("name", sorted(SHA256))
def test_job_runs_to_rule_sentences_without_keys(api, name):
    created = api.post("/api/narrative/jobs", json=_raw(name))
    assert created.status_code == 200, created.text
    view = api.get(f"/api/narrative/jobs/{created.json()['job_id']}").json()
    assert (view["status"], view["stage"], view["failure"]) == ("done", "done", None), view
    sources = view["result"]["section_sources"]
    assert set(sources.values()) <= {"fallback", "fixed"} and "fallback" in sources.values()
    assert view["result"]["interpretation"], "규칙 문장 서술이 비었다"
