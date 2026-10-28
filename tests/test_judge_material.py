"""tests/test_judge_material.py — 판 4 서술 요청의 심판 읽기 재료(judge_material, 심판 라운드 J2 ①, 계획 10-1).

- 읽기는 app.judge.material.parse_material 한 곳: 없음·너무 큼·형식 틀림은 재료만 버린다(문제 이름을 돌려줌).
- 요청 검증은 이 칸 때문에 422·줄이기(input_too_large)·facts 버림으로 가지 않는다.
- 서술 생성은 이 칸을 읽지 않는다: 재료에 심은 표식이 Claude 에 보낸 어떤 인자에도 없고, 재료가 있든 없든 보낸 지시문이 같다.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, configure_sqlite
from app.judge.material import MAX_MATERIAL_CHARS, parse_material
from app.main import app
from app.narrative import jobs

DEV = Path(__file__).parent / "data" / "judge_dev"
MARK = "ZQX심판표식"


def _material(**over) -> dict:
    m = {
        "version": "judge-material-1",
        "material_check_version": "점검표-1",
        "segments": [
            {"id": "f-4.4-boundary", "slot": "7.1", "place": "4.4", "kind": "fixed",
             "text": f"경계 오류는 12건이다. {MARK}", "numbers": [{"text": "12", "field": "error_ratios.boundary_count"}]},
            {"id": "t-6-recommend", "slot": "9", "place": "6", "kind": "table", "header": ["번호", "권고"],
             "rows": [["1", f"재학습 {MARK}"]], "numbers": []},
        ],
    }
    m.update(over)
    return m


def _raw(**material) -> dict:
    raw = json.loads((DEV / "dev_binary.request.json").read_text(encoding="utf-8"))
    raw["judge_material"] = _material(**material)
    return raw


# ── 읽기 ──


def test_parse_ok():
    mat, problem = parse_material(_material())
    assert problem is None
    assert mat.segments[0].when == "always"
    rec = {"id": "r-6", "slot": "9", "place": "6", "kind": "recommend_rows", "header": ["우선순위", "대응 문제"],
           "rows": [["1순위", "P1 · 소수 클래스"]], "row_keys": ["P1"], "numbers": []}
    obs = {"id": "f-15", "slot": "7.3", "place": "5.6", "kind": "fixed", "when": "slot_empty", "text": "해당 없음"}
    ok, problem = parse_material(_material(segments=[rec, obs]))
    assert problem is None and ok.segments[0].row_keys == ["P1"] and ok.segments[1].when == "slot_empty"
    assert [s.id for s in mat.segments] == ["f-4.4-boundary", "t-6-recommend"]
    assert mat.segments[0].numbers[0].field == "error_ratios.boundary_count"


@pytest.mark.parametrize("raw, problem", [
    (None, "missing"),
    ([], "invalid"),
    ("글", "invalid"),
    (_material(version="judge-material-2"), "invalid"),
    (_material(extra_key=1), "invalid"),
    ({**_material(), "segments": [{**_material()["segments"][0], "slot": "10"}]}, "invalid"),
    ({**_material(), "segments": [_material()["segments"][0]] * 2}, "invalid"),  # id 겹침
    ({**_material(), "segments": [{**_material()["segments"][1], "rows": [["x"] * 17]}]}, "invalid"),
    ({**_material(), "segments": [{**_material()["segments"][0], "header": ["a"]}]}, "invalid"),  # 고정 문구에 표 칸
    ({**_material(), "segments": [{**_material()["segments"][1], "header": []}]}, "invalid"),  # 표에 머리 없음
    ({**_material(), "segments": [{**_material()["segments"][1], "kind": "recommend_rows"}]}, "invalid"),  # 문제 번호 없음
    ({**_material(), "segments": [{**_material()["segments"][1], "row_keys": ["P1"]}]}, "invalid"),
    ({**_material(), "segments": [{**_material()["segments"][0], "when": "sometimes"}]}, "invalid"),
    (_material(material_check_version="x" * (MAX_MATERIAL_CHARS + 1)), "too_large"),
])
def test_parse_problems(raw, problem):
    assert parse_material(raw) == (None, problem)


# ── 요청 검증은 그대로 ──


@pytest.mark.parametrize("junk", [[1, 2], "글", {"version": "x"}, {"segments": "x" * (MAX_MATERIAL_CHARS + 10)}])
def test_request_with_bad_material_still_parses_without_repairs(junk):
    raw = json.loads((DEV / "dev_binary.request.json").read_text(encoding="utf-8"))
    raw["judge_material"] = junk
    req, repairs = jobs.parse_request(raw)
    assert repairs == []  # 줄이기·facts 버림 없음
    assert req.fact_sheet.facts is not None and req.judge_material == junk


# ── 서술 생성은 재료를 읽지 않는다 ──


class _Recorder:
    """받은 인자를 적고 거절로 답하는 가짜 Anthropic(그 장은 규칙 문장으로 끝난다)."""

    def __init__(self):
        self.sent: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def with_options(self, **_kwargs):
        return self

    async def _create(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(model="recording", stop_reason="refusal", stop_details=None, content=[],
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0, iterations=None))


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


def _sent_prompts(api, raw) -> list[str]:
    rec = _Recorder()
    app.state.anthropic_client = rec  # conftest 가 테스트마다 None 으로 되돌린다
    created = api.post("/api/narrative/jobs", json=raw)
    assert created.status_code == 200, created.text
    view = api.get(f"/api/narrative/jobs/{created.json()['job_id']}").json()
    assert view["status"] == "done"
    assert rec.sent, "가짜 클라이언트가 한 번도 불리지 않았다"
    return sorted(json.dumps(kw, ensure_ascii=False, sort_keys=True, default=str) for kw in rec.sent)


def test_material_never_reaches_generation_prompts(api):
    with_material = _sent_prompts(api, _raw())
    assert all(MARK not in s for s in with_material)
    plain = copy.deepcopy(_raw())
    del plain["judge_material"]
    assert _sent_prompts(api, plain) == with_material


# ── 계약(tests/contract/judge_material_ed4.json)과 실제 판 4 요청 예시 ──

CONTRACT = Path(__file__).parent / "contract"
FRONT_COPY = Path(__file__).resolve().parents[2] / "Capstone_Front" / "src" / "test" / "contract" / "judge_material_ed4.json"
MATERIAL_SHA256 = "bc32c9ecdfec3f3fe4de03a254525a361e3b534dae53b94df780f2de33474e63"


def test_material_contract_pinned():
    import hashlib

    assert hashlib.sha256((CONTRACT / "judge_material_ed4.json").read_bytes()).hexdigest() == MATERIAL_SHA256


@pytest.mark.skipif(not FRONT_COPY.exists(), reason="옆 저장소(Capstone_Front)가 없다 — CI 는 저장소 하나만 받는다")
def test_material_contract_front_copy_identical():
    assert FRONT_COPY.read_bytes() == (CONTRACT / "judge_material_ed4.json").read_bytes()


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_edition4_examples_carry_material_that_parses(task):
    raw = json.loads((CONTRACT / f"fact_sheet_{task}_ed4.json").read_text(encoding="utf-8"))
    assert list(raw)[-1] == "judge_material"
    mat, problem = parse_material(raw["judge_material"])
    assert problem is None
    ids = [s.id for s in mat.segments]
    assert {"4.2-table", "4.2-basis", "4.4-counts", "4.4-fixed17", "5.4-b18", "5.6-fixed15"} <= set(ids)
    assert any(s.kind == "recommend_rows" for s in mat.segments)
    assert all(s.slot != "9" or s.place.startswith("6") for s in mat.segments)
    vocab = json.loads((CONTRACT / "judge_material_ed4.json").read_text(encoding="utf-8"))["number_fields"]
    prefixes = {k.split("<")[0] for k in vocab}
    fields = [n.field for s in mat.segments for n in s.numbers]
    assert fields and all(any(f.startswith(p) for p in prefixes) for f in fields)
