"""tests/test_fact_sheet_contract.py — 사실 시트 계약(facts-1)의 백엔드 쪽.

계약 원본은 라운드 폴더 NARRATIVE_ROUND/사실시트_계약.json 이고, 이 저장소의 사본이 tests/contract/ 에 있다
(프론트 Capstone_Front/src/test/contract/ 에도 같은 사본). 예시 요청(fact_sheet_*.json)은 프론트가 백엔드 골든
평가 응답 세 개로 조립해 만든 것이다 — 이 테스트는 각 저장소가 자기 사본으로 CI 에서 돈다.

지키는 것:
- 예시가 NarrativeRequest 로 검증되고, **조용히 버려지는 키가 없다**(S2 의 value_text·gain_text·display 가 모르는
  키라 버려지던 일).
- 계약 상수 = 백엔드 상수, 계약의 facts 칸 목록 = 스키마 칸 목록.
- 운영 main 프론트 모양(facts 없음)의 요청도 옛 경로·작업 경로 모두 200.
- 두 저장소 사본이 같은지는 옆 저장소가 있을 때만 본다(skipIf).
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.narrative import schemas
from app.narrative.schemas import NarrativeRequest

CONTRACT = Path(__file__).parent / "contract"
FRONT_CONTRACT = Path(__file__).parents[2] / "Capstone_Front" / "src" / "test" / "contract"
ROUND_CONTRACT = Path(__file__).parents[2] / "NARRATIVE_ROUND" / "사실시트_계약.json"
EXAMPLES = ("binary", "multiclass", "multilabel")
FILES = ("fact_sheet_contract.json", *(f"fact_sheet_{t}.json" for t in (*EXAMPLES, "legacy_main")))


def _load(name: str) -> dict:
    return json.loads((CONTRACT / name).read_text(encoding="utf-8"))


CONTRACT_SPEC = _load("fact_sheet_contract.json")


def _dropped_keys(raw, dumped, path="") -> list[str]:
    """raw 에 있는데 검증·직렬화 뒤 사라진 키(점 표기). 목록은 같은 자리끼리 본다."""
    out: list[str] = []
    if isinstance(raw, dict):
        if not isinstance(dumped, dict):
            return [path or "<root>"]
        for key, value in raw.items():
            here = f"{path}.{key}" if path else key
            if key not in dumped:
                out.append(here)
            else:
                out.extend(_dropped_keys(value, dumped[key], here))
    elif isinstance(raw, list) and isinstance(dumped, list):
        for i, (a, b) in enumerate(zip(raw, dumped)):
            out.extend(_dropped_keys(a, b, f"{path}[{i}]"))
    return out


@pytest.mark.parametrize("name", [*EXAMPLES, "legacy_main"])
def test_example_validates_without_dropping_keys(name):
    raw = _load(f"fact_sheet_{name}.json")
    req = NarrativeRequest.model_validate(raw)
    assert _dropped_keys(raw, req.model_dump(mode="json")) == []


@pytest.mark.parametrize("name", EXAMPLES)
def test_new_examples_carry_facts_and_display_texts(name):
    fs = NarrativeRequest.model_validate(_load(f"fact_sheet_{name}.json")).fact_sheet
    assert fs.facts is not None and fs.facts.contract == CONTRACT_SPEC["contract"]
    assert fs.facts.verdict_detail is not None and fs.facts.verdict_detail.verdict == fs.verdict
    assert fs.facts.weakest_class is not None
    assert all(m.metric_id not in ("M21", "M22") for m in fs.metrics)
    assert fs.imbalance_effect is not None and fs.imbalance_effect.display is not None
    judged = [m for m in fs.metrics if m.threshold is not None]
    assert judged and all(m.value_text and m.margin_text and m.higher_is_better is not None for m in judged)


def test_legacy_example_has_no_facts():
    fs = NarrativeRequest.model_validate(_load("fact_sheet_legacy_main.json")).fact_sheet
    assert fs.facts is None
    assert {m.metric_id for m in fs.metrics} >= {"M21", "M22"}


def test_contract_constants_match_backend():
    k = CONTRACT_SPEC["constants"]
    pairs = {
        name: getattr(schemas, name)
        for name in k
        if not name.startswith("$") and hasattr(schemas, name)
    }
    assert pairs == {name: k[name] for name in pairs}
    # 상한 상수는 계약에 모두 있다 — 한쪽에만 생기면 실패(프론트 전용 예산·문제 근거 상한 이름 제외).
    backend_caps = {n for n in dir(schemas) if n.startswith("MAX_")}
    backend_only = backend_caps - set(k)
    assert backend_only == {"MAX_CLIENT_INPUTS_CHARS", "MAX_DISPLAY_TEXT_CHARS"}


def test_contract_benchmark_names_match_baselines():
    """고정 문구 16 — 프론트가 들고 있는 기준표 지표 이름 사본(계약 constants)이 BASELINES 키와 같다(순서까지)."""
    from app.narrative.baselines import BASELINES

    names = CONTRACT_SPEC["constants"]["benchmark_metric_names"]
    assert set(names) == set(BASELINES)
    for task, table in BASELINES.items():
        assert names[task] == list(table), task


def test_contract_facts_fields_match_schema():
    declared = {key for key in CONTRACT_SPEC["facts"] if not key.startswith("$")}
    assert declared == set(schemas.NarrativeFactsFact.model_fields)


@pytest.mark.parametrize("name", ["legacy_main", *EXAMPLES])
@pytest.mark.parametrize("route", ["/api/generate-narrative", "/api/narrative/jobs"])
def test_old_and_new_shapes_are_accepted_by_both_routes(name, route):
    prev = getattr(app.state, "anthropic_client", None)
    app.state.anthropic_client = None  # 키 없음 → 규칙 문장(결정적)
    try:
        resp = TestClient(app).post(route, json=_load(f"fact_sheet_{name}.json"))
    finally:
        app.state.anthropic_client = prev
    assert resp.status_code == 200, resp.text


@pytest.mark.skipif(not FRONT_CONTRACT.exists(), reason="옆 저장소(Capstone_Front)가 없다 — CI 는 저장소 하나만 받는다")
@pytest.mark.parametrize("name", FILES)
def test_frontend_copy_is_identical(name):
    ours = json.loads((CONTRACT / name).read_text(encoding="utf-8"))
    theirs = json.loads((FRONT_CONTRACT / name).read_text(encoding="utf-8"))
    assert ours == theirs


@pytest.mark.skipif(not ROUND_CONTRACT.exists(), reason="라운드 폴더가 없다")
def test_round_folder_contract_is_identical():
    assert json.loads(ROUND_CONTRACT.read_text(encoding="utf-8")) == CONTRACT_SPEC
