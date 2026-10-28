"""tests/test_j1_item2_contract.py — 보고 항목 2(평가 데이터 출처) 제공 판정을 프론트와 같은 대조 표로 본다.

심판 라운드 1단계 통합 확인 뒤 고침(2026-10-28). 8.2 '평가 데이터의 대표성과 한계'(8.2-14, 판 4 는 5.1)는 보고 항목 2 가
제공되었을 때만 나온다(required_items._eval_data_source). 같은 판정을 프론트 재료 점검(materialCheck 항목 2 → 성적서 1.3, 판 3 은
1.4)이 따로 계산한다 — 서술 요청에 재료 점검 결과가 실리지 않기 때문이다. 둘이 갈리면 1.3 이 '판단할 수 없다'고 찍은 성적서에
대표성 해석이 실리거나 그 반대가 된다.

`tests/contract/reporting_item2_cases.json` 이 대조 표다. 프론트 src/test/contract/ 에 같은 바이트가 있고, 프론트
src/lib/report/reportingItem2Contract.test.ts 가 같은 입력에서 앱이 실제로 보내는 test_data 가 표와 같은지와 자기 판정을 본다.
이 파일은 표의 test_data 를 계약 예시 요청에 넣어 8.2-14 가 나오는지가 provided.backend 인지 본다(판 3·4, 세 유형).
표를 바꾸면 두 사본과 두 시험의 sha256 을 함께 고친다. 옆 저장소 바이트 대조는 옆 저장소가 있을 때만 한다(CI 는 저장소 하나).
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from app.narrative.required_items import select_items
from app.narrative.schemas import NarrativeRequest

TESTS = Path(__file__).parent
CONTRACT = TESTS / "contract"
NAME = "reporting_item2_cases.json"
FRONT_COPY = TESTS.parents[1] / "Capstone_Front" / "src" / "test" / "contract" / NAME

# 심판 라운드 1단계 통합 확인 뒤 고침(2026-10-28)에 만든 바이트. 프론트 reportingItem2Contract.test.ts 와 같은 값이다.
SHA256 = "2ee408f7b451d9a8ed110054ce738445d18f917ea2be7a52b51cfb2a88d935fa"

TABLE = json.loads((CONTRACT / NAME).read_text(encoding="utf-8"))
CASES = TABLE["cases"]
TASKS = ("binary", "multiclass", "multilabel")
EDITIONS = (3, 4)
ITEM = "8.2-14"


def _request(task: str, edition: int, test_data: dict) -> NarrativeRequest:
    """계약 예시(판 3 fact_sheet_{task}, 판 4 fact_sheet_{task}_ed4)의 고객 입력 test_data 만 표의 값으로 바꾼 요청."""
    name = f"fact_sheet_{task}" + ("_ed4" if edition == 4 else "")
    raw = json.loads((CONTRACT / f"{name}.json").read_text(encoding="utf-8"))
    inputs = copy.deepcopy(raw.get("client_inputs") or {})
    inputs["test_data"] = copy.deepcopy(test_data)
    raw["client_inputs"] = inputs
    req = NarrativeRequest.model_validate(raw)
    assert req.report_edition == edition and req.task_type.value == task
    return req


def _picked(req: NarrativeRequest) -> dict:
    out = select_items(req.fact_sheet, req.client_inputs, req.task_type.value, edition=req.report_edition)
    assert out, "결론 목록이 비었다"
    return {c.item_id: c for c in out}


def test_table_is_pinned_by_sha256():
    assert hashlib.sha256((CONTRACT / NAME).read_bytes()).hexdigest() == SHA256


@pytest.mark.skipif(not FRONT_COPY.exists(), reason="옆 저장소(Capstone_Front)가 없다 — CI 는 저장소 하나만 받는다")
def test_frontend_copy_is_byte_identical():
    assert FRONT_COPY.read_bytes() == (CONTRACT / NAME).read_bytes()


def test_table_shape():
    # 빈 표를 돌면 늘 통과한다 — 줄이 있고 제공·빠짐이 둘 다 있는지 먼저 본다.
    assert len(CASES) >= 10
    assert {c["provided"]["backend"] for c in CASES} == {True, False}
    assert len({c["name"] for c in CASES}) == len(CASES)
    gaps = [c["name"] for c in CASES if c["provided"]["frontend"] != c["provided"]["backend"]]
    assert gaps == ["출처 글 '제공되지 않음'(직접 적음) + 같은 출처"]
    assert [(c["name"], bool((c.get("pending") or "").strip())) for c in CASES] == [
        (c["name"], c["name"] in gaps) for c in CASES
    ]


@pytest.mark.parametrize("edition", EDITIONS)
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_backend_judgment_matches_table(case, task, edition):
    picked = _picked(_request(task, edition, case["test_data"]))
    assert (ITEM in picked) is case["provided"]["backend"]
    if case["provided"]["backend"]:
        # 규칙 템플릿은 의뢰자가 적은 출처 글(앞뒤 공백 뗌)을 그대로 인용한다.
        source = case["test_data"]["평가 데이터 출처"].strip()
        assert picked[ITEM].template_text == f"의뢰자가 밝힌 평가 데이터 출처는 '{source}'이다."
