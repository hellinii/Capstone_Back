"""tests/test_p4_benchmark_ed4.py — 판 4 라운드 S2 ⑧: 판 4 출처 글과 참조 범위 위치 묶음(대응표 22, 부록 A10·B21).

- 출처 글: 판 4 는 예시본 4.6 ※ 줄 글(“…자체 설정 기준이며…”), 판 3 은 지금 글(“…사내 합의 보수 기준이며…”) 그대로.
  기준표의 위치·방향·품질 코드와 숫자는 판과 무관하다(출처 글만 다르다).
- 묶음: 판 4 는 수치 위치(position)로 — 참조 범위보다 높은/안에 드는/낮은 지표. 방향을 반영한 우열(우수·미흡)을 쓰지 않는다.
  낮을수록 좋은 지표(Hamming Loss)도 수치로만: 범위 아래면 '참조 범위보다 낮은 지표'(판 3 은 '우수한 지표').
- 이음은 B21 그대로(…이고, …이며, …이다), 없는 갈래는 뺀다. 값은 KPI 원값의 판 4 표기, 범위 끝은 둘째 자리.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi import BackgroundTasks
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, configure_sqlite
from app.narrative import baselines, generator, jobs
from app.narrative.schemas import NarrativeRequest

TESTS = Path(__file__).parent
CONTRACT = TESTS / "contract"
EXAMPLE_REPORT = TESTS.parents[1] / "P4_ROUND" / "원문" / "4_예시본.md"

# 예시본 4.6 ※ 줄(※ 와 띄어쓰기 뒤의 글) — 글자 그대로 옮김.
ED4_NOTE = "내부 참조 기준치 v1 (공개 벤치마크 평균이 아닌 자체 설정 기준이며, 도메인별로 상이할 수 있음)"
ED3_NOTE = "내부 참조 기준치 v1 (공개 벤치마크 평균이 아닌 사내 합의 보수 기준이며, 도메인별로 상이할 수 있음)"

# 예시본 5.4 의 비교 결과 괄호들과 같은 꼴(B21 문장, bc_strong).
BC_STRONG_B21 = (
    "내부 참조 기준과 비교하면 참조 범위보다 높은 지표는 Accuracy(0.982, 참조 범위 0.85~0.92), Precision(0.969, 참조 범위 "
    "0.80~0.92), Recall(0.984, 참조 범위 0.80~0.92), F1 Score(0.977, 참조 범위 0.83~0.91), AUROC(0.999, 참조 범위 "
    "0.85~0.95), AUPRC(0.999, 참조 범위 0.80~0.93)이다."
)


def _raw(name: str = "binary_ed4") -> dict:
    return json.loads((CONTRACT / f"fact_sheet_{name}.json").read_text(encoding="utf-8"))


def _conclusions(raw: dict) -> dict[str, str]:
    req = NarrativeRequest.model_validate(raw)
    conclusions = generator._prepare(req)[3]
    return {c.item_id: c.template_text for c in conclusions if c.section == "8.1" and c.template_text}


def test_source_notes_by_edition():
    assert baselines.SOURCE_NOTE == ED3_NOTE  # 판 3 글은 그대로
    assert baselines.SOURCE_NOTE_ED4 == ED4_NOTE
    assert baselines.SOURCE_NOTES == {3: ED3_NOTE, 4: ED4_NOTE}


@pytest.mark.skipif(not EXAMPLE_REPORT.exists(), reason="라운드 폴더(원문 사본)가 없다")
def test_edition4_note_is_the_example_report_line():
    lines = EXAMPLE_REPORT.read_text(encoding="utf-8").splitlines()
    assert f"※ {ED4_NOTE}" in lines


def test_refs_differ_only_in_source_note():
    req = NarrativeRequest.model_validate(_raw())
    ed3 = baselines.build_benchmark_refs("binary", req.fact_sheet.metrics, edition=3)
    ed4 = baselines.build_benchmark_refs("binary", req.fact_sheet.metrics, edition=4)
    assert len(ed3) == len(ed4) == 6
    assert {r["source_note"] for r in ed3} == {ED3_NOTE} and {r["source_note"] for r in ed4} == {ED4_NOTE}
    assert [{**r, "source_note": None} for r in ed3] == [{**r, "source_note": None} for r in ed4]


@pytest.fixture
def factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}", connect_args={"check_same_thread": False})
    configure_sqlite(engine)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, autocommit=False)
    engine.dispose()


async def test_created_job_returns_edition4_note(factory):
    created = await jobs.create_job(_raw(), client=None, session_factory=factory, background_tasks=BackgroundTasks())
    assert created.benchmark_refs and {r.source_note for r in created.benchmark_refs} == {ED4_NOTE}
    ed3_raw = _raw()
    ed3_raw.pop("report_edition")
    created3 = await jobs.create_job(ed3_raw, client=None, session_factory=factory, background_tasks=BackgroundTasks())
    assert {r.source_note for r in created3.benchmark_refs} == {ED3_NOTE}


async def test_rule_text_8_1_carries_edition4_note():
    req = NarrativeRequest.model_validate(_raw())
    gen = await generator.generate_sections(None, req)
    text = gen.response.conclusion.benchmark
    assert text.startswith(BC_STRONG_B21)
    assert f"비교 기준은 {ED4_NOTE}이다." in text
    assert ED3_NOTE not in text


def test_bc_strong_b21_sentence():
    got = _conclusions(_raw())
    assert got["8.1-01"] == BC_STRONG_B21
    assert got["8.1-02"] == (
        "목표값이 참조 범위보다 높은 지표는 Accuracy(목표 0.950), Recall(목표 0.970), F1 Score(목표 0.950), AUROC(목표 0.980)이다."
    )


def _binary_positions(values: dict[str, float]) -> dict:
    """판 4 예시의 지표 값을 바꿔 위치를 골라 만든 요청 — 참고 지표만 바꿔 판정 표기와 엮이지 않게 한다."""
    raw = _raw()
    for m in raw["fact_sheet"]["metrics"]:
        if m["display_name"] in values:
            assert m["threshold"] is None, m["display_name"]
            m["value"] = values[m["display_name"]]
    return raw


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {"Precision": 0.85, "AUPRC": 0.70},
            "내부 참조 기준과 비교하면 참조 범위보다 높은 지표는 Accuracy(0.982, 참조 범위 0.85~0.92), Recall(0.984, 참조 범위 "
            "0.80~0.92), F1 Score(0.977, 참조 범위 0.83~0.91), AUROC(0.999, 참조 범위 0.85~0.95)이고, 참조 범위 안에 드는 지표는 "
            "Precision(0.850, 참조 범위 0.80~0.92)이며, 참조 범위보다 낮은 지표는 AUPRC(0.700, 참조 범위 0.80~0.93)이다.",
        ),
        (
            {"Precision": 0.79996, "AUPRC": 0.93},
            "내부 참조 기준과 비교하면 참조 범위보다 높은 지표는 Accuracy(0.982, 참조 범위 0.85~0.92), Recall(0.984, 참조 범위 "
            "0.80~0.92), F1 Score(0.977, 참조 범위 0.83~0.91), AUROC(0.999, 참조 범위 0.85~0.95)이고, 참조 범위 안에 드는 지표는 "
            "AUPRC(0.930, 참조 범위 0.80~0.93)이며, 참조 범위보다 낮은 지표는 Precision(0.79996, 참조 범위 0.80~0.92)이다.",
        ),
        (
            {"Precision": 0.70, "AUPRC": 0.70},
            "내부 참조 기준과 비교하면 참조 범위보다 높은 지표는 Accuracy(0.982, 참조 범위 0.85~0.92), Recall(0.984, 참조 범위 "
            "0.80~0.92), F1 Score(0.977, 참조 범위 0.83~0.91), AUROC(0.999, 참조 범위 0.85~0.95)이고, 참조 범위보다 낮은 지표는 "
            "Precision(0.700, 참조 범위 0.80~0.92), AUPRC(0.700, 참조 범위 0.80~0.93)이다.",
        ),
    ],
)
def test_b21_groups_and_joins(values, expected):
    assert _conclusions(_binary_positions(values))["8.1-01"] == expected


def _multilabel(hamming: float, edition: int) -> dict:
    from app.narrative.wording_ed4 import ed4_judged

    raw = _raw("multilabel")
    if edition == 4:
        raw["report_edition"] = 4
    [m] = [m for m in raw["fact_sheet"]["metrics"] if m["display_name"] == "Hamming Loss"]
    assert m["threshold"] == 0.6 and m["higher_is_better"] is False
    m["value"], m["status"] = hamming, "pass"
    m["value_text"] = ed4_judged(hamming, 0.6, False, True) if edition == 4 else f"{hamming:.3f}"
    m["margin"] = 0.6 - hamming
    m["margin_text"] = ed4_judged(0.6 - hamming, 0, True, True) if edition == 4 else None
    return raw


@pytest.mark.parametrize(("hamming", "head"), [(0.03, "참조 범위보다 낮은 지표는"), (0.25, "참조 범위보다 높은 지표는")])
def test_lower_is_better_is_grouped_by_numeric_position(hamming, head):
    text = _conclusions(_multilabel(hamming, 4))["8.1-01"]
    shown = f"Hamming Loss({hamming:.3f}, 참조 범위 0.05~0.20)"
    assert shown in text
    # 그 지표는 바로 앞 묶음 머리 아래에 있다(다른 묶음에 들어가지 않는다).
    before = text[: text.index(shown)]
    assert before.rindex(head) == max(before.rfind(h) for h in ("참조 범위보다 높은 지표는", "참조 범위 안에 드는 지표는", "참조 범위보다 낮은 지표는"))
    # 판 3 은 방향을 반영해 '우수/미흡'으로 묶는다(같은 입력에서 판이 실제로 갈린다).
    ed3 = _conclusions(_multilabel(hamming, 3))["8.1-01"]
    assert ("우수한 지표" if hamming < 0.05 else "미흡한 지표") in ed3


@pytest.mark.parametrize("name", ["binary", "multiclass", "multilabel"])
def test_no_evaluation_words_in_edition4_benchmark_text(name):
    raw = _raw(name)
    raw["report_edition"] = 4
    got = _conclusions(raw)
    assert got, name
    joined = " ".join(got.values())
    assert joined.startswith("내부 참조 기준과 비교하면")
    for word in ("우수", "미흡", "기준 범위 내", "더 나은", "부족"):
        assert word not in joined, (name, word)
    ed3 = _conclusions(_raw(name))
    assert ed3["8.1-01"] != got["8.1-01"]
