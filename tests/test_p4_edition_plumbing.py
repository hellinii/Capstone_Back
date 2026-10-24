"""tests/test_p4_edition_plumbing.py — 판 4 라운드 S2 ②: 서술 요청의 판 신호(report_edition) 배관.

- 판 신호는 NarrativeRequest 최상위 report_edition 하나다. 선택 칸이고(판 3 요청에는 키가 없다 — 기본 3), 3·4 밖은 422.
- 안쪽 함수(기준표·허용 목록·결론 목록·지시문·사실 보기·검문·규칙 문장)는 기본값 없는 keyword-only edition 으로 받는다 —
  빠뜨리면 바로 오류다. 판 4 요청이면 작업 경로의 하위 호출이 **모두** edition=4 로 가는지 spy 로 본다(배선 시험).
- 생성기가 만든 기준표를 결론 목록이 그대로 읽는다(다시 만들지 않는다). 작업 생성 응답의 8.1 표 출처 글 = 지시문의 출처 글.
- 옛 경로(/api/generate-narrative)는 늘 판 3 이고, 사실 시트 덤프에서 판 4 칸이 빠진다(판 3 지시문이 한 글자도 안 바뀜).
- 판 4 예시 요청을 스키마로 읽어도 버려지는 키가 없다(pydantic 기본 extra=ignore 라 스키마에 없는 칸은 말없이 버려진다).

판 3 요청이 지금과 한 글자도 다르지 않은 것은 얼린 정답(tests/golden_frozen — test_edition3_frozen·test_frozen_goldens)이 본다.
"""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, configure_sqlite
from app.main import app
from app.narrative import (
    baselines,
    checks,
    fact_view,
    fallback,
    generator,
    grounding,
    jobs,
    legacy,
    prompt,
    required_items,
)
from app.narrative.schemas import NarrativeRequest
from tests.conftest import requested_fields

TESTS = Path(__file__).parent
CONTRACT = TESTS / "contract"
SPEC = json.loads((CONTRACT / "fact_sheet_contract_ed4.json").read_text(encoding="utf-8"))
ED4_ONLY = [e["path"] for e in SPEC["edition4_only_keys"]]


def _ed4_raw() -> dict:
    """판 4 임시 예시(bc_strong) — 문제 P2 가 판 4 에만 있는 종류 same_source 다."""
    return json.loads((CONTRACT / "fact_sheet_binary_ed4.json").read_text(encoding="utf-8"))


def _strip_keys(raw: dict) -> dict:
    """판 4 예시에서 판 4 키(명세 edition4_only_keys — 판 신호 포함)만 지운 사본."""
    out = copy.deepcopy(raw)
    for path in ED4_ONLY:
        for node, key in _containers(out, path):
            node.pop(key, None)
    return out


def _strip_edition4(raw: dict) -> dict:
    """판 4 키와 판 4 문제 종류(same_source)를 지운 사본 — 판 3 프론트가 보낼 법한 같은 요청."""
    out = _strip_keys(raw)
    out["fact_sheet"]["problems"] = [p for p in out["fact_sheet"]["problems"] if p["kind"] != "same_source"]
    return out


def _containers(obj, path: str) -> list[tuple[dict, str]]:
    parts = path.split(".")
    nodes = [obj]
    for part in parts[:-1]:
        many = part.endswith("[]")
        key = part[:-2] if many else part
        nxt = []
        for node in nodes:
            if not isinstance(node, dict) or node.get(key) is None:
                continue
            value = node[key]
            nxt.extend(v for v in value if isinstance(v, dict)) if many else nxt.append(value)
        nodes = nxt
    return [(n, parts[-1]) for n in nodes if isinstance(n, dict)]


# ── 요청 경계 ────────────────────────────────────────────────────────────────────


def test_report_edition_is_optional_and_defaults_to_3():
    raw = _strip_edition4(_ed4_raw())
    assert "report_edition" not in raw
    req = NarrativeRequest.model_validate(raw)
    assert req.report_edition == 3
    assert "report_edition" not in req.model_fields_set
    assert NarrativeRequest.model_validate({**raw, "report_edition": 4}).report_edition == 4
    assert NarrativeRequest.model_validate({**raw, "report_edition": 3}).report_edition == 3


@pytest.mark.parametrize("bad", [5, 2, "4", None, True, "판 4"])
def test_report_edition_outside_3_4_is_422(bad):
    raw = {**_ed4_raw(), "report_edition": bad}
    for path in ("/api/narrative/jobs", "/api/generate-narrative"):
        r = TestClient(app).post(path, json=raw)
        assert r.status_code == 422, (path, bad, r.text)
        assert any("report_edition" in e["loc"] for e in r.json()["detail"]), r.json()


def test_edition4_example_loses_no_key_through_the_schema():
    raw = _ed4_raw()
    dumped = NarrativeRequest.model_validate(raw).model_dump(mode="json")
    missing: list[str] = []

    def walk(a, b, path):
        if isinstance(a, dict):
            if not isinstance(b, dict):
                missing.append(path)
                return
            for k, v in a.items():
                if k not in b:
                    missing.append(f"{path}.{k}")
                else:
                    walk(v, b[k], f"{path}.{k}")
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                missing.append(path)
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif a != b:
            missing.append(f"{path} ({a!r} != {b!r})")

    walk(raw, dumped, "")
    assert missing == []
    # 판 4 키가 정말 예시에 다 있다(빈 갈래를 돌면 위 비교는 늘 통과한다).
    for path in ED4_ONLY:
        sites = _containers(raw, path)
        assert sites and all(key in node for node, key in sites), path


@pytest.mark.parametrize("field", ["ci_low_text", "best_recall_text", "risk_side", "evaluation_minus_training_text"])
def test_bad_edition4_text_is_dropped_softly_not_422(field):
    """판 4 표기 칸이 숫자 꼴이 아니면(또는 위험 쪽 묶음이 틀리면) 그 칸만 버리고 서술은 계속한다 — 판 3 의 value_text 와 같은 대접."""
    raw = _ed4_raw()
    fs = raw["fact_sheet"]
    if field == "ci_low_text":
        fs["metrics"][0]["ci_low_text"] = "약 0.95"
    elif field == "best_recall_text":
        fs["f1_threshold_gain"]["best_recall_text"] = "0.98 이상"
    elif field == "risk_side":
        fs["f1_threshold_gain"]["risk_side"]["side"] = "both"
    else:
        fs["facts"]["distribution_shift"][0]["evaluation_minus_training_text"] = "−0.2%p"
    req, trimmed = jobs.parse_request(raw)
    assert req.report_edition == 4
    if field == "evaluation_minus_training_text":
        assert trimmed == [jobs.FACTS_DROPPED]  # facts 안 칸은 facts 를 버린다(판 3 facts 형식 오류와 같다)
    else:
        assert trimmed == [jobs.DISPLAY_DROPPED]
    g = req.fact_sheet.f1_threshold_gain
    if field == "ci_low_text":
        assert req.fact_sheet.metrics[0].ci_low_text is None and req.fact_sheet.metrics[0].ci_high_text == "0.994"
    elif field == "best_recall_text":
        assert g.best_recall_text is None and g.best_precision_text == "0.984"
    elif field == "risk_side":
        assert g.risk_side is None and g.best_recall_text == "0.984"


# ── 안쪽 함수는 기본값 없는 edition ───────────────────────────────────────────────

INNER = [
    baselines.build_benchmark_refs,
    grounding.build_number_whitelist,
    required_items.select_items,
    prompt.build_context,
    fact_view.build_fact_view,
    checks.check_section,
    fallback.section_template,
    fallback._input,
    fallback.build_fallback_narrative,
]


@pytest.mark.parametrize("fn", INNER, ids=lambda f: f.__name__)
def test_inner_functions_take_edition_keyword_only_without_default(fn):
    param = inspect.signature(fn).parameters["edition"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


def test_inner_records_hold_edition_without_default():
    import dataclasses

    for cls in (required_items.Ctx, prompt.PromptContext):
        f = {x.name: x for x in dataclasses.fields(cls)}["edition"]
        assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING, cls
    assert inspect.signature(fallback._Input).parameters["edition"].default is inspect.Parameter.empty


def test_missing_edition_fails_at_once():
    req = NarrativeRequest.model_validate(_ed4_raw())
    with pytest.raises(TypeError):
        baselines.build_benchmark_refs("binary", req.fact_sheet.metrics)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        required_items.select_items(req.fact_sheet, req.client_inputs, "binary")  # type: ignore[call-arg]


# ── 배선(spy): 판 4 요청이면 하위 호출이 모두 edition=4 ──────────────────────────────


@pytest.fixture
def factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}", connect_args={"check_same_thread": False})
    configure_sqlite(engine)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, autocommit=False)
    engine.dispose()


# (모듈, 이름) — 작업 경로가 그 이름을 찾는 자리. 생성기는 기준표·허용 목록·결론 목록을 자기 이름공간에서, 지시문·검문·
# 규칙 문장은 모듈 속성으로, 사실 보기는 prompt 가 fact_view 모듈 속성으로 부른다. jobs 는 응답용 기준표를 따로 만든다.
SPIED = [
    (jobs, "build_benchmark_refs"),
    (generator, "build_benchmark_refs"),
    (generator, "build_number_whitelist"),
    (generator, "select_items"),
    (prompt, "build_context"),
    (fact_view, "build_fact_view"),
    (checks, "check_section"),
    (fallback, "section_template"),
    (fallback, "_input"),
    (required_items, "build_benchmark_refs"),  # 결론 목록이 기준표를 다시 만들면 여기 잡힌다(불려선 안 된다)
]


@pytest.fixture
def spies(monkeypatch):
    calls: dict[str, list[dict]] = {f"{m.__name__.rsplit('.', 1)[-1]}.{n}": [] for m, n in SPIED}
    for module, name in SPIED:
        original = getattr(module, name)
        key = f"{module.__name__.rsplit('.', 1)[-1]}.{name}"

        def wrapper(*args, __orig=original, __key=key, **kwargs):
            calls[__key].append(kwargs)
            return __orig(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)
    return calls


async def _run(raw: dict, factory, client) -> tuple[dict, list]:
    tasks = BackgroundTasks()
    created = await jobs.create_job(raw, client=client, session_factory=factory, background_tasks=tasks)
    for task in tasks.tasks:
        await task.func(*task.args, **task.kwargs)
    return created.model_dump(mode="json"), client.beta.messages.create.call_args_list if client is not None else []


def _assert_all(calls: dict, edition: int, *, expect_called: set[str]):
    for key, recorded in calls.items():
        if key == "required_items.build_benchmark_refs":
            assert recorded == [], "결론 목록이 기준표를 다시 만들었다 — 생성기가 만든 것을 넘겨야 한다"
            continue
        if key in expect_called:
            assert recorded, key  # 빈 목록을 돌면 아래 단정이 늘 통과한다
        assert [c.get("edition") for c in recorded] == [edition] * len(recorded), key


@pytest.mark.parametrize("edition", [4, 3])
async def test_every_inner_call_gets_the_request_edition_with_ai(edition, factory, spies, make_section_client):
    raw = _ed4_raw() if edition == 4 else _strip_edition4(_ed4_raw())
    client = make_section_client({})
    created, sent = await _run(raw, factory, client)
    assert len(sent) >= 3  # 7·8·9장(7.1 은 D24 선언 숫자가 없어 한 번 더 부를 수 있다)
    _assert_all(spies, edition, expect_called={
        "jobs.build_benchmark_refs", "generator.build_benchmark_refs", "generator.build_number_whitelist",
        "generator.select_items", "prompt.build_context", "fact_view.build_fact_view", "checks.check_section",
        # 7.3 은 AI 가 빈 글을 내면 규칙 칸(고정 문구 15)으로 — 규칙 문장 함수가 불린다.
        "fallback.section_template", "fallback._input",
    })
    assert len(spies["checks.check_section"]) >= 6
    # 결론 목록은 생성기가 만든 기준표·파생값을 그대로 받는다(같은 객체).
    [sel] = spies["generator.select_items"]
    [ctx_call] = spies["prompt.build_context"]
    assert sel["benchmark_refs"] is not None and sel["derived"] is not None
    assert sel["edition"] == ctx_call["edition"] == edition


@pytest.mark.parametrize("edition", [4, 3])
async def test_every_inner_call_gets_the_request_edition_without_ai(edition, factory, spies):
    raw = _ed4_raw() if edition == 4 else _strip_edition4(_ed4_raw())
    await _run(raw, factory, None)
    _assert_all(spies, edition, expect_called={
        "jobs.build_benchmark_refs", "generator.build_benchmark_refs", "generator.build_number_whitelist",
        "generator.select_items", "prompt.build_context", "fact_view.build_fact_view",
        "fallback.section_template", "fallback._input",
    })
    assert len(spies["fallback.section_template"]) == 7  # 키 없음 — 일곱 칸 모두 규칙 문장


def _source_note_in(user_message: str) -> str:
    sheet = json.loads(user_message.split("[사실 시트]\n", 1)[1].split("\n", 1)[0])
    return sheet["벤치마크(내부 참조 기준)"]["기준의 출처·성격"]


@pytest.mark.parametrize("edition", [4, 3])
async def test_created_job_source_note_equals_prompt_source_note(edition, factory, make_section_client):
    raw = _ed4_raw() if edition == 4 else _strip_edition4(_ed4_raw())
    client = make_section_client({})
    created, sent = await _run(raw, factory, client)
    refs = created["benchmark_refs"]
    assert refs and len({r["source_note"] for r in refs}) == 1
    chapter8 = [c.kwargs for c in sent if "section_8_1" in requested_fields(c.kwargs)]
    assert len(chapter8) == 1
    assert _source_note_in(chapter8[0]["messages"][0]["content"]) == refs[0]["source_note"]
    # 같은 판으로 기준표를 직접 만든 것과도 같다(응답 표 = 생성기 기준표).
    req = NarrativeRequest.model_validate(raw)
    direct = baselines.build_benchmark_refs("binary", req.fact_sheet.metrics, edition=edition)
    assert [r["source_note"] for r in refs] == [r["source_note"] for r in direct]


# ── 옛 경로는 늘 판 3 ────────────────────────────────────────────────────────────


def test_legacy_edition4_fields_match_the_contract():
    """옛 경로 덤프에서 빼는 칸 = 명세의 판 4 키 가운데 사실 시트 안의 것(빠뜨리면 판 3 지시문에 None 키가 는다)."""

    def flatten(node, prefix: str) -> set[str]:
        if isinstance(node, set):
            return {f"{prefix}.{leaf}" for leaf in node}
        out: set[str] = set()
        for key, value in node.items():
            if key == "__all__":
                out |= flatten(value, prefix + "[]")
            else:
                out |= flatten(value, f"{prefix}.{key}")
        return out

    want = {p for p in ED4_ONLY if p != "report_edition"}
    assert want and all(p.startswith("fact_sheet.") for p in want)
    assert flatten(legacy.EDITION4_ONLY_SHEET_FIELDS, "fact_sheet") == want


def test_legacy_prompts_ignore_edition4_request():
    """판 4 요청이 옛 경로로 와도 판 4 칸·판 신호를 지운 같은 요청과 지시문·허용 목록·기준표가 같다."""
    ed4 = NarrativeRequest.model_validate(_ed4_raw())
    ed3 = NarrativeRequest.model_validate(_strip_keys(_ed4_raw()))
    assert ed4.report_edition == 4 and ed3.report_edition == 3
    got = []
    for req in (ed4, ed3):
        derived, refs, whitelist = legacy._prepare_facts(req.fact_sheet, req.task_type.value, req.client_inputs)
        system, user, dropped = legacy.build_prompts_within_limit(req, req.fact_sheet, derived, refs)
        got.append((system, user, dropped, refs, sorted(map(str, whitelist))))
    for a, b in zip(*got):
        assert a == b
    # 덤프에 판 4 칸 이름이 하나도 없다.
    user = got[0][1]
    for key in ("ci_low_text", "best_threshold_text", "risk_side", "evaluation_minus_training_text", "report_edition"):
        assert key not in user, key


def test_legacy_route_answers_edition4_request_as_edition3():
    ed4 = TestClient(app).post("/api/generate-narrative", json=_ed4_raw())
    ed3 = TestClient(app).post("/api/generate-narrative", json=_strip_keys(_ed4_raw()))
    assert ed4.status_code == ed3.status_code == 200
    a, b = ed4.json(), ed3.json()
    assert a["benchmark_refs"] and a == b
