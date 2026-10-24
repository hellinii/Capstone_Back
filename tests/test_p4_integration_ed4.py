"""tests/test_p4_integration_ed4.py — 판 4 라운드 S3 ⑧ 통합: 프론트가 뜬 판 4 요청을 실제 앱(TestClient)에 넣어 끝까지.

입력은 tests/contract/fact_sheet_{binary,multiclass,multilabel}_ed4.json — 프론트가 앱이 실제로 보내는 판 4 요청 본문으로 뜬 예시
(RPT-2026-0001·0003·0005 입력, 프론트 src/test/edition4/contractExamplesEd4.test.tsx). 판 3 쪽은 같은 입력의 판 3 발급본 요청
(tests/data/edition3_issued/requests — 판 신호 없음)이다.

- 키 없음(폴백 — LLM 호출 없음): `POST /api/narrative/jobs` → BackgroundTasks 가 응답 전에 작업을 끝까지 돌리고 `GET` 이 done.
  판 4 는 판 4 규칙 문장(판 4 표기·판 4 절 번호·통과율 없음)·판 4 출처 글·B23·B24, 판 3 은 S0 얼린 정답(tests/golden_frozen/
  edition3/<입력>.templates.json)과 같다.
- 지시문: app.state 에 받아 적기만 하는 가짜 클라이언트(거절 응답 — 다시 부르지 않고 규칙 문장으로 끝남)를 끼워, 판 4 는 판 4
  system·판 4 자리·판 3 절 번호 없음, 판 3 은 S0 얼린 지시문 정답(edition3/<입력>.job_prompts.json)과 글자 그대로 같은지 본다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, configure_sqlite
from app.main import app
from app.narrative import jobs, prompt
from app.narrative.baselines import SOURCE_NOTE_ED4
from app.narrative.sections import SECTION_IDS, SECTIONS, SECTIONS_BY_ID, place, title
from app.narrative.wording_ed4 import ed4_distinct
from tests.p4_example_ed4 import EXAMPLE_TEMPLATES

TESTS = Path(__file__).parent
CONTRACT = TESTS / "contract"
FROZEN = TESTS / "golden_frozen" / "edition3"
ISSUED = TESTS / "data" / "edition3_issued" / "requests"

# 판 4 예시 → 같은 입력의 판 3 발급본 요청 이름.
PAIRS = {"binary": "bc_strong", "multiclass": "digits_strong", "multilabel": "yeast_strong"}
FALLBACK_SOURCES = {"7.1": "fallback", "7.2": "fallback", "7.3": "fixed", "8.1": "fallback", "8.2": "fallback",
                    "8.3": "fallback", "9": "fallback"}
_FIELD_TO_SECTION = {f: s.id for s in SECTIONS for f in s.ai_fields}


def _ed4(task: str) -> dict:
    return json.loads((CONTRACT / f"fact_sheet_{task}_ed4.json").read_text(encoding="utf-8"))


def _ed3(task: str) -> dict:
    return json.loads((ISSUED / f"{PAIRS[task]}.request.json").read_text(encoding="utf-8"))


def _frozen(task: str, kind: str) -> dict:
    return json.loads((FROZEN / f"{PAIRS[task]}.{kind}.json").read_text(encoding="utf-8"))


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


class _Recorder:
    """작업 경로가 보내는 요청을 받아 적는 가짜 Anthropic 클라이언트. 응답은 거절 — 그 장을 규칙 문장으로 끝낸다(다시 부르지 않음)."""

    def __init__(self):
        self.sent: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def with_options(self, **_kwargs):
        return self

    async def _create(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(
            model="recording", stop_reason="refusal", stop_details=None, content=[],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0, iterations=None),
        )

    def chapters(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for kw in self.sent:
            sids = sorted({_FIELD_TO_SECTION[f] for f in kw["output_config"]["format"]["schema"]["properties"]},
                          key=SECTION_IDS.index)
            chapter = SECTIONS_BY_ID[sids[0]].chapter
            assert chapter not in out, chapter  # 장마다 한 번(거절이면 다시 부르지 않는다)
            out[chapter] = {"sections": sids, "system": kw["system"], "user": kw["messages"][0]["content"]}
        return out


def _run(api: TestClient, raw: dict) -> dict:
    created = api.post("/api/narrative/jobs", json=raw)
    assert created.status_code == 200, created.text
    view = api.get(f"/api/narrative/jobs/{created.json()['job_id']}").json()
    assert (view["status"], view["stage"], view["failure"]) == ("done", "done", None), view
    assert view["benchmark_refs"] == created.json()["benchmark_refs"]
    return view


def _parts(result: dict) -> dict[str, str]:
    out = {f"interpretation.{k}": v for k, v in result["interpretation"].items()}
    out |= {f"conclusion.{k}": result["conclusion"][k] for k in ("benchmark", "narrative", "risks")}
    out |= {f"recommendation_narrative.{k}": v for k, v in result["recommendation_narrative"].items()}
    return out


# 판 3 칸 이름의 절 번호('8.3절', '7.1·8.3절', '9절') — 판 4 글·지시문에 나오면 안 된다. 판 4 절은 1~6·부록.
_ED3_SECTION_REF = re.compile(r"(?<![\d.])(?:[789]\.[123]|9)(?:·(?:[789]\.[123]|9))*절")
_DECIMAL = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])")


def _extended_texts(raw: dict) -> set[str]:
    """요청이 판 4 규칙으로 늘려 찍은 글자(셋째 자리보다 긴 것) — 경계·임계값 표기, 최약 클래스 F1(facts.weakest_class.f1_text),
    그리고 셋째 자리로 최약 클래스와 같아 보이는 다른 클래스 F1 을 늘린 표기(D3 — 공통 대조 표 distinct 규칙, 다중 예시의 5 = 0.9725)."""
    fs = raw["fact_sheet"]
    out = {m[k] for m in fs["metrics"] for k in ("ci_low_text", "ci_high_text") if m.get(k)}
    if fs.get("f1_threshold_gain") and fs["f1_threshold_gain"].get("best_threshold_text"):
        out.add(fs["f1_threshold_gain"]["best_threshold_text"])
    weakest = (fs.get("facts") or {}).get("weakest_class")
    if weakest:
        out.add(weakest["f1_text"])
        out |= {ed4_distinct(p["f1"], [weakest["f1"]]) for p in fs["per_class"] if p["label"] != weakest["label"]}
    return {t for t in out if len(t.split(".")[-1]) > 3}


# ── 판 4: 키 없음 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("task", list(PAIRS))
def test_edition4_job_runs_to_the_end_with_edition4_rule_sentences(api, task):
    raw = _ed4(task)
    view = _run(api, raw)
    result = view["result"]
    assert result["section_sources"] == FALLBACK_SOURCES  # 세 예시 모두 문제가 있다 — 9칸도 규칙 문장
    assert result["section_models"] == dict.fromkeys(FALLBACK_SOURCES)
    assert {r["source_note"] for r in view["benchmark_refs"]} == {SOURCE_NOTE_ED4}
    texts = _parts(result)
    assert "내부 참조 기준치 v1 (공개 벤치마크 평균이 아닌 자체 설정 기준이며" in texts["conclusion.benchmark"]
    extended = _extended_texts(raw)
    checked = 0
    for key, text in texts.items():
        assert not _ED3_SECTION_REF.search(text), (key, text)
        assert "통과율" not in text, key
        for m in _DECIMAL.finditer(text):
            num, frac = m.group(0), m.group(2)
            checked += 1
            if text[m.end():m.end() + 1] == "%":
                assert len(frac) == 1, (key, num)  # 백분율·%p 는 소수 첫째 자리
            elif len(frac) > 3:
                assert num in extended, (key, num)  # 넷째 자리 이상은 판 4 규칙이 늘린 경계·임계값 표기뿐
            elif len(frac) < 3:
                after, before = text[m.end():m.end() + 1], text[max(0, m.start() - 1):m.start()]
                # 판정 기준값(0.1·1.5)·의뢰자 결정 임계값(0.5) 글자 그대로, 참조 범위 끝(둘째 자리 'a~b')
                assert num in ("0.1", "1.5", "0.5") or "~" in (after, before), (key, num)
    assert checked > 10
    assert "6절 권고표" in texts["recommendation_narrative.data_quality"]
    recs = {r["problem_id"]: r for r in result["recommendations"]}
    same_source = [p["id"] for p in raw["fact_sheet"]["problems"] if p["kind"] == "same_source"]
    assert len(same_source) == 1 and recs[same_source[0]]["category"] == "다른 출처 데이터로 재평가"  # B23


def test_edition4_binary_job_result_is_the_example_rule_sentences(api):
    result = _run(api, _ed4("binary"))["result"]
    assert _parts(result) == EXAMPLE_TEMPLATES
    p1 = next(r for r in result["recommendations"] if r["problem_id"] == "P1")
    assert p1["action"].startswith("판정 여유가 작은 채 합격한 핵심 지표를")  # B24


def test_edition4_multilabel_failed_metric_is_written_in_edition4_notation(api):
    """목표 미달 목록(8.3 첫 문장)의 목표는 셋째 자리 — 8.2 판정 문장과 같은 글자(판 3 은 의뢰자 글자 0.45)."""
    texts = _parts(_run(api, _ed4("multilabel"))["result"])
    assert texts["conclusion.risks"].startswith(
        "목표에 미달한 지표가 있어 해당 영역의 성능 저하 위험이 존재한다(목표 미달 지표: F1 Score(0.399, 목표 0.450)).",
    )
    assert "(목표 미달 지표: F1 Score(0.399, 목표 0.450))" in texts["conclusion.narrative"]
    # 같은 입력의 판 3 은 그대로(의뢰자 글자).
    ed3 = _parts(_run(api, _ed3("multilabel"))["result"])
    assert "(목표 미달 지표: F1 Score(0.399, 목표 0.45))" in ed3["conclusion.risks"]


# ── 판 4: 지시문 ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("task", list(PAIRS))
def test_edition4_job_sends_edition4_prompts(api, task):
    raw = _ed4(task)
    rec = _Recorder()
    app.state.anthropic_client = rec  # conftest 가 테스트마다 None 으로 되돌린다
    view = _run(api, raw)
    chapters = rec.chapters()
    assert sorted(chapters) == ["7", "8", "9"]
    extended = _extended_texts(raw)
    for chapter, sent in chapters.items():
        assert sent["system"] == prompt.SYSTEM_PROMPT_ED4, chapter
        user = sent["user"]
        for sid in sent["sections"]:
            assert f"■ {place(sid, edition=4)} {title(sid, edition=4)}" in user, (chapter, sid)
        assert not _ED3_SECTION_REF.search(user), (chapter, _ED3_SECTION_REF.findall(user))
        # 사실 보기·결론에 넷째 자리 이상 숫자는 판 4 규칙이 늘린 표기뿐(엔진 넷째 자리·판 3 표기가 새지 않음).
        long = {m.group(0) for m in _DECIMAL.finditer(user) if len(m.group(2)) > 3}
        assert long <= extended, (chapter, sorted(long - extended))
    # 거절이라 규칙 문장으로 끝났다 — 키 없는 결과와 칸 글이 같다.
    plain = _run(api, raw)  # 같은 앱, 가짜 클라이언트 그대로라 다시 거절 → 같은 글
    assert _parts(view["result"]) == _parts(plain["result"])
    app.state.anthropic_client = None
    assert _parts(_run(api, raw)["result"]) == _parts(view["result"])


# ── 판 3: S0 얼린 정답과 같다 ────────────────────────────────────────────────────


@pytest.mark.parametrize("task", list(PAIRS))
def test_edition3_job_through_the_app_equals_frozen_templates(api, task):
    raw = _ed3(task)
    assert "report_edition" not in raw
    result = _run(api, raw)["result"]
    golden = _frozen(task, "templates")
    for key in ("interpretation", "conclusion", "recommendation_narrative", "recommendations"):
        assert result[key] == golden["response"][key], key
    assert result["section_sources"] == golden["section_sources"]
    assert result["section_models"] == golden["section_models"]


@pytest.mark.parametrize("task", list(PAIRS))
def test_edition3_job_through_the_app_sends_frozen_prompts(api, task):
    rec = _Recorder()
    app.state.anthropic_client = rec
    _run(api, _ed3(task))
    golden = _frozen(task, "job_prompts")["chapters"]
    chapters = rec.chapters()
    assert sorted(chapters) == sorted(golden) == ["7", "8", "9"]
    for chapter, sent in chapters.items():
        assert sent["sections"] == golden[chapter]["sections"], chapter
        assert sent["system"] == golden[chapter]["system"] == prompt.SYSTEM_PROMPT, chapter
        assert sent["user"] == golden[chapter]["user"], chapter


def test_edition_patterns_catch_what_they_must():
    """자기 검문 — 판 3 절 번호 꼴은 잡고 판 4 절 번호는 놓아준다."""
    for bad in ("8.3절", "7.1·8.3절", "9절 권고표", "(9.2절)"):
        assert _ED3_SECTION_REF.search(bad), bad
    for ok in ("5.5절", "5.2·5.5절", "6절 권고표", "6.1절", "부록 1.2", "0.983절"):
        assert not _ED3_SECTION_REF.search(ok), ok
    assert [m.group(0) for m in _DECIMAL.finditer("0.9497~0.994, 49,56, 1.5")] == ["0.9497", "0.994", "1.5"]


# ── 다중·멀티 예시가 드러낸 어긋남(S3 ⑧ 에서 고침) — 순수 함수로도 ──────────────────────


def test_joined_section_names_get_edition4_places_one_by_one():
    """프론트 최약 클래스 문제의 '드러난 절'은 '7.1·8.3' — 판 4 는 하나씩 바꿔 잇는다(판 3 은 그대로)."""
    assert place("7.1·8.3", edition=4) == "5.2·5.5"
    assert place("7.1·8.3", edition=3) == "7.1·8.3"
    from app.narrative import fact_view_ed4
    from app.narrative.schemas import NarrativeRequest

    fs = NarrativeRequest.model_validate(_ed4("multilabel")).fact_sheet
    weakest = [p for p in fact_view_ed4._problems(fs) if p["번호"] == "P3"]
    assert [(p["문제"], p["드러난 절"]) for p in weakest] == [("최약 클래스 (Class9)", "5.2·5.5절")]


def test_imbalance_evidence_uses_the_edition4_display_of_the_fact_sheet():
    """근거에 실린 판 3 display(52.9231·0.7016·0.1333)가 아니라 사실 시트의 판 4 display 글자로."""
    from app.narrative import fact_view_ed4
    from app.narrative.schemas import NarrativeRequest

    raw = _ed4("multilabel")
    [problem] = [p for p in raw["fact_sheet"]["problems"] if p["kind"] == "imbalance_effect"]
    assert problem["evidence"]["display"]["minority_f1"] == "0.1333"  # 프론트가 싣는 근거는 판 3 표기(계약 not_decided)
    fs = NarrativeRequest.model_validate(raw).fact_sheet
    [p2] = [p for p in fact_view_ed4._problems(fs) if p["번호"] == problem["id"]]
    ev = p2["근거"]
    assert (ev["불균형 비율"], ev["다수 − 소수 F1 차이"], ev["minority_f1"], ev["majority_f1"]) == ("52.923", "0.702", "0.133", "0.835")
