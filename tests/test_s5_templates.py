"""test_s5_templates.py — 규칙 문장을 소절 7칸 함수로 쪼갠 것(서술 라운드 S5 첫 일).

검문 사다리는 걸린 칸만 그 칸의 템플릿으로 바꾼다. 그래서 칸 함수가 등록표(app.narrative.sections)의 칸·응답 필드와
정확히 맞아야 하고, 일곱 칸을 모은 것이 종전 build_fallback_narrative 와 글자까지 같아야 한다(문장은 S6 에서 고친다).
"""
import json
from pathlib import Path

from app.narrative import fallback, sections
from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.schemas import NarrativeMeta, NarrativeRequest

CONTRACT = Path(__file__).parent / "contract"


def _requests():
    for name in ("binary", "multiclass", "multilabel", "legacy_main"):
        yield name, NarrativeRequest.model_validate(json.loads((CONTRACT / f"fact_sheet_{name}.json").read_text()))


def test_templates_cover_exactly_the_seven_sections():
    assert tuple(fallback.SECTION_TEMPLATES) == sections.SECTION_IDS


def test_each_template_writes_only_its_own_fields():
    for name, req in _requests():
        fs = req.fact_sheet
        refs = build_benchmark_refs(req.task_type.value, fs.metrics)
        derived = compute_derived(fs, task_type=req.task_type.value)
        for section in sections.SECTIONS:
            parts = fallback.section_template(section.id, fs, refs, derived)
            assert set(parts) == set(section.response_paths), (name, section.id)


def test_seven_templates_rebuild_the_old_fallback_exactly():
    """쪼개기 전 응답(골든 파일이 고정)과 같다 — 여기서는 칸별 조립 = build_fallback_narrative 를 본다."""
    for name, req in _requests():
        fs = req.fact_sheet
        refs = build_benchmark_refs(req.task_type.value, fs.metrics)
        derived = compute_derived(fs, task_type=req.task_type.value)
        whole = fallback.build_fallback_narrative(fs, refs, derived, reason="no_key")
        parts: dict = {}
        for sid in sections.SECTION_IDS:
            parts.update(fallback.section_template(sid, fs, refs, derived))
        rebuilt = fallback.assemble_response(parts, fs, whole.meta)
        assert rebuilt == whole, name
        assert whole.interpretation.additional_observations == ""  # 7.3 은 규칙 문장이 없다(고정 문구 15)
        assert whole.conclusion.verdict == fs.verdict


def test_assemble_response_forces_the_fact_sheet_verdict():
    _, req = next(_requests())
    resp = fallback.assemble_response(
        {"conclusion.narrative": "글"}, req.fact_sheet, NarrativeMeta(source="llm"),
    )
    assert resp.conclusion.verdict == req.fact_sheet.verdict
    assert resp.conclusion.narrative == "글"
    assert resp.recommendations == []
