"""tests/test_judge_sources.py — 심판 칸 목록(app/judge/sources.py field_catalog)."""
import json
from pathlib import Path

import pytest

from app.judge.sources import ALIASES, catalog_text, field_catalog
from app.narrative.baselines import build_benchmark_refs
from app.narrative.schemas import NarrativeRequest

DEV = Path(__file__).parent / "data" / "judge_dev"
NAMES = ["dev_binary.request.json", "dev_multiclass.request.json", "dev_multilabel.request.json"]


def _catalog(name):
    req = NarrativeRequest.model_validate(json.loads((DEV / name).read_text(encoding="utf-8")))
    refs = build_benchmark_refs(req.task_type.value, req.fact_sheet.metrics, edition=4)
    return req, field_catalog(req.fact_sheet.model_dump(), refs)


@pytest.mark.parametrize("name", NAMES)
def test_every_material_number_field_resolves(name):
    req, cat = _catalog(name)
    fields = [n["field"] for s in req.judge_material["segments"] for n in s["numbers"] if n["field"]]
    assert fields and [f for f in fields if f not in cat] == []


def test_names_and_values():
    req, cat = _catalog("dev_multiclass.request.json")
    fs = req.fact_sheet
    m1 = next(m for m in fs.metrics if m.metric_id == "M1")
    assert cat["metric.M1.value"].value == m1.value and cat["metric.M1.value"].description.startswith("Accuracy(M1)")
    assert cat["per_class.cat.f1"].value == next(p.f1 for p in fs.per_class if p.label == "cat")
    assert cat["confusion_counts.dog.fp"].value == next(c.fp for c in fs.confusion_counts if c.label == "dog")
    assert cat["confusion.dog>cat"].value == fs.confusion.matrix[2][1]
    assert cat["facts.top_confusion_pairs.dog>cat.count"].value == 28
    assert cat["facts.prediction_bias.bird.difference"].value == -10
    assert not any(k.startswith(("problems", "facts.fixed_sentences", "score")) for k in cat)
    for alias, target in ALIASES.items():
        assert cat[alias].value == cat[target].value
    assert any(k.startswith("benchmark.") for k in cat)


def test_catalog_text_has_names_not_values():
    _, cat = _catalog("dev_binary.request.json")
    text = catalog_text(cat)
    assert "- metric.M1.value — Accuracy(M1) value" in text
    assert str(cat["metric.M1.value"].value) not in text
