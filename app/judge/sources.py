"""app/judge/sources.py — 심판이 대조하는 데이터 칸 목록(필드 카탈로그, 계획 11-2 의 바탕).

사실 시트(fact_sheet)의 숫자 잎을 "칸 이름 → (설명, 값)" 으로 편다. 칸 이름은 두 곳이 같이 쓴다.
- 공용 전처리(주장 추출): 숫자·비교 주장마다 가리키는 칸 이름을 이 목록에서 고른다.
- 사실 일치 값 조회 도구·일관성 값 묶기: 이 이름으로 값을 찾고 묶는다. 프론트 재료의 숫자 칸 이름(계약 judge_material_ed4.json
  number_fields)과 같은 이름을 쓴다(아래 ALIASES).

이름 규칙: 목록 줄은 그 줄의 이름 값(metric_id·label·id)으로 연다 — metrics[M1].value → metric.M1.value,
per_class[cat].f1 → per_class.cat.f1, confusion_counts[cat].tp → confusion_counts.cat.tp. 혼동행렬은 confusion.<실제>><예측>.
참조 범위는 작업이 계산한 benchmark_refs 로 benchmark.<지표>.<model_value|ref_low|ref_high|target>.

조회 순서(설계 6.1 결정: 사실 시트 → 성적서 데이터 → 고객 입력)와 생성기가 AI 에 준 값(파생값 등)은 J3 에서 붙인다.

상호작용
- 의존(import): 없음(사실 시트는 dict 로 받는다 — model_dump)
- 사용처: app.judge.extract, (J3) faithfulness·consistency
"""
from __future__ import annotations

from dataclasses import dataclass

# 목록 줄을 여는 이름 칸(앞에서부터 찾는다).
_ROW_KEYS = ("metric_id", "label", "id")
# 펴지 않는 칸 — 숫자가 아니거나(문제 근거는 글), 같은 값을 다른 이름으로 또 싣는 묶음.
_SKIP = {"problems", "facts.fixed_sentences", "facts.contract", "facts.omitted", "facts.selected_metrics", "score"}
# 프론트 재료 숫자 칸 이름 → 사실 시트 칸 이름(같은 값). 계약 judge_material_ed4.json number_fields 와 맞춘다.
ALIASES = {
    "facts.weakest_class.gap": "facts.weakest_class.gap_from_macro",
    "verdict.judged_count": "facts.verdict_detail.judged_count",
    "verdict.passed_count": "facts.verdict_detail.passed_count",
}


@dataclass(frozen=True)
class FieldEntry:
    name: str
    description: str
    value: float | int


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _row_name(row: dict) -> str | None:
    for k in _ROW_KEYS:
        if k in row:
            return "*" if row[k] is None else str(row[k])
    return None


def _walk(value, path: str, out: dict[str, float | int]) -> None:
    if path in _SKIP:
        return
    if _is_number(value):
        out[path] = value
    elif isinstance(value, dict):
        for k, v in value.items():
            _walk(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(value, list):
        for i, row in enumerate(value):
            if isinstance(row, dict):
                if path == "facts.top_confusion_pairs":
                    name = f"{row.get('actual')}>{row.get('predicted')}"
                elif path.endswith("prediction_bias.rows"):
                    _walk(row, f"{path[: -len('.rows')]}.{_row_name(row) or i}", out)
                    continue
                else:
                    name = _row_name(row) or str(i)
                _walk({k: v for k, v in row.items() if k not in _ROW_KEYS}, f"{path}.{name}", out)


def _rename(path: str) -> str:
    return "metric." + path[len("metrics."):] if path.startswith("metrics.") else path


def field_catalog(fact_sheet: dict, benchmark_refs: list[dict] | None = None) -> dict[str, FieldEntry]:
    """칸 이름 → FieldEntry. 사실 시트는 model_dump() dict."""
    raw: dict[str, float | int] = {}
    _walk({k: v for k, v in fact_sheet.items() if k != "confusion"}, "", raw)
    conf = fact_sheet.get("confusion") or {}
    labels, matrix = conf.get("labels") or [], conf.get("matrix") or []
    for i, row in enumerate(matrix):
        for j, n in enumerate(row):
            if i < len(labels) and j < len(labels):
                raw[f"confusion.{labels[i]}>{labels[j]}"] = n
    names = {m["metric_id"]: m.get("display_name", m["metric_id"]) for m in fact_sheet.get("metrics") or []}
    out: dict[str, FieldEntry] = {}
    for path, v in raw.items():
        name = _rename(path)
        desc = name
        if name.startswith("metric."):
            mid = name.split(".")[1]
            desc = f"{names.get(mid, mid)}({mid}) {name.split('.', 2)[2]}"
        out[name] = FieldEntry(name, desc, v)
    for r in benchmark_refs or []:
        for key, field in (("model_value", "model_value"), ("ref_low", "ref_low"), ("ref_high", "ref_high"), ("target_value", "target")):
            if _is_number(r.get(key)):
                name = f"benchmark.{r['metric']}.{field}"
                out[name] = FieldEntry(name, f"{r['metric']} 참조 범위 {field}", r[key])
    for alias, target in ALIASES.items():
        if target in out:
            out[alias] = FieldEntry(alias, out[target].description, out[target].value)
    return out


def catalog_text(catalog: dict[str, FieldEntry]) -> str:
    """추출 지시문에 실을 칸 목록(값은 싣지 않는다 — 추출은 판단하지 않고 가리키기만 한다)."""
    return "\n".join(f"- {e.name}" if e.description == e.name else f"- {e.name} — {e.description}" for e in catalog.values())
