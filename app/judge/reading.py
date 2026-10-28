"""app/judge/reading.py — 심판이 읽는 범위를 조각 목록으로 모으는 한 곳(설계 4절 "읽는 범위", 계획 10-2).

조각 = 성적서에 찍힌 글 한 덩어리. 출처는 넷이다.
- ai: AI 가 쓴 칸 글(서술 생성 결과에서 칸 출처 llm)
- template: 규칙 문장으로 바뀐 칸 글(칸 출처 fallback)
- fixed: 고정 문구(프론트 재료, 칸 출처 fixed 인 칸 글)
- table: 표(프론트 재료, 4.6 참조 범위 표는 여기서 만든다)

칸 이름은 판 3 이름(7.1·7.2·7.3·8.1·8.2·8.3·9)과 8(8절 머리 = 판 4 의 4.2)이고, place 는 판 4 에서 찍힌 자리다(정5).
6절 권고표의 권고·기대 효과는 서술 결과의 권고(AI·규칙 문장)라 9칸 출처를 따르고, 우선순위·대응 문제 칸은 프론트 재료에서 온다.

모듈마다 받는 범위는 `for_module` 한 곳에서 거른다(설계 4절): 사실 일치·표기·전처리는 ai·template 만, 필수 내용·일관성은 전부.

상호작용
- 의존(import): app.judge.material, app.narrative.wording_ed4(판 4 숫자 표기 — 판정 규칙이 아니라 표기 함수)
- 사용처: app.judge 의 전처리·4모듈, 정답지 표시 도구
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.judge.material import JudgeMaterial
from app.narrative.wording_ed4 import ed4_target, same_value_text_ed4

SLOT_ORDER = ("7.1", "7.2", "7.3", "8", "8.1", "8.2", "8.3", "9")
# 판 3 칸 → 판 4 자리(서술 설계 D47, 계약 tests/contract/narrative_slots_ed4.json 과 같은지 시험이 본다).
ED4_PLACE = {"8.2": "5.1", "7.1": "5.2", "7.2": "5.3", "8.1": "5.4", "8.3": "5.5", "7.3": "5.6"}
NARRATIVE_PATHS = {
    "7.1": ("interpretation", "confusion_analysis"),
    "7.2": ("interpretation", "distribution_analysis"),
    "7.3": ("interpretation", "additional_observations"),
    "8.1": ("conclusion", "benchmark"),
    "8.2": ("conclusion", "narrative"),
    "8.3": ("conclusion", "risks"),
}
RECOMMEND_PATHS = (("6.1", "data_quality"), ("6.2", "model_ops"))
SOURCE_OF = {"llm": "ai", "fallback": "template", "fixed": "fixed"}
SOURCES = ("ai", "template", "fixed", "table")
MODULE_SOURCES = {
    "extract": ("ai", "template"),
    "faithfulness": ("ai", "template"),
    "readability": ("ai", "template"),
    "coverage": SOURCES,
    "consistency": SOURCES,
}

# 4.6 참조 범위 표(프론트 render/resultsDetail.ts benchmarkView 와 같은 글자 — tests/contract/ref_table_ed4_cases.json 이 맞댄다).
REF_COLUMNS = ("지표", "모델 값", "참조 범위", "위치", "목표값", "목표값 수준")
POSITION_LABEL = {"below": "범위보다 낮음", "within": "범위 안", "above": "범위보다 높음"}
LOWER_IS_BETTER = "(낮을수록 좋음)"
DASH = "—"
NOTE_PREFIX = "※ "


@dataclass(frozen=True)
class Piece:
    id: str
    slot: str
    place: str
    source: str
    text: str
    numbers: tuple[tuple[str, str | None], ...] = field(default=())


def _table_text(title_lines: list[str], header: list[str], rows: list[list[str]]) -> str:
    lines = [t for t in title_lines if t]
    lines.append(" | ".join(header))
    lines.extend(" | ".join(r) for r in rows)
    return "\n".join(lines)


def _position_of_quality(quality: str, lower_is_better: bool) -> str | None:
    """프론트 judgedNumber.positionOfQuality 와 같다."""
    if quality == "within":
        return "within"
    if quality not in ("better", "worse"):
        return None
    return "below" if (quality == "better") == lower_is_better else "above"


def _model_value_text(ref: dict, metrics) -> str:
    """프론트 benchmarkModelValueTextEd4 — 같은 이름의 산출된 지표의 판 4 표기(판정 대상이면 value_text, 아니면 같은 수 같은 글자)."""
    m = next((m for m in metrics if m.display_name == ref["metric"] and m.value is not None and m.status != "unavailable"), None)
    if m is None:
        return DASH
    # 판정 대상이면 same_value_text_ed4 가 그 지표 자신의 value_text 를 찾는다(프론트 metricValueTextEd4 ?? sameValueTextEd4 와 같다).
    return same_value_text_ed4(metrics, m.value)


def ref_table_rows(benchmark_refs: list[dict], metrics) -> list[list[str]]:
    rows = []
    for r in benchmark_refs:
        lower = r.get("direction") == "lower"
        position = _position_of_quality(r.get("quality", ""), lower)
        target = r.get("target_value")
        has_target = isinstance(target, (int, float)) and not isinstance(target, bool)
        rows.append([
            f"{r['metric']} {LOWER_IS_BETTER}" if lower else r["metric"],
            _model_value_text(r, metrics),
            f"{r['ref_low']:.2f} ~ {r['ref_high']:.2f}",
            POSITION_LABEL[position] if position else DASH,
            ed4_target(target) if has_target else DASH,
            POSITION_LABEL[r["target_position"]] if has_target and r.get("target_position") in POSITION_LABEL else DASH,
        ])
    return rows


def ref_table_piece(benchmark_refs: list[dict], metrics) -> Piece | None:
    if not benchmark_refs:
        return None
    rows = ref_table_rows(benchmark_refs, metrics)
    notes = list(dict.fromkeys(f"{NOTE_PREFIX}{r['source_note']}" for r in benchmark_refs if r.get("source_note")))
    numbers: list[tuple[str, str | None]] = []
    for r, row in zip(benchmark_refs, rows):
        name = r["metric"]
        if row[1] != DASH:
            numbers.append((row[1], f"benchmark.{name}.model_value"))
        low, high = row[2].split(" ~ ")
        numbers += [(low, f"benchmark.{name}.ref_low"), (high, f"benchmark.{name}.ref_high")]
        if row[4] != DASH:
            numbers.append((row[4], f"benchmark.{name}.target"))
    text = _table_text([], list(REF_COLUMNS), rows) + ("\n" + "\n".join(notes) if notes else "")
    return Piece("4.6-table", "8.1", "4.6", "table", text, tuple(numbers))


def build_pieces(result, section_sources: dict, material: JudgeMaterial | None, benchmark_refs: list[dict], metrics) -> list[Piece]:
    """서술 결과(NarrativeResponse 또는 NarrativeJobResult 모양) + 프론트 재료 + 참조 범위 → 조각 목록(판 3 칸 순서)."""
    pieces: list[Piece] = []
    slot_text: dict[str, str] = {}
    for sid, (group, name) in NARRATIVE_PATHS.items():
        text = getattr(getattr(result, group), name) or ""
        slot_text[sid] = text
        if text.strip():
            pieces.append(Piece(f"slot:{sid}", sid, ED4_PLACE[sid], SOURCE_OF[section_sources[sid]], text))
    rec_source = SOURCE_OF[section_sources["9"]]
    for place, name in RECOMMEND_PATHS:
        text = getattr(result.recommendation_narrative, name) or ""
        if text.strip():
            pieces.append(Piece(f"slot:9:{place}", "9", place, rec_source, text))

    for seg in material.segments if material else ():
        if seg.when == "slot_empty":
            if slot_text.get(seg.slot, "").strip() or (seg.slot == "8.1" and section_sources.get("8.1") != "fixed"):
                continue
        if seg.kind == "fixed":
            pieces.append(Piece(f"m:{seg.id}", seg.slot, seg.place, "fixed", seg.text,
                                tuple((n.text, n.field) for n in seg.numbers)))
        elif seg.kind == "table":
            pieces.append(Piece(f"m:{seg.id}", seg.slot, seg.place, "table",
                                _table_text(seg.text.split("\n"), seg.header, seg.rows),
                                tuple((n.text, n.field) for n in seg.numbers)))
        else:  # recommend_rows — 문제마다 서술 결과의 권고 수만큼 줄(화면 권고표와 같은 차례)
            for (priority, label), pid in zip(seg.rows, seg.row_keys):
                recs = [r for r in result.recommendations if r.problem_id == pid]
                for i, rec in enumerate(recs):
                    cells = [priority, label, rec.action, rec.expected_impact]
                    text = " / ".join(f"{h}: {c}" for h, c in zip(seg.header, cells))
                    pieces.append(Piece(f"rec:{seg.place}:{pid}:{i}", "9", seg.place, rec_source, text))

    ref = ref_table_piece(benchmark_refs, metrics)
    if ref is not None:
        pieces.append(ref)
    return sorted(pieces, key=lambda p: SLOT_ORDER.index(p.slot))


def for_module(pieces: list[Piece], module: str) -> list[Piece]:
    """모듈이 받는 범위만(설계 4절). 모르는 모듈 이름이면 KeyError."""
    allowed = MODULE_SOURCES[module]
    return [p for p in pieces if p.source in allowed]


def render(pieces: list[Piece]) -> str:
    """LLM 에 줄 글 — 조각마다 머리 줄 [id | 칸 | 판 4 자리 | 출처] 다음에 글."""
    return "\n\n".join(f"[{p.id} | 칸 {p.slot} | {p.place} | {p.source}]\n{p.text}" for p in pieces)
