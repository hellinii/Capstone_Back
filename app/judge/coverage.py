"""app/judge/coverage.py — 필수 내용(설계 6.2, 계획 11-4). 코드 고르기 + 단일 호출.

- 적용 항목은 코드가 요청 데이터로 고른다(criteria/rules_conditions, Q4 d). 분모 = 적용 항목 수.
- LLM(GPT-6.1 Sol)은 그 항목에만 있다/없다 + 인용. 받는 범위는 전부(고정 문구·표 포함), 주장 목록은 받지 않는다.
- 근거(Q5): 인용 조각 출처가 ai·template·fixed 면 "서술됨", 표(table)면 "없다"로 본다(맥락으로만).
- "있다"는 인용이 그 조각 글에 있어야 한다. 빠지거나 없으면 그 항목만 한 번 다시 묻고, 그래도 없으면 실패(uncited — 보류, Q4 c).
- 부록 체크리스트에는 인용 조각의 실제 판 4 자리(place)를 쓴다(5.5 — 판 4 에서 옮겨 간 글이면 4.4 같은 자리).

상호작용
- 의존(import): app.judge.llm·config·criteria(rules_conditions)·reading
- 사용처: app.judge.core
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.judge import config, criteria, llm
from app.judge.criteria.rules_conditions import applicable_items
from app.judge.reading import Piece, for_module, render

CALL_TIMEOUT_SECONDS = 180.0


@dataclass
class ItemResult:
    id: str
    described: bool
    piece_id: str
    slot: str
    place: str
    quote: str


@dataclass
class CoverageResult:
    items: list[ItemResult]
    not_applicable: list[str]
    counts: dict
    calls: list[dict] = field(default_factory=list)
    failure: str | None = None

    def as_record(self) -> dict:
        return {"items": [asdict(i) for i in self.items], "not_applicable": self.not_applicable, "counts": self.counts,
                "calls": self.calls, "failure": self.failure}


def instructions(items: list[dict]) -> str:
    text = "\n".join(f"- {i['id']}: {i['reader_learns']}" for i in items)
    return criteria.load_text("prompts/coverage.md").format(items=text)


def answer_schema(item_ids: list[str], piece_ids: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": ["items"],
        "properties": {"items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "present", "quote", "piece_id"],
            "properties": {"id": {"type": "string", "enum": item_ids}, "present": {"type": "boolean"},
                           "quote": {"type": "string"}, "piece_id": {"type": "string", "enum": [*piece_ids, ""]}},
        }}},
    }


def _norm(t: str) -> str:
    return " ".join(t.split())


def _cited(a: dict, by_id: dict[str, Piece]) -> bool:
    p = by_id.get(a.get("piece_id", ""))
    return p is not None and bool(a.get("quote", "").strip()) and _norm(a["quote"]) in _norm(p.text)


async def judge_coverage(client, pieces: list[Piece], task_type: str, fs, client_inputs, benchmark_refs: list[dict], *,
                         deadline: float | None = None, clock=None) -> CoverageResult:
    checklist = criteria.load_json("checklist.json")
    counted = set(checklist["evidence_rule"]["counts"])
    on, off = applicable_items(checklist["items"], task_type, fs, client_inputs, benchmark_refs)
    scope = for_module(pieces, "coverage")
    by_id = {p.id: p for p in scope}
    ids = [i["id"] for i in on]
    if not ids:
        return CoverageResult([], [i["id"] for i in off], {"applicable": 0, "described": 0, "not_applicable": len(off)})
    schema = llm.json_schema_format("coverage", answer_schema(ids, list(by_id)))
    user = render(scope)
    kw = {"clock": clock} if clock else {}
    answers: dict[str, dict] = {}
    calls: list[dict] = []
    ask = ids
    for attempt in (1, 2):
        items_now = [i for i in on if i["id"] in ask]
        extra = "" if attempt == 1 else "\n\n앞 답에서 아래 항목은 '있다'고 했지만 인용이 없거나 글에 없었다. 이 항목만 다시 답하라."
        out = await llm.create_response(
            client, purpose=f"coverage_{attempt}", model=config.judge_model(), instructions=instructions(items_now) + extra,
            input=[{"role": "user", "content": user}], timeout_s=CALL_TIMEOUT_SECONDS, deadline=deadline, text_format=schema, **kw,
        )
        data, failure = llm.parse_json(out)
        calls.append(out.record.as_dict())
        if failure:
            return CoverageResult([], [i["id"] for i in off], {}, calls, failure)
        for a in data.get("items") or []:
            if isinstance(a, dict) and a.get("id") in ask:
                answers[a["id"]] = a
        if attempt == 1 and any(i not in answers for i in ids):
            return CoverageResult([], [i["id"] for i in off], {}, calls, "bad_output")
        ask = [i for i in ids if answers[i].get("present") and not _cited(answers[i], by_id)]
        if not ask:
            break
    else:
        return CoverageResult([], [i["id"] for i in off], {}, calls, "uncited")
    results: list[ItemResult] = []
    for item_id in ids:
        a = answers[item_id]
        piece = by_id.get(a.get("piece_id", ""))
        described = bool(a.get("present")) and piece is not None and piece.source in counted
        results.append(ItemResult(item_id, described, piece.id if described else "", piece.slot if described else "",
                                  piece.place if described else "", a.get("quote", "") if described else ""))
    counts = {"applicable": len(ids), "described": sum(r.described for r in results), "not_applicable": len(off)}
    return CoverageResult(results, [i["id"] for i in off], counts, calls)
