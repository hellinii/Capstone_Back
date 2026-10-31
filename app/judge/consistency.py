"""app/judge/consistency.py — 일관성(설계 6.3, 계획 11-3). 코드 + 단일 호출.

- 값 불일치(코드): 숫자 주장(공용 전처리, 칸이 정해진 것)과 읽는 범위 안 표·고정 문구의 숫자(재료의 칸 이름, 4.6 표)를 칸별로
  묶어, 같은 칸의 찍힌 숫자가 같은 양으로 읽히는지(criteria/rules_numbers.same_quantity) 본다. 묶음 단위로 센다(Q4 h —
  읽는 범위 밖은 읽지 않는다).
- 논리 모순 3유형(LLM, GPT-6.1 Sol): 판정 ↔ 권고, 약점 ↔ 권고, 분석 ↔ 소견. 입력은 읽는 범위 글 전체 + 판정 결과뿐(사실 시트·
  주장 목록·데이터 원본은 주지 않는다 — 측정의 순수성). 있다면 어긋나는 두 구절을 인용해야 한다. 인용이 빠지거나 글에 없으면
  그 유형만 한 번 다시 묻고, 그래도 없으면 실패(uncited — 보류)다(Q4 c, 설계 7.3 안전장치).

상호작용
- 의존(import): app.judge.llm·config·criteria(rules_numbers)·reading·sources(ALIASES)·extract(Claim)
- 사용처: app.judge.core(J4)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.judge import config, criteria, llm
from app.judge.criteria.rules_numbers import matches, parse_printed, same_quantity
from app.judge.extract import Claim
from app.judge.reading import Piece, for_module, render
from app.judge.sources import ALIASES

CALL_TIMEOUT_SECONDS = 180.0


@dataclass
class Contradiction:
    type: str
    pieces: list[str]
    slots: list[str]
    sources: list[str]
    quotes: list[str]
    detail: dict = field(default_factory=dict)


@dataclass
class ConsistencyResult:
    contradictions: list[Contradiction]
    counts: dict
    calls: list[dict] = field(default_factory=list)
    failure: str | None = None

    def as_record(self) -> dict:
        return {"contradictions": [asdict(c) for c in self.contradictions], "counts": self.counts, "calls": self.calls,
                "failure": self.failure}


def _canonical(name: str) -> str:
    return ALIASES.get(name, name)


def value_groups(pieces: list[Piece], claims: list[Claim]) -> dict[str, list[tuple[str, Piece]]]:
    by_id = {p.id: p for p in pieces}
    groups: dict[str, list[tuple[str, Piece]]] = {}
    for c in claims:
        if c.kind == "number" and c.field and c.number_text and c.piece_id in by_id:
            groups.setdefault(_canonical(c.field), []).append((c.number_text, by_id[c.piece_id]))
    for p in pieces:
        if p.source in ("fixed", "table"):
            for text, fld in p.numbers:
                if fld:
                    groups.setdefault(_canonical(fld), []).append((text, p))
    return groups


def _misassigned(text: str, piece: Piece, fld: str, data: dict | None) -> bool:
    """서술 쪽 숫자가 묶인 칸의 데이터와는 다르지만 다른 칸의 데이터 값과는 맞나, 또는 그 칸 값의 표기일 수 없나(비율 칸에 1보다
    큰 수, 1.1) — 추출이 칸을 잘못 짚은 것으로 본다(J4 파일럿).
    표·고정 문구 숫자는 프론트가 칸을 알고 찍으므로 보지 않는다."""
    if not data or piece.source not in ("ai", "template"):
        return False
    own = data.get(fld)
    if own is not None and matches(text, own):
        return False
    if own is not None and _not_a_ratio_print(text, own):
        return True
    return any(matches(text, v) for name, v in data.items() if name != fld)


def _not_a_ratio_print(text: str, own) -> bool:
    """칸 값이 비율(0~1)인데 서술 숫자가 백분율도 아닌 1보다 큰 수 — 그 칸 값을 어떻게 찍어도 나올 수 없는 숫자라 칸을 잘못
    붙인 것이다(1.1, V2 1차 — '2,000회' 를 신뢰수준 0.95 칸에 붙여 '95%' 와 모순으로 셌다)."""
    p = parse_printed(text)
    return p is not None and not p.percent and 0 <= own <= 1 and abs(p.value) > 1


def value_mismatches(groups: dict[str, list[tuple[str, Piece]]], data: dict | None = None) -> tuple[list[Contradiction], int, list[dict]]:
    """(모순 목록, 검사한 묶음 수, 칸을 잘못 짚은 것으로 보고 넘긴 것). 숫자가 둘 이상인 묶음만 검사한다. 묶음마다 모순은 하나.
    data(칸 이름 → 값)는 코드만 본다 — LLM 에는 가지 않는다(설계 6.3 '지표표·주장 목록은 값 불일치를 보는 코드에만')."""
    found: list[Contradiction] = []
    skipped: list[dict] = []
    checked = 0
    for fld, entries in groups.items():
        if len(entries) < 2:
            continue
        checked += 1
        for (ta, pa), (tb, pb) in ((entries[i], entries[j]) for i in range(len(entries)) for j in range(i + 1, len(entries))):
            if same_quantity(ta, tb) is not False:
                continue
            if _misassigned(ta, pa, fld, data) or _misassigned(tb, pb, fld, data):
                skipped.append({"field": fld, "numbers": [ta, tb], "pieces": [pa.id, pb.id]})
                continue
            found.append(Contradiction("value_mismatch", [pa.id, pb.id], [pa.slot, pb.slot], [pa.source, pb.source], [ta, tb],
                                       {"field": fld}))
            break
    return found, checked, skipped


def logic_types() -> list[dict]:
    return [t for t in criteria.load_json("contradictions.json")["types"] if t["by"] == "llm"]


def instructions() -> str:
    types = "\n".join(f"- {t['id']}({t['name']}): {t['what']}" for t in logic_types())
    return criteria.load_text("prompts/consistency.md").format(types=types)


def answer_schema(type_ids: list[str], piece_ids: list[str]) -> dict:
    s = {"type": "string"}
    pid = {"type": "string", "enum": [*piece_ids, ""]}
    return {
        "type": "object", "additionalProperties": False, "required": ["checks"],
        "properties": {"checks": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["type", "present", "quote_a", "piece_a", "quote_b", "piece_b"],
            "properties": {"type": {"type": "string", "enum": type_ids}, "present": {"type": "boolean"},
                           "quote_a": s, "piece_a": pid, "quote_b": s, "piece_b": pid},
        }}},
    }


def verdict_summary(fs) -> str:
    """판정 결과만(값 없음): 종합 판정과 기준을 넘은·못 넘은 지표 이름."""
    judged = [m for m in fs.metrics if m.threshold is not None and m.status in ("pass", "fail")]
    passed = [m.display_name for m in judged if m.status == "pass"]
    failed = [m.display_name for m in judged if m.status == "fail"]
    return (f"종합 판정: {fs.verdict}\n기준 충족 지표: {', '.join(passed) or '없음'}\n"
            f"기준 미달 지표: {', '.join(failed) or '없음'}")


def _norm(t: str) -> str:
    return " ".join(t.split())


def _cited(check: dict, by_id: dict[str, Piece]) -> bool:
    for q, p in ((check.get("quote_a", ""), check.get("piece_a", "")), (check.get("quote_b", ""), check.get("piece_b", ""))):
        if not q.strip() or p not in by_id or _norm(q) not in _norm(by_id[p].text):
            return False
    return True


async def judge_consistency(client, pieces: list[Piece], claims: list[Claim], fs, *, data: dict | None = None,
                            deadline: float | None = None, clock=None) -> ConsistencyResult:
    scope = for_module(pieces, "consistency")
    by_id = {p.id: p for p in scope}
    mismatches, value_checked, skipped = value_mismatches(value_groups(scope, claims), data)
    types = logic_types()
    type_ids = [t["id"] for t in types]
    schema = llm.json_schema_format("consistency", answer_schema(type_ids, list(by_id)))
    user = f"## 서술\n{render(scope)}\n\n## 판정 결과\n{verdict_summary(fs)}"
    kw = {"clock": clock} if clock else {}
    calls: list[dict] = []
    answers: dict[str, dict] = {}
    ask = type_ids
    for attempt in (1, 2):
        extra = "" if attempt == 1 else (
            "\n\n앞 답에서 아래 유형은 모순이 있다고 했지만 두 구절 인용이 없거나 글에 없었다. 이 유형만 다시 답하라: " + ", ".join(ask))
        out = await llm.create_response(
            client, purpose=f"consistency_{attempt}", model=config.judge_model(), instructions=instructions() + extra,
            input=[{"role": "user", "content": user}], timeout_s=CALL_TIMEOUT_SECONDS, deadline=deadline,
            text_format=schema, **kw,
        )
        data, failure = llm.parse_json(out)
        calls.append(out.record.as_dict())
        if failure:
            return ConsistencyResult([], {}, calls, failure)
        for check in data.get("checks") or []:
            if isinstance(check, dict) and check.get("type") in ask:
                answers[check["type"]] = check
        if attempt == 1 and any(t not in answers for t in type_ids):
            return ConsistencyResult([], {}, calls, "bad_output")  # 세 유형 모두 답해야 한다
        ask = [t for t in type_ids if answers[t].get("present") and not _cited(answers[t], by_id)]
        if not ask:
            break
    else:
        return ConsistencyResult([], {}, calls, "uncited")
    logic = [
        Contradiction(t, [a["piece_a"], a["piece_b"]], [by_id[a["piece_a"]].slot, by_id[a["piece_b"]].slot],
                      [by_id[a["piece_a"]].source, by_id[a["piece_b"]].source], [a["quote_a"], a["quote_b"]])
        for t, a in answers.items() if a.get("present")
    ]
    found = mismatches + logic
    counts = {"checked": value_checked + len(type_ids), "value_groups_checked": value_checked, "logic_types_checked": len(type_ids),
              "value_skipped_misassigned": len(skipped),
              "contradictions": len(found),
              "by_type": {t: sum(1 for c in found if c.type == t) for t in ["value_mismatch", *type_ids]}}
    return ConsistencyResult(found, counts, calls)
