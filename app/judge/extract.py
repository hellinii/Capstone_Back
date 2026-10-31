"""app/judge/extract.py — 공용 전처리: 서술에서 사실 주장 뽑기(설계 5.3, 계획 10-3).

- 읽는 범위: AI 칸과 규칙 문장 조각만(reading.for_module "extract").
- 한 번 뽑고, 같은 모델에 빠진 주장을 최대 2번 더 묻는다(자기 점검). 2번째에도 새 주장이 나오면 넣고 "멈춤"을 기록한다.
  0·1·2번 시점 목록을 따로 남긴다(설계 7.1 추출 재현율 비교용).
- 칸은 코드가 조각 id 로 정한다(LLM 은 조각 id 만 고른다 — 문서 심판-40).
- 코드 검사: 조각 id 가 범위 밖이거나 인용 구절이 그 조각에 없으면 버리고 기록한다. 칸 이름이 목록에 없으면 빈 글로 바꾼다.
- 안전장치(Q4 c): 읽는 범위에 숫자가 있는데 주장이 0개면 전처리 실패(no_claims)로 본다.

상호작용
- 의존(import): app.judge.llm·config·criteria·reading·sources
- 사용처: app.judge.core(J4), 검증 실행기
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from app.judge import config, criteria, llm
from app.judge.reading import Piece, for_module, render
from app.judge.sources import FieldEntry, catalog_text

KINDS = ("number", "name", "comparison", "degree", "inference")
POLARITIES = ("positive", "negative", "neutral", "")
CALL_TIMEOUT_SECONDS = 120.0
_SPACE = re.compile(r"\s+")


@dataclass
class Claim:
    id: str
    kind: str
    quote: str
    piece_id: str
    slot: str
    place: str
    source: str
    field: str
    field_b: str
    number_text: str
    polarity: str
    round: int  # 0 = 첫 추출, 1·2 = 자기 점검에서 나옴
    unresolved_field: bool = False


@dataclass
class ExtractResult:
    claims: list[Claim]
    rounds: list[int]  # 시점마다(0·1·2번 뒤) 누적 주장 수
    stopped: bool  # 2번째 자기 점검에도 새 주장이 나왔다
    rejected: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    failure: str | None = None

    def as_record(self) -> dict:
        return {"claims": [asdict(c) for c in self.claims], "rounds": self.rounds, "stopped": self.stopped,
                "rejected": self.rejected, "calls": self.calls, "failure": self.failure}


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def claim_rules_text() -> dict[str, str]:
    rules = criteria.load_json("claims.json")
    return {
        "kinds": "\n".join(f"- {k['id']}({k['name']}): {k['what']} 예: {k['example']}" for k in rules["kinds"]),
        "unit_rules": "\n".join(f"- {r}" for r in rules["unit_rules"]),
        "not_claims": "\n".join(f"- {r}" for r in rules["not_claims"]),
        "inference_markers": ", ".join(criteria.claim_markers("inference_markers")),
    }


def instructions(catalog: dict[str, FieldEntry]) -> str:
    return criteria.load_text("prompts/extract.md").format(**claim_rules_text(), catalog=catalog_text(catalog))


def response_schema(piece_ids: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": ["claims"],
        "properties": {"claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["kind", "quote", "piece_id", "field", "field_b", "number_text", "polarity"],
            "properties": {
                "kind": {"type": "string", "enum": list(KINDS)},
                "quote": {"type": "string"},
                "piece_id": {"type": "string", "enum": piece_ids},
                "field": {"type": "string"},
                "field_b": {"type": "string"},
                "number_text": {"type": "string"},
                "polarity": {"type": "string", "enum": list(POLARITIES)},
            },
        }}},
    }


_SENT = re.compile(r"(?<=[.?!])\s+|\n+")


def _sentence_of(quote: str, piece: Piece) -> str:
    for s in _SENT.split(piece.text):
        if _norm(quote) in _norm(s):
            return s
    return quote


def _not_inference(sentence: str) -> str | None:
    """추론으로 뽑혔지만 규범 문장이거나 시험 범위 문장이면 그 사유(claims.json + inference_rules.json 의 normative_markers·scope_markers)."""
    if any(m in sentence for m in criteria.claim_markers("normative_markers")):
        return "normative"
    if any(m in sentence for m in criteria.claim_markers("scope_markers")):
        return "scope"
    return None


def _found_text(claims: list[Claim]) -> str:
    return "\n".join(f"- [{c.piece_id}] {c.kind}: {c.quote}" for c in claims) or "(없음)"


def _accept(raw: list, pieces: dict[str, Piece], catalog: dict[str, FieldEntry], have: list[Claim], round_no: int,
            rejected: list[dict]) -> list[Claim]:
    seen = {(c.piece_id, _norm(c.quote), c.kind) for c in have}
    new: list[Claim] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("kind") not in KINDS:
            rejected.append({"round": round_no, "reason": "shape", "item": str(item)[:200]})
            continue
        piece = pieces.get(item.get("piece_id"))
        quote = str(item.get("quote") or "")
        if piece is None:
            rejected.append({"round": round_no, "reason": "piece", "piece_id": item.get("piece_id"), "quote": quote[:200]})
            continue
        if not quote.strip() or _norm(quote) not in _norm(piece.text):
            rejected.append({"round": round_no, "reason": "quote", "piece_id": piece.id, "quote": quote[:200]})
            continue
        if item["kind"] == "inference":
            reason = _not_inference(_sentence_of(quote, piece))
            if reason:
                rejected.append({"round": round_no, "reason": reason, "piece_id": piece.id, "quote": quote[:200]})
                continue
        key = (piece.id, _norm(quote), item["kind"])
        if key in seen:
            continue
        seen.add(key)
        fld, fld_b = str(item.get("field") or ""), str(item.get("field_b") or "")
        unresolved = (fld and fld not in catalog) or (fld_b and fld_b not in catalog)
        polarity = item.get("polarity") if item.get("polarity") in POLARITIES else ""
        new.append(Claim(
            id="", kind=item["kind"], quote=quote, piece_id=piece.id, slot=piece.slot, place=piece.place, source=piece.source,
            field=fld if fld in catalog else "", field_b=fld_b if fld_b in catalog else "",
            number_text=str(item.get("number_text") or ""), polarity=polarity, round=round_no, unresolved_field=bool(unresolved),
        ))
    return new


async def extract_claims(client, pieces: list[Piece], catalog: dict[str, FieldEntry], *, deadline: float | None = None,
                         clock=None) -> ExtractResult:
    """주장 목록(자기 점검 포함). 호출 실패·형식 오류·주장 0개(숫자가 있는데)는 failure 로 돌려준다."""
    scope = for_module(pieces, "extract")
    by_id = {p.id: p for p in scope}
    rounds_max = int(criteria.load_json("claims.json")["self_check_rounds"])
    base = instructions(catalog)
    schema = llm.json_schema_format("claims", response_schema(list(by_id)))
    user = render(scope)
    claims: list[Claim] = []
    rejected: list[dict] = []
    calls: list[dict] = []
    rounds: list[int] = []
    stopped = False
    kw = {"clock": clock} if clock else {}
    for round_no in range(rounds_max + 1):
        text = base if round_no == 0 else base + "\n\n" + criteria.load_text("prompts/extract_self_check.md").format(
            found=_found_text(claims))
        out = await llm.create_response(
            client, purpose=f"extract_{round_no}", model=config.extract_model(), instructions=text,
            input=[{"role": "user", "content": user}], timeout_s=CALL_TIMEOUT_SECONDS, deadline=deadline,
            text_format=schema, effort=config.extract_effort(), **kw,
        )
        data, failure = llm.parse_json(out)
        calls.append(out.record.as_dict())
        if failure:
            return ExtractResult(claims, rounds, stopped, rejected, calls, failure)
        new = _accept(data.get("claims") or [], by_id, catalog, claims, round_no, rejected)
        claims.extend(new)
        rounds.append(len(claims))
        if round_no > 0 and not new:
            break
        if round_no == rounds_max and new:
            stopped = True
    for i, c in enumerate(claims, 1):
        c.id = f"C{i}"
    failure = "no_claims" if not claims and any(re.search(r"\d", p.text) for p in scope) else None
    return ExtractResult(claims, rounds, stopped, rejected, calls, failure)


def claims_json(result: ExtractResult) -> str:
    return json.dumps([asdict(c) for c in result.claims], ensure_ascii=False)
