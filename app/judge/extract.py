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
from collections import Counter
from dataclasses import asdict, dataclass, field

from app.judge import config, criteria, llm
from app.judge.criteria.rules_numbers import matches, parse_printed
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


# 절 나누기(1.1 검토): 쉼표, 그리고 '…며·…고·…지만·…으나·…는데·…아니라·…것으로'(1.5) 뒤의 띄어쓰기.
_CLAUSE_BREAK = re.compile(r"[,，]\s*|(?<=며)\s+|(?<=고)\s+|(?<=지만)\s+|(?<=으나)\s+|(?<=는데)\s+|(?<=아니라)\s+|(?<=것으로)\s+")


def _clause_of(quote: str, sentence: str) -> str:
    """문장 가운데 인용 구절이 걸친 절(들). 구절을 못 찾으면 문장 전체. 규범·범위 낱말은 이 범위에서만 본다 — 앞 절의 단정이 뒤
    절의 '…단정할 수 없다' 때문에 함께 버려지지 않게(1.1 검토)."""
    start = sentence.find(quote)
    if start < 0:
        return sentence
    end = start + len(quote)
    cuts = [0, *(m.end() for m in _CLAUSE_BREAK.finditer(sentence)), len(sentence)]
    lo = max(c for c in cuts if c <= start)
    hi = min((c for c in cuts if c >= end), default=len(sentence))
    # 끝 경계가 다음 절 시작이면 그 절은 넣지 않는다 — hi 는 구절 끝을 덮는 첫 경계
    return sentence[lo:hi]


def _not_inference(sentence: str) -> str | None:
    """추론으로 뽑혔지만 규범 문장이거나 시험 범위 문장이면 그 사유(claims.json + inference_rules.json 의 normative_markers·scope_markers)."""
    if (any(m in sentence for m in criteria.claim_markers("normative_markers"))
            and not any(w in sentence for w in criteria.claim_markers("normative_not_when"))):
        return "normative"  # '…로 읽어야 한다'처럼 해석을 감싼 꼴은 규범이 아니다(normative_not_when, 1.5)
    if any(m in sentence for m in criteria.claim_markers("scope_markers")):
        return "scope"
    return None


def occurrences(quote: str, text: str) -> int:
    """인용이 글에 나오는 횟수(띄어쓰기 정리 뒤). 숫자로 시작·끝나는 인용은 다른 숫자의 일부를 세지 않는다('171건' 안의 '1건')."""
    q, t = _norm(quote), _norm(text)
    count, i = 0, t.find(q) if q else -1
    while i >= 0:
        before = t[i - 1] if i > 0 else ""
        after = t[i + len(q)] if i + len(q) < len(t) else ""
        if not ((q[0].isdigit() and (before.isdigit() or before == ".")) or
                (q[-1].isdigit() and (after.isdigit() or after == "."))):
            count += 1
        i = t.find(q, i + 1)
    return max(count, 1)


def _not_a_claim(item: dict, quote: str, sentence: str) -> str | None:
    """종류와 상관없이 주장이 아닌 것(1.12, 검증 추출 분석 — 사람 정답지·민서 판정과 맞춤): 권고 문장('권고: …') 안의 말,
    판정 기준값 숫자 바로 뒤에 경계 말·'신뢰구간'이 오는 기준선 숫자('0.1 미만'·'95% 신뢰구간')."""
    if any(sentence.strip().startswith(p) for p in criteria.claim_markers("recommendation_prefix")):
        return "recommendation"
    return None


def is_threshold_number(claim: Claim, sentence: str) -> bool:
    """판정 기준값 숫자 바로 뒤에 경계 말·'신뢰구간'이 오는 기준선 숫자('0.1 미만'·'95% 신뢰구간')인가(1.12)."""
    if claim.kind != "number":
        return False
    text = (claim.number_text or claim.quote).strip()
    if parse_printed(text) is None:
        return False
    at = sentence.find(text)
    rest = sentence[at + len(text):].lstrip() if at >= 0 else ""
    constants = criteria.load_json("rule_constants.json")["values"].values()
    return any(matches(text, c) for c in constants) and any(rest.startswith(w) for w in criteria.claim_markers("threshold_after"))


def drop_covered_thresholds(claims: list[Claim], pieces: dict[str, Piece], rejected: list[dict]) -> list[Claim]:
    """기준선 숫자는 같은 문장에 그 숫자를 담은 다른 종류 주장(대개 비교 '…0.1 미만')이 있을 때만 버린다 — 그 비교와 겹치는 중복이다.
    비교 주장이 없으면 남긴다(사람 정답지는 기준선을 비교 주장으로 표시했다 — 숫자만 뽑혀도 그 짝이다. 1.13, V2 1.12 R-0052:
    1.12 가 기준선 숫자를 무조건 버려 정답 짝 16개를 잃었다)."""
    kept = []
    for c in claims:
        piece = pieces.get(c.piece_id)
        sentence = _sentence_of(c.quote, piece) if piece else c.quote
        if piece and is_threshold_number(c, sentence):
            text = (c.number_text or c.quote).strip()
            covered = any(o is not c and o.kind != "number" and o.piece_id == c.piece_id and text in o.quote
                          and _norm(o.quote) in _norm(sentence) for o in claims)
            if covered:
                rejected.append({"round": c.round, "reason": "threshold", "piece_id": c.piece_id, "quote": c.quote[:200]})
                continue
        kept.append(c)
    return kept


def empty_sentences(scope: list[Piece], claims: list[Claim], limit: int = 400) -> str:
    """아직 주장을 하나도 뽑지 않은 문장 목록(자기 점검용, 1.13 — 숫자 없는 문장을 통째로 건너뛰었다). 1.15: 60 문장에서 잘라
    뒤쪽 문장이 목록에 들지 못했다 — 서술 전체가 들도록 넉넉히."""
    lines = []
    for p in scope:
        quotes = [_norm(c.quote) for c in claims if c.piece_id == p.id]
        for s in _SENT.split(p.text):
            if s.strip() and not any(q and q in _norm(s) for q in quotes):
                lines.append(f"- [{p.id}] {s.strip()}")
    return "\n".join(lines[:limit]) or "(없음)"


def _found_text(claims: list[Claim]) -> str:
    return "\n".join(f"- [{c.piece_id}] {c.kind}: {c.quote}" for c in claims) or "(없음)"


def _accept(raw: list, pieces: dict[str, Piece], catalog: dict[str, FieldEntry], have: list[Claim], round_no: int,
            rejected: list[dict]) -> list[Claim]:
    seen = Counter((c.piece_id, _norm(c.quote), c.kind) for c in have)
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
        sentence = _sentence_of(quote, piece)
        reason = _not_a_claim(item, quote, sentence)
        if reason:
            rejected.append({"round": round_no, "reason": reason, "piece_id": piece.id, "quote": quote[:200]})
            continue
        if item["kind"] == "inference":
            reason = _not_inference(_clause_of(quote, sentence))
            if reason:
                rejected.append({"round": round_no, "reason": reason, "piece_id": piece.id, "quote": quote[:200]})
                continue
        key = (piece.id, _norm(quote), item["kind"])
        # 같은 인용은 칸 글에 나오는 횟수만큼 받는다 — 주장 규칙 "같은 숫자를 두 번 쓰면 두 번 센다"(claims.json unit_rules).
        # 예전엔 하나만 남겨 되풀이 숫자('1건'·'0.984')를 놓쳤다(1.11, 검증 추출 미달 분석: 놓친 244건 중 56건).
        if seen[key] >= occurrences(quote, piece.text):
            rejected.append({"round": round_no, "reason": "duplicate", "piece_id": piece.id, "quote": quote[:200]})
            continue
        seen[key] += 1
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
            found=_found_text(claims), empty=empty_sentences(scope, claims))
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
    claims = drop_covered_thresholds(claims, by_id, rejected)
    for i, c in enumerate(claims, 1):
        c.id = f"C{i}"
    failure = "no_claims" if not claims and any(re.search(r"\d", p.text) for p in scope) else None
    return ExtractResult(claims, rounds, stopped, rejected, calls, failure)


def claims_json(result: ExtractResult) -> str:
    return json.dumps([asdict(c) for c in result.claims], ensure_ascii=False)
