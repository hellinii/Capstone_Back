"""app/judge/extract.py — 공용 전처리: 서술에서 사실 주장 뽑기(설계 5.3, 계획 10-3).

- 읽는 범위: AI 칸과 규칙 문장 조각만(reading.for_module "extract").
- 한 번 뽑고, 같은 모델에 빠진 주장을 최대 2번 더 묻는다(자기 점검). 2번째에도 새 주장이 나오면 넣고 "멈춤"을 기록한다.
  0·1·2번 시점 목록을 따로 남긴다(설계 7.1 추출 재현율 비교용).
- 문장마다 판정(1.17): 첫 추출은 문장 id 마다 "주장 n개" 또는 "주장 없음 + 이유"를 먼저 낸다. 판정이 없는데 주장도 없는 문장,
  이유 없는 "주장 없음" 문장은 그 문장들만 한 번 다시 묻는다(채우기 — 첫 추출과 같은 0번 시점). 1.18: 덜 뽑힌 문장은 다시 묻지 않는다.
- 자기 점검(1.17)은 문장마다 이미 뽑은 주장을 붙여 보여 주고 꾸밈말·비교·집계·판정 사유·사용자 입력값 가운데 빠진 것만 묻는다.
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
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

from app.judge import config, criteria, llm
from app.judge.criteria.rules_numbers import matches, parse_printed
from app.judge.reading import Piece, for_module, render
from app.judge.sources import FieldEntry, catalog_text

KINDS = ("number", "name", "comparison", "degree", "inference")
POLARITIES = ("positive", "negative", "neutral", "")
CALL_TIMEOUT_SECONDS = 120.0
# 채우기(1.17)는 남은 시간이 이보다 적으면 건너뛴다 — 추출 뒤 사실 일치(실측 약 70~75초)와 채우기 자체(약 15~20초)가 들어갈 자리.
FILL_MIN_SECONDS = 100.0
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
    # 문장마다 판정(1.17): 문장 수, 첫 추출 뒤 다시 물은 문장 {sid: 까닭}, 채우기 뒤에도 남은 문장
    sentence_checks: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        return {"claims": [asdict(c) for c in self.claims], "rounds": self.rounds, "stopped": self.stopped,
                "rejected": self.rejected, "calls": self.calls, "failure": self.failure,
                "sentence_checks": self.sentence_checks}


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


def response_schema(piece_ids: list[str], sentence_ids: list[str] | None = None) -> dict:
    """주장 목록. sentence_ids 를 주면 주장 다음에 문장마다 판정(sentences)을 점검표로 내게 한다. 1.17 은 판정을 먼저 내게 했는데
    1.18 R-0059 에서 판정 목록만 쓰고 주장 목록을 비운 채 끝났다(79문장 '있다'고 하고 주장 0개) — 1.19 부터 주장이 먼저다."""
    props: dict = {}
    if sentence_ids is not None:
        props["sentences"] = {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["sid", "n_claims", "none_reason"],
            "properties": {"sid": {"type": "string", "enum": sentence_ids}, "n_claims": {"type": "integer"},
                           "none_reason": {"type": "string"}},
        }}
    return {
        "type": "object", "additionalProperties": False, "required": ["claims", *props],
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
        }}, **props},
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
    """인용이 글에 나오는 횟수(띄어쓰기 정리 뒤, 적어도 1). 숫자로 시작·끝나는 인용은 다른 숫자의 일부를 세지 않는다('171건' 안의 '1건')."""
    return max(hits(quote, text), 1)


def hits(quote: str, text: str) -> int:
    """occurrences 와 같되 없으면 0."""
    q, t = _norm(quote), _norm(text)
    count, i = 0, t.find(q) if q else -1
    while i >= 0:
        before = t[i - 1] if i > 0 else ""
        after = t[i + len(q)] if i + len(q) < len(t) else ""
        if not ((q[0].isdigit() and (before.isdigit() or before == ".")) or
                (q[-1].isdigit() and (after.isdigit() or after == "."))):
            count += 1
        i = t.find(q, i + 1)
    return count


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


@dataclass(frozen=True)
class Sentence:
    sid: str
    piece_id: str
    text: str


def sentences_of(scope: list[Piece]) -> list[Sentence]:
    """읽는 범위의 문장에 차례로 s1, s2, … id 를 붙인다(1.17 — 문장마다 판정·채우기·자기 점검의 단위)."""
    out: list[Sentence] = []
    for p in scope:
        for t in _SENT.split(p.text):
            if t.strip():
                out.append(Sentence(f"s{len(out) + 1}", p.id, t.strip()))
    return out


def render_sentences(sents: list[Sentence]) -> str:
    return "\n".join(f"[{s.sid}] ({s.piece_id}) {s.text}" for s in sents) or "(없음)"


def _in_sentence(piece_id: str, quote: str, s: Sentence) -> bool:
    return bool(_norm(quote)) and piece_id == s.piece_id and hits(quote, s.text) > 0


def place(sents: list[Sentence], items: list[tuple[str, str]]) -> list[str | None]:
    """(조각 id, 인용)마다 붙을 문장 id 하나(검토 P1 — 같은 글자가 여러 문장에 있어도 한 문장에만). 같은 (조각, 인용)의 k 번째는
    그 인용이 나오는 k 번째 자리의 문장으로(숫자 경계를 본다 — '171건' 안의 '1건'은 자리가 아니다). 자리가 모자라면 마지막 자리."""
    seen: Counter = Counter()
    out: list[str | None] = []
    for pid, quote in items:
        spots = [s.sid for s in sents if s.piece_id == pid for _ in range(hits(quote, s.text))] if _norm(quote) else []
        k = seen[(pid, _norm(quote))]
        seen[(pid, _norm(quote))] += 1
        out.append(spots[min(k, len(spots) - 1)] if spots else None)
    return out


MISSED_WORD = "빠뜨림"  # 점검표의 none_reason 에 이 낱말이 있으면 그 문장을 다시 묻는다(prompts/extract.md 문장마다 판정, 1.19)
# 모델이 냈지만 코드가 주장 규칙으로 버린 것 — 그 문장을 본 것으로 친다(다시 물어도 같은 답이 온다). 인용·조각이 틀린 것은 치지 않는다.
_LOOKED = ("normative", "scope", "recommendation", "duplicate")


def sentence_gaps(sents: list[Sentence], verdicts: list, claims: list[Claim], rejected: list[dict]) -> dict[str, str]:
    """다시 물을 문장 {sid: 까닭}(1.17). 까닭: missing(판정도 주장도 없음) · lost(주장이 있다고 했는데 하나도 안 들어감 — 인용이
    틀려 버려진 것 등) · missed(까닭에 '빠뜨림', 1.19) · no_reason(이유 없는 '없음').
    판정이 없어도 주장이 들어간 문장은 본 것으로 친다. 1.18: '적은 수보다 덜 뽑힘'(short)으로는 다시 묻지 않는다 — R-0058 에서 109문장 중
    51문장이 short 로 불려 채우기가 21초를 더했고, 수를 채우려 설명·한계 문장까지 뽑게 했다(정밀도 0.789). 적은 수는 기록에만 남는다."""
    said: dict[str, dict] = {}
    for v in verdicts or []:
        if isinstance(v, dict) and isinstance(v.get("sid"), str):
            said.setdefault(v["sid"], v)
    looked = [(r["piece_id"], r.get("quote") or "") for r in rejected if r.get("reason") in _LOOKED and r.get("piece_id")]
    count = Counter(place(sents, [(c.piece_id, c.quote) for c in claims])) + Counter(place(sents, looked))
    gaps: dict[str, str] = {}
    for s in sents:
        got = count[s.sid]
        v = said.get(s.sid)
        n = v.get("n_claims") if v else None
        n = n if isinstance(n, int) and not isinstance(n, bool) else None
        if v is None or n is None:
            if not got:
                gaps[s.sid] = "missing"
        elif n >= 1 and not got:
            gaps[s.sid] = "lost"
        elif not got and MISSED_WORD in str(v.get("none_reason") or ""):
            gaps[s.sid] = "missed"  # 점검표를 쓰다 빠뜨린 것을 모델이 스스로 적음(1.19)
        elif n <= 0 and not got and not str(v.get("none_reason") or "").strip():
            gaps[s.sid] = "no_reason"
    return gaps


def by_sentence(sents: list[Sentence], claims: list[Claim], only: set[str] | None = None) -> str:
    """문장마다 이미 뽑은 주장을 붙인 목록(자기 점검·채우기용, 1.17). 1.13·1.15 의 '빈 문장 목록'을 대신한다 — 빈 문장은 '(아직 없음)'."""
    lines = []
    at = place(sents, [(c.piece_id, c.quote) for c in claims])  # 자리는 늘 문장 전체로 정하고, only 로 보여 줄 문장만 거른다
    for s in sents:
        if only is not None and s.sid not in only:
            continue
        got = [f"{c.kind}: {c.quote}" for c, sid in zip(claims, at) if sid == s.sid]
        lines.append(f"[{s.sid}] ({s.piece_id}) {s.text}\n    이미 뽑은 것: " + (" / ".join(got) if got else "(아직 없음)"))
    return "\n".join(lines) or "(없음)"


REPAIR_MIN_SHARE = 0.75
REPAIR_MIN_CHARS = 6


def repair_quote(quote: str, text: str) -> str | None:
    """인용이 조각 글에 글자 그대로 없을 때, 끝말만 바꿔 옮긴 것이면('…않으므로' → '…않는다') 글에 있는 앞부분으로 되살린다(1.21,
    V2 1.20 R-0060: 끝말을 바꾼 인용 4개를 버렸다). 앞부분이 인용의 75% 이상·6자 이상이어야 한다. 못 살리면 None."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return None
    if q in t:
        return quote
    floor = max(REPAIR_MIN_CHARS, -(-len(q) * 3 // 4))
    for k in range(len(q) - 1, floor - 1, -1):
        head = q[:k].rstrip()
        if re.search(r"\d", q[len(head):]):
            break  # 잘려 나갈 끝에 숫자가 있으면 되살리지 않는다 — 숫자를 바꾸는 셈이다('0.9123' → '0.912')
        if len(head) >= floor and head in t:
            return head
    return None


def _accept(raw: list, pieces: dict[str, Piece], catalog: dict[str, FieldEntry], have: list[Claim], round_no: int,
            rejected: list[dict], repaired: list[dict] | None = None) -> list[Claim]:
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
        fixed = repair_quote(quote, piece.text)
        if fixed is None:
            rejected.append({"round": round_no, "reason": "quote", "piece_id": piece.id, "quote": quote[:200]})
            continue
        if fixed != quote:
            if repaired is not None:
                repaired.append({"round": round_no, "piece_id": piece.id, "from": quote[:200], "to": fixed})
            quote = fixed
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


def fill_items(raw: list, sents: list[Sentence], asked: set[str], rejected: list[dict]) -> list:
    """채우기 답 가운데 다시 물은 문장 안의 것만(검토 P1). 버린 것은 rejected 에 fill_outside. 1.18: 다시 묻는 문장은 주장이 하나도
    없는 문장뿐이라 이미 뽑은 주장과의 겹침 거름(1.17 fill_overlap)은 뺐다."""
    kept = []
    for item in raw:
        if not isinstance(item, dict):
            kept.append(item)  # 모양 검사는 _accept 가 한다
            continue
        pid, quote = item.get("piece_id"), str(item.get("quote") or "")
        sid = next((x.sid for x in sents if x.sid in asked and _in_sentence(pid, quote, x)), None)
        if sid is None:
            rejected.append({"round": 0, "reason": "fill_outside", "piece_id": pid, "quote": quote[:200]})
            continue
        kept.append(item)
    return kept


async def extract_claims(client, pieces: list[Piece], catalog: dict[str, FieldEntry], *, deadline: float | None = None,
                         clock=None) -> ExtractResult:
    """주장 목록(문장마다 판정·채우기·자기 점검 포함). 호출 실패·형식 오류·주장 0개(숫자가 있는데)는 failure 로 돌려준다."""
    scope = for_module(pieces, "extract")
    by_id = {p.id: p for p in scope}
    sents = sentences_of(scope)
    sids = [x.sid for x in sents] or None
    rounds_max = int(criteria.load_json("claims.json")["self_check_rounds"])
    base = instructions(catalog)
    user = render(scope) + "\n\n## 문장 목록\n" + render_sentences(sents)
    claims: list[Claim] = []
    rejected: list[dict] = []
    calls: list[dict] = []
    rounds: list[int] = []
    checks: dict = {"sentences": len(sents), "gaps": {}, "left": {}, "repaired": []}
    stopped = False
    kw = {"clock": clock} if clock else {}

    async def ask(purpose: str, text: str, schema_sids: list[str] | None) -> tuple[dict | None, str | None]:
        out = await llm.create_response(
            client, purpose=purpose, model=config.extract_model(), instructions=text,
            input=[{"role": "user", "content": user}], timeout_s=CALL_TIMEOUT_SECONDS, deadline=deadline,
            text_format=llm.json_schema_format("claims", response_schema(list(by_id), schema_sids)),
            effort=config.extract_effort(), **kw,
        )
        data, failure = llm.parse_json(out)
        calls.append(out.record.as_dict())
        return data, failure

    def result(failure: str | None) -> ExtractResult:
        return ExtractResult(claims, rounds, stopped, rejected, calls, failure, checks)

    for round_no in range(rounds_max + 1):
        if round_no == 0:
            data, failure = await ask("extract_0", base, sids)
        else:
            data, failure = await ask(f"extract_{round_no}", base + "\n\n" + criteria.load_text(
                "prompts/extract_self_check.md").format(by_sentence=by_sentence(sents, claims)), None)
        if failure:
            return result(failure)
        new = _accept(data.get("claims") or [], by_id, catalog, claims, round_no, rejected, checks["repaired"])
        claims.extend(new)
        if round_no == 0:
            # 문장마다 판정 — 빠진 문장만, 한 번만 다시 묻는다(1.17). 채운 주장도 0번 시점이다.
            gaps = sentence_gaps(sents, data.get("sentences") or [], claims, rejected)
            checks["gaps"] = gaps
            remaining = None if deadline is None else deadline - (clock or time.monotonic)()
            if gaps and remaining is not None and remaining < FILL_MIN_SECONDS:
                checks["fill_skipped"] = "deadline"  # 시간이 모자라면 채우기 없이 간다(검토 P1)
            elif gaps:
                asked = [x for x in sents if x.sid in gaps]
                data, failure = await ask("extract_0_fill", base + "\n\n" + criteria.load_text(
                    "prompts/extract_fill.md").format(by_sentence=by_sentence(sents, claims, set(gaps))),
                    [x.sid for x in asked])
                if failure:
                    checks["fill_failure"] = failure  # 채우기는 덤 — 실패해도 첫 추출은 살린다(검토 P1)
                else:
                    filled = _accept(fill_items(data.get("claims") or [], sents, set(gaps), rejected),
                                     by_id, catalog, claims, 0, rejected, checks["repaired"])
                    claims.extend(filled)
                    checks["filled"] = len(filled)
                    checks["left"] = sentence_gaps(asked, data.get("sentences") or [], claims, rejected)
        rounds.append(len(claims))
        if round_no > 0 and not new:
            break
        if round_no == rounds_max and new:
            stopped = True
    claims[:] = drop_covered_thresholds(claims, by_id, rejected)
    for i, c in enumerate(claims, 1):
        c.id = f"C{i}"
    failure = "no_claims" if not claims and any(re.search(r"\d", p.text) for p in scope) else None
    return result(failure)


def claims_json(result: ExtractResult) -> str:
    return json.dumps([asdict(c) for c in result.claims], ensure_ascii=False)
