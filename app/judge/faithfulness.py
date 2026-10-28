"""app/judge/faithfulness.py — 사실 일치(설계 6.1, 계획 11-2). 4모듈 가운데 유일한 에이전트.

흐름(설계 6.1 에이전트 구조 Resolution → Lookup → Judgment → Synthesis)
1. 코드: 칸이 정해진 숫자 주장은 바로 조회해 숫자 대조 규칙(criteria/rules_numbers)으로 판정한다.
2. 에이전트(GPT-6.1 Sol, 도구 4개): 나머지 주장(칸 못 정한 숫자·비교·이름·정도·추론)의 대상을 찾아 칸 이름·식·관계로 답한다.
   도구를 몇 번 부를지는 실행 중에 정한다(최대 회차는 criteria/model.json faithfulness_max_turns).
3. 코드: 답한 식을 계산하고(criteria/rules_calc — eval 없음) 대조한다. 정도 표현은 criteria/rules_degree, 추론 표시 어미는
   criteria/claims.json inference_markers 로 본다. LLM 은 맞다·틀리다를 정하지 않는다.
받는 범위: AI·규칙 문장 조각만(reading.for_module "faithfulness"), 주장 목록(공용 전처리).

판정 종류: match · violation(데이터와 어긋남) · unverifiable(데이터 없음) · target_unclear(대상 불명) · inference_ok ·
inference_violation(단정형 추론). 점수 = match ÷ (match + violation). 게이트는 violation + inference_violation ≥ 1.

상호작용
- 의존(import): app.judge.llm·config·criteria(rules_*)·reading·sources·extract(Claim)
- 사용처: app.judge.core(J4)
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from app.judge import config, criteria, llm
from app.judge.criteria.rules_calc import CalcError, calculate
from app.judge.criteria.rules_degree import degree_verdict
from app.judge.criteria.rules_numbers import matches, parse_printed
from app.judge.extract import Claim
from app.judge.reading import Piece, for_module, render
from app.judge.sources import Sources

CALL_TIMEOUT_SECONDS = 180.0
RELATIONS = ("gt", "lt", "ge", "le", "eq", "")
BASES = ("data", "client_input", "other", "")
_FIELD = re.compile(r"\{([^{}]+)\}")
_NUMBER_IN_TEXT = re.compile(r"[+\-−]?\d[\d,]*(?:\.\d+)?\s*(?:%p|%)?")
_SENTENCE = re.compile(r"(?<=[.?!])\s+|\n+")

TOOLS = [
    {"type": "function", "name": "lookup_value", "strict": True, "description": "칸 이름의 값과 출처",
     "parameters": {"type": "object", "properties": {"field": {"type": "string"}}, "required": ["field"], "additionalProperties": False}},
    {"type": "function", "name": "find_fields", "strict": True, "description": "이름이나 글에 query 가 든 칸 이름 목록",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}},
    {"type": "function", "name": "read_context", "strict": True, "description": "조각 글과 같은 칸의 앞뒤 조각 글",
     "parameters": {"type": "object", "properties": {"piece_id": {"type": "string"}}, "required": ["piece_id"], "additionalProperties": False}},
    {"type": "function", "name": "calc", "strict": True, "description": "사칙연산 식(숫자만)의 값",
     "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"], "additionalProperties": False}},
]


@dataclass
class ClaimVerdict:
    claim_id: str
    kind: str
    verdict: str
    slot: str
    piece_id: str
    quote: str
    detail: dict = field(default_factory=dict)


@dataclass
class FaithResult:
    verdicts: list[ClaimVerdict]
    counts: dict
    calls: list[dict] = field(default_factory=list)
    tool_calls: int = 0
    failure: str | None = None

    def violations(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.verdict in ("violation", "inference_violation")]

    def as_record(self) -> dict:
        return {"verdicts": [asdict(v) for v in self.verdicts], "counts": self.counts, "calls": self.calls,
                "tool_calls": self.tool_calls, "failure": self.failure}


def answer_schema(claim_ids: list[str]) -> dict:
    s = {"type": "string"}
    return {
        "type": "object", "additionalProperties": False, "required": ["claims"],
        "properties": {"claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["claim_id", "resolution", "expr", "expr_b", "relation", "name_field", "named", "metric_id",
                         "inference_basis"],
            "properties": {
                "claim_id": {"type": "string", "enum": claim_ids},
                "resolution": {"type": "string", "enum": ["resolved", "no_data", "target_unclear"]},
                "expr": s, "expr_b": s,
                "relation": {"type": "string", "enum": list(RELATIONS)},
                "name_field": s, "named": s, "metric_id": s,
                "inference_basis": {"type": "string", "enum": list(BASES)},
            },
        }}},
    }


def eval_expr(expr: str, sources: Sources) -> float | None:
    """{칸} 을 값으로 바꿔 계산. 칸이 없거나 숫자가 아니면 None."""
    if not expr.strip():
        return None
    missing = False

    def sub(m):
        nonlocal missing
        v = sources.number(m.group(1).strip())
        if v is None:
            missing = True
            return "0"
        return repr(float(v))

    text = _FIELD.sub(sub, expr)
    if missing:
        return None
    try:
        return calculate(text)
    except CalcError:
        return None


def printed_number(claim: Claim) -> str | None:
    if claim.number_text and parse_printed(claim.number_text):
        return claim.number_text
    m = _NUMBER_IN_TEXT.search(claim.quote)
    return m.group(0).strip() if m and parse_printed(m.group(0).strip()) else None


def sentence_of(quote: str, piece: Piece | None) -> str:
    if piece is None:
        return quote
    norm = " ".join(quote.split())
    for s in _SENTENCE.split(piece.text):
        if norm in " ".join(s.split()):
            return s
    return quote


def inference_marked(sentence: str) -> bool:
    markers = criteria.load_json("claims.json")["inference_markers"]
    return any(m in sentence for m in markers)


def _relation_holds(a: float, b: float, rel: str) -> bool | None:
    return {"gt": a > b, "lt": a < b, "ge": a >= b, "le": a <= b, "eq": abs(a - b) < 1e-9}.get(rel)


def _norm_name(text: str) -> str:
    return re.sub(r"[\s'\"‘’“”]", "", str(text)).lower()


def judge_answer(claim: Claim, ans: dict | None, sources: Sources, metrics, refs, piece: Piece | None) -> ClaimVerdict:
    """에이전트 답(없으면 None)과 자료로 주장 하나를 판정한다(코드)."""
    v = ClaimVerdict(claim.id, claim.kind, "unverifiable", claim.slot, claim.piece_id, claim.quote)
    if claim.kind == "inference":
        marked = inference_marked(sentence_of(claim.quote, piece))
        v.verdict = "inference_ok" if marked else "inference_violation"
        v.detail = {"marked": marked, "basis": (ans or {}).get("inference_basis", "")}
        return v
    if ans is None:
        return v
    v.detail = {k: ans.get(k) for k in ("resolution", "expr", "expr_b", "relation", "name_field", "named", "metric_id")}
    if ans.get("resolution") == "target_unclear":
        v.verdict = "target_unclear"
        return v
    if ans.get("resolution") != "resolved":
        return v
    if claim.kind == "number":
        text, value = printed_number(claim), eval_expr(ans.get("expr", ""), sources)
        if text is None or value is None:
            return v
        v.detail["data"] = value
        v.verdict = "match" if matches(text, value) else "violation"
    elif claim.kind == "comparison":
        a, b = eval_expr(ans.get("expr", ""), sources), eval_expr(ans.get("expr_b", ""), sources)
        holds = None if a is None or b is None else _relation_holds(a, b, ans.get("relation", ""))
        if holds is not None:
            v.detail.update({"a": a, "b": b})
            v.verdict = "match" if holds else "violation"
    elif claim.kind == "name":
        actual = sources.texts.get(ans.get("name_field", "").strip())
        if actual is not None and ans.get("named"):
            v.detail["data"] = actual
            v.verdict = "match" if _norm_name(actual) in _norm_name(ans["named"]) else "violation"
    elif claim.kind == "degree":
        result = degree_verdict(claim.polarity, ans.get("metric_id", "").strip(), claim.quote, metrics, refs)
        v.verdict = {"ok": "match", "violation": "violation", "target_unclear": "target_unclear"}[result]
    return v


def _tool_output(name: str, args: dict, sources: Sources, pieces: list[Piece]) -> dict:
    if name == "lookup_value":
        return sources.lookup(str(args.get("field", "")))
    if name == "find_fields":
        return {"fields": sources.find(str(args.get("query", "")))}
    if name == "read_context":
        pid = str(args.get("piece_id", ""))
        idx = next((i for i, p in enumerate(pieces) if p.id == pid), None)
        if idx is None:
            return {"found": False}
        same = [p for p in pieces if p.slot == pieces[idx].slot]
        return {"pieces": [{"id": p.id, "text": p.text} for p in same]}
    if name == "calc":
        try:
            return {"value": calculate(str(args.get("expr", "")))}
        except CalcError as exc:
            return {"error": str(exc)}
    return {"error": f"모르는 도구 {name}"}


def _count(verdicts: list[ClaimVerdict]) -> dict:
    by = lambda k: sum(1 for v in verdicts if v.verdict == k)  # noqa: E731
    return {
        "claims": len(verdicts), "checked": by("match") + by("violation"), "matched": by("match"), "violations": by("violation"),
        "unverifiable": by("unverifiable"), "target_unclear": by("target_unclear"),
        "inference_ok": by("inference_ok"), "inference_violation": by("inference_violation"),
        "client_input_assertive": sum(1 for v in verdicts if v.verdict == "inference_violation"
                                      and v.detail.get("basis") == "client_input"),
    }


async def judge_faithfulness(client, pieces: list[Piece], claims: list[Claim], sources: Sources, metrics, refs, *,
                             deadline: float | None = None, clock=None) -> FaithResult:
    scope = for_module(pieces, "faithfulness")
    by_piece = {p.id: p for p in scope}
    kw = {"clock": clock} if clock else {}
    direct: dict[str, ClaimVerdict] = {}
    for c in claims:
        if c.kind == "number" and c.field and sources.number(c.field) is not None and printed_number(c):
            value = sources.number(c.field)
            direct[c.id] = ClaimVerdict(c.id, c.kind, "match" if matches(printed_number(c), value) else "violation",
                                        c.slot, c.piece_id, c.quote, {"field": c.field, "data": value, "by": "code"})
    pending = [c for c in claims if c.id not in direct]
    answers: dict[str, dict] = {}
    calls: list[dict] = []
    tool_calls = 0
    if pending:
        max_turns = int(criteria.load_json("model.json")["faithfulness_max_turns"])
        listing = "\n".join(f"- {c.id} [{c.kind}] (조각 {c.piece_id}) {c.quote}"
                            + (f" — 칸 후보 {c.field}" if c.field else "") + (f" / {c.field_b}" if c.field_b else "")
                            for c in pending)
        convo: list = [{"role": "user", "content": f"## 서술\n{render(scope)}\n\n## 조사할 주장\n{listing}"}]
        schema = llm.json_schema_format("faithfulness", answer_schema([c.id for c in pending]))
        finished = False
        for _turn in range(max_turns):
            out = await llm.create_response(
                client, purpose="faithfulness", model=config.judge_model(),
                instructions=criteria.load_text("prompts/faithfulness.md"), input=convo, timeout_s=CALL_TIMEOUT_SECONDS,
                deadline=deadline, tools=TOOLS, text_format=schema, **kw,
            )
            calls.append(out.record.as_dict())
            if out.failure:
                return FaithResult([], {}, calls, tool_calls, out.failure)
            fcalls = llm.function_calls(out.response)
            if not fcalls:
                data, failure = llm.parse_json(out)
                if failure:
                    return FaithResult([], {}, calls, tool_calls, failure)
                answers = {a["claim_id"]: a for a in data.get("claims") or [] if isinstance(a, dict) and "claim_id" in a}
                finished = True
                break
            tool_calls += len(fcalls)
            convo = convo + llm.next_turn_items(out.response) + [
                {"type": "function_call_output", "call_id": fc.call_id,
                 "output": json.dumps(_tool_output(fc.name, _safe_args(fc.arguments), sources, scope), ensure_ascii=False)}
                for fc in fcalls
            ]
        if not finished:
            return FaithResult([], {}, calls, tool_calls, "agent_turns")
    verdicts = [direct.get(c.id) or judge_answer(c, answers.get(c.id), sources, metrics, refs, by_piece.get(c.piece_id))
                for c in claims]
    return FaithResult(verdicts, _count(verdicts), calls, tool_calls)


def _safe_args(arguments: str) -> dict:
    try:
        data = json.loads(arguments)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}
