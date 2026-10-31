"""app/judge/faithfulness.py — 사실 일치(설계 6.1, 계획 11-2). 4모듈 가운데 유일한 에이전트.

흐름(설계 6.1 에이전트 구조 Resolution → Lookup → Judgment → Synthesis)
1. 코드: 칸이 정해진 숫자 주장은 바로 조회해 숫자 대조 규칙(criteria/rules_numbers)으로 판정한다.
2. 에이전트(GPT-6.1 Sol, 도구 4개): 나머지 주장(칸 못 정한 숫자·비교·이름·정도·추론)의 대상을 찾아 칸 이름·식·관계로 답한다.
   도구를 몇 번 부를지는 실행 중에 정한다(최대 회차는 criteria/model.json faithfulness_max_turns).
3. 코드: 답한 식을 계산하고(criteria/rules_calc — eval 없음) 대조한다. 정도 표현은 criteria/rules_degree, 추론 표시 어미는
   criteria/claims.json inference_markers 로 본다. LLM 은 맞다·틀리다를 정하지 않는다.
받는 범위: AI·규칙 문장 조각만(reading.for_module "faithfulness"), 주장 목록(공용 전처리).

판정 종류: match · violation(데이터와 어긋남) · unverifiable(데이터 없음) · target_unclear(대상 불명) · inference_ok ·
inference_violation(단정형 추론) · not_claim(추론으로 뽑혔지만 용어 뜻풀이·일반 원리 설명 — 1.1). 점수 = match ÷ (match + violation). 게이트는 violation + inference_violation ≥ 1.

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
from app.judge.criteria.rules_numbers import bound_after, bound_holds, first_quantity, fraction_value, matches, parse_printed
from app.judge.extract import Claim
from app.judge.reading import Piece, for_module, render
from app.judge.sources import Sources

CALL_TIMEOUT_SECONDS = 180.0
RELATIONS = ("gt", "lt", "ge", "le", "eq", "")
# not_claim 은 추론 주장에만 받는다(1.1) — 용어 뜻풀이·일반 원리 설명이라 이 데이터에 대한 주장이 아님. 다른 종류면 검증 불가로 둔다.
RESOLUTIONS = ("resolved", "no_data", "target_unclear", "not_claim")
EQ_TOLERANCE = 0.0005 + 1e-9
BASES = ("data", "client_input", "other", "")
_FIELD = re.compile(r"\{([^{}]+)\}")
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
                "resolution": {"type": "string", "enum": list(RESOLUTIONS)},
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
    """대조할 찍힌 숫자 — 추출이 적은 숫자 글(분수 말 포함), 없으면 문장의 첫 수량(이름에 붙은 숫자는 건너뜀, criteria/rules_numbers)."""
    if claim.number_text and (parse_printed(claim.number_text) or fraction_value(claim.number_text) is not None):
        return claim.number_text
    return first_quantity(claim.quote)


def sentence_of(quote: str, piece: Piece | None) -> str:
    if piece is None:
        return quote
    norm = " ".join(quote.split())
    for s in _SENTENCE.split(piece.text):
        if norm in " ".join(s.split()):
            return s
    return quote


def inference_marked(sentence: str) -> bool:
    return any(m in sentence for m in criteria.claim_markers("inference_markers"))


def _relation_holds(a: float, b: float, rel: str) -> bool | None:
    # '같다'는 찍힌 자릿수(판 4 셋째 자리)로 같으면 참이다 — 0.48007 과 0.48 은 둘 다 0.480 으로 찍힌다(J4 파일럿).
    return {"gt": a > b, "lt": a < b, "ge": a >= b, "le": a <= b, "eq": abs(a - b) < EQ_TOLERANCE}.get(rel)


def _norm_name(text: str) -> str:
    return re.sub(r"[\s'\"‘’“”.,]", "", str(text)).lower()


def _occurrences(text: str, labels: dict[str, list[str]]) -> list[tuple[int, int, str]]:
    """글(정규화) 안의 이름 자리 (시작, 끝, 코드). 더 긴 이름 자리에 덮이는 짧은 자리는 뺀다('불합격' 안의 '합격')."""
    spans = []
    for code, labs in labels.items():
        for lab in labs:
            n = _norm_name(lab)
            start = text.find(n)
            while n and start >= 0:
                spans.append((start, start + len(n), code))
                start = text.find(n, start + 1)
    return [s for s in spans if not any(o[0] <= s[0] and s[1] <= o[1] and (o[1] - o[0]) > (s[1] - s[0]) for o in spans)]


def name_matches(actual: str, named: str) -> bool | None:
    """데이터의 이름(actual)이 문장이 지목한 이름(named)에 들어 있나. 모호하면 None(검증 불가).
    코드값(판정 FAIL·선언 fn 등)은 기준 묶음 names.json 의 찍히는 이름으로 보되, 실제 값이 든 묶음(code_groups)의 이름만 본다.
    - 같은 묶음의 서로 다른 코드가 둘 다 지목되면 None('FN 이 FP 보다 위험' — 1.1 검토, 예전엔 코드 글자 순서로 정해졌다).
    - 이름 바로 뒤에 부정이 오면 지목이 아니다. 실제 값을 부정하면 아니다('모든 클래스가 동등하지 않다')."""
    names = criteria.load_json("names.json")
    labels = names["code_labels"]
    target = _norm_name(named)
    if str(actual) in labels:
        group = next((g for g in names.get("code_groups", []) if str(actual) in g), list(labels))
        occ = _occurrences(target, {c: labels[c] for c in group if c in labels})
        if occ:
            window = int(names.get("negation_window", 6))
            neg_words = [_norm_name(w) for w in names.get("negation_after", [])]
            # 이름 자체가 부정형('제공되지 않')이면 뒤의 부정은 다음 말의 것이다(1.3)
            negated = lambda o: (not any(w in target[o[0]: o[1]] for w in neg_words)  # noqa: E731
                                 and any(w in target[o[1]: o[1] + window] for w in neg_words))
            if any(negated(o) and o[2] == str(actual) for o in occ):
                return False
            said = {o[2] for o in occ if not negated(o)}
            if len(said) > 1:
                return None
            if said:
                return said.pop() == str(actual)
            return None
    return bool(_norm_name(actual)) and _norm_name(actual) in target


_CLIENT_NOTE = re.compile(r"\s*\(의뢰자 제공:.*\)\s*$")
CLIENT_OVERLAP = 0.6


def _bigrams(t: str) -> set[str]:
    return {t[i:i + 2] for i in range(len(t) - 1)}


def client_text_matches(actual: str, named: str) -> bool | None:
    """고객 입력 칸은 다시 말한 글도 맞다(설계 6.1 "고객 입력을 다시 말하는 문장은 그 입력과 맞으면 통과").
    - 의뢰자 참고 글 '(의뢰자 제공: …)' 은 떼고 본다.
    - 한쪽이 다른 쪽에 들어 있거나(6자 이상 — 앞부분만 옮김, J4 2차 파일럿),
    - 두 글자 묶음이 어느 한쪽 기준으로 60% 이상 겹치면(어미만 바꿔 말함 — '쓴다'↔'쓰려는 것', J4 AI 파일럿) 맞다."""
    n = _norm_name(named)
    full = str(actual)
    # 선택지 값을 바꿔 말한 것(names.json client_option_labels — '같은 출처'↔'같은 조건', 1.1). 반대 값의 말이 같이 있으면 아니다.
    options = criteria.load_json("names.json").get("client_option_labels", {})
    choice = _CLIENT_NOTE.sub("", full).strip()
    if choice in options:
        said = lambda opt: any(_norm_name(lab) in n for lab in options[opt])  # noqa: E731
        other_said = any(said(other) for other in options if other != choice)
        if other_said:
            # 반대 값의 말이 있으면 뒤의 글 겹침 대조로 넘기지 않는다(1.1 검토 — '다른 곳이 아닌 같은 출처'): 두 값이 다 있으면 모호,
            # 반대 값만 있으면 아니다
            return None if said(choice) else False
        if said(choice):
            return True
    note = re.search(r"\(의뢰자 제공:(.*)\)\s*$", full)
    # 선택 값만·참고 글 붙은 전체·참고 글만 — 문장이 참고 글을 말할 수도 있다(J4 AI 파일럿)
    for cand in {_norm_name(_CLIENT_NOTE.sub("", full)), _norm_name(full), _norm_name(note.group(1)) if note else ""}:
        if not cand or not n:
            continue
        if cand in n or (len(n) >= 6 and n in cand):
            return True
        ba, bn = _bigrams(cand), _bigrams(n)
        if ba and bn:
            shared = len(ba & bn)
            if shared / len(ba) >= CLIENT_OVERLAP or shared / len(bn) >= CLIENT_OVERLAP:
                return True
    return False


def judge_answer(claim: Claim, ans: dict | None, sources: Sources, metrics, refs, piece: Piece | None) -> ClaimVerdict:
    """에이전트 답(없으면 None)과 자료로 주장 하나를 판정한다(코드)."""
    v = ClaimVerdict(claim.id, claim.kind, "unverifiable", claim.slot, claim.piece_id, claim.quote)
    if claim.kind == "inference":
        return _judge_inference(v, claim, ans, sources, piece)
    if ans is None:
        return v
    v.detail = {k: ans.get(k) for k in ("resolution", "expr", "expr_b", "relation", "name_field", "named", "metric_id")}
    if ans.get("resolution") == "target_unclear":
        v.verdict = "target_unclear"
        return v
    if ans.get("resolution") != "resolved":
        return v
    if claim.kind == "number" and _is_comparison(ans):
        # 숫자로 뽑혔지만 두 값의 관계를 말하는 문장('예측된 표본이 실제 표본보다 많이') — 에이전트가 두 식과 관계를 주면 비교로
        # 대조한다(1.1, V2 1차 — 숫자 하나로 대조해 이름 속 숫자 '클래스 3' 을 예측 건수와 맞댔다). 문장에 수량이 있으면 그 숫자도
        # 앞쪽 식 값과 맞아야 한다 — 답의 꼴만 바꿔 틀린 숫자가 통과하지 않게(1.1 검토).
        v.detail["as"] = "comparison"
        _compare(v, ans, sources)
        text = printed_number(claim)
        if text is not None and v.verdict == "match":
            value = eval_expr(ans.get("expr", ""), sources)
            v.detail["data"] = value
            if value is None or not matches(text, value):
                v.verdict = "violation" if value is not None else "unverifiable"
    elif claim.kind == "number":
        text, value = printed_number(claim), eval_expr(ans.get("expr", ""), sources)
        if text is None or value is None:
            return v
        v.detail["data"] = value
        ok = matches(text, value)
        rel = bound_after(claim.quote, text)
        if not ok and rel:
            # 경계선 숫자('0.1 이상') — 데이터가 그 경계를 지키면 맞다(1.4)
            ok = bool(bound_holds(text, rel, value))
            v.detail["bound"] = rel
        v.verdict = "match" if ok else "violation"
    elif claim.kind == "comparison":
        _compare(v, ans, sources)
    elif claim.kind == "name":
        _name(v, ans, sources)
    elif claim.kind == "degree":
        result = degree_verdict(claim.polarity, ans.get("metric_id", "").strip(), claim.quote, metrics, refs)
        v.verdict = {"ok": "match", "violation": "violation", "target_unclear": "target_unclear"}[result]
    return v


def _is_comparison(ans: dict) -> bool:
    return bool(ans.get("expr", "").strip() and ans.get("expr_b", "").strip() and ans.get("relation"))


def _compare(v: ClaimVerdict, ans: dict, sources: Sources) -> None:
    a, b = eval_expr(ans.get("expr", ""), sources), eval_expr(ans.get("expr_b", ""), sources)
    holds = None if a is None or b is None else _relation_holds(a, b, ans.get("relation", ""))
    if holds is not None:
        v.detail.update({"a": a, "b": b})
        v.verdict = "match" if holds else "violation"


def _name(v: ClaimVerdict, ans: dict, sources: Sources) -> None:
    actual = sources.texts.get(ans.get("name_field", "").strip())
    if actual is None or not ans.get("named"):
        return
    v.detail["data"] = actual
    if ans.get("name_field", "").strip().startswith("client."):
        ok = client_text_matches(actual, ans["named"])
        if ok is None:
            return
    else:
        ok = name_matches(actual, ans["named"])
        if ok is None:
            return
        positive = sources.texts.get("facts.positive_class")
        if not ok and positive is not None and str(actual) == positive:  # 양성 클래스를 '양성'으로 부름
            ok = any(lab in ans["named"] for lab in criteria.load_json("names.json")["positive_class_labels"])
    v.verdict = "match" if ok else "violation"


def is_definition(sentence: str) -> bool:
    """용어 뜻풀이 문장인가 — 뜻풀이 어미(inference_rules.json definition_markers)가 있고, 숫자가 없고, 앞 문장·데이터를 가리키는
    말(definition_not_after — 이는·이 결과는·그래서 등)로 시작하지 않는다('이는 모델이 취약함을 뜻한다'는 해석이다, 1.3)."""
    head = sentence.strip()
    if any(head.startswith(w) for w in criteria.claim_markers("definition_not_after")):
        return False
    return any(m in sentence for m in criteria.claim_markers("definition_markers")) and not re.search(r"\d", sentence)


def _judge_inference(v: ClaimVerdict, claim: Claim, ans: dict | None, sources: Sources, piece: Piece | None) -> ClaimVerdict:
    """추론 주장. 기본은 추론 표시 어미(코드)로 inference_ok / inference_violation.
    1.1(V2 1차 — 사실·판정·고객 입력을 옮긴 문장과 용어 뜻풀이가 단정형 추론으로 걸렸다):
    - 에이전트가 그 문장을 데이터 대조로 옮기면(두 식과 관계, 또는 이름 칸과 이름) 코드가 비교·이름 주장처럼 대조한다 — 맞으면
      match, 틀리면 violation. 대조할 수 없으면(칸 없음) 어미 판정으로 돌아간다. 맞다·틀리다는 여전히 코드가 정한다.
      문장에 인과 낱말(causal_markers — 때문·원인 등)이 있으면 옮기지 않는다 — 인과 부분은 대조가 보지 못한다(1.1 검토).
    - 에이전트가 not_claim(용어 뜻풀이)으로 답하면, 코드가 뜻풀이 어미가 있고 숫자·인과 낱말이 없음을 확인할 때만 주장이 아니다
      (not_claim — 분모·게이트 밖, 기록에 남는다). 아니면 어미 판정."""
    ans = ans or {}
    sentence = sentence_of(claim.quote, piece)
    marked = inference_marked(sentence)
    causal = any(m in sentence for m in criteria.claim_markers("causal_markers"))
    resolution = ans.get("resolution")
    # 에이전트가 고른 갈래도 남긴다 — 기록만 보고 다시 판정·검토할 수 있게(1.2)
    v.detail = {"marked": marked, "basis": ans.get("inference_basis", ""), "resolution": resolution or "", "causal": causal}
    if resolution == "not_claim":
        # 코드가 확인한다(1.1 검토): 뜻풀이 어미가 있고 숫자·인과 낱말이 없을 때만. 아니면 어미 판정으로.
        if is_definition(sentence) and not causal:
            v.verdict = "not_claim"
            return v
        v.detail["not_claim_refused"] = True
    elif causal and resolution == "resolved" and (_is_comparison(ans) or ans.get("name_field", "").strip()):
        v.detail["remap_refused"] = "causal"
    elif resolution == "resolved" and (_is_comparison(ans) or (ans.get("name_field", "").strip() and ans.get("named"))):
        v.detail.update({k: ans.get(k) for k in ("expr", "expr_b", "relation", "name_field", "named")})
        if _is_comparison(ans):
            v.detail["as"] = "comparison"
            _compare(v, ans, sources)
        else:
            v.detail["as"] = "name"
            _name(v, ans, sources)
        if v.verdict in ("match", "violation"):
            return v
    if not marked and not causal and is_definition(sentence) and not v.detail.get("not_claim_refused"):
        # 뜻풀이 문장은 에이전트 답과 상관없이 코드가 주장이 아님으로 정한다 — 같은 문장에 회차마다 답이 갈렸다(1.3, V2 1.2 2회차:
        # 1회차는 not_claim, 2회차는 ③). 에이전트가 데이터 대조로 옮겨 판정이 났으면 위에서 이미 돌아갔다.
        v.verdict = "not_claim"
        v.detail["by"] = "code"
        return v
    v.verdict = "inference_ok" if marked else "inference_violation"
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
        "inference_ok": by("inference_ok"), "inference_violation": by("inference_violation"), "not_claim": by("not_claim"),
        "from_inference": sum(1 for v in verdicts if v.kind == "inference" and v.verdict in ("match", "violation")),
        "client_input_assertive": sum(1 for v in verdicts if v.verdict == "inference_violation"
                                      and v.detail.get("basis") == "client_input"),
    }


async def _ask_agent(client, content: str, ask: list[Claim], instructions: str, purpose: str, sources: Sources,
                     scope: list[Piece], *, deadline: float | None, kw: dict) -> tuple[dict[str, dict], list[dict], int, str | None]:
    """에이전트(도구 4개)에게 주장 목록의 대상을 묻는다. (답, 호출 기록, 도구 호출 수, 실패 사유)."""
    max_turns = int(criteria.load_json("model.json")["faithfulness_max_turns"])
    convo: list = [{"role": "user", "content": content}]
    schema = llm.json_schema_format(purpose, answer_schema([c.id for c in ask]))
    calls: list[dict] = []
    tool_calls = 0
    for _turn in range(max_turns):
        out = await llm.create_response(
            client, purpose=purpose, model=config.judge_model(), instructions=instructions, input=convo,
            timeout_s=CALL_TIMEOUT_SECONDS, deadline=deadline, tools=TOOLS, text_format=schema, **kw,
        )
        calls.append(out.record.as_dict())
        if out.failure:
            return {}, calls, tool_calls, out.failure
        fcalls = llm.function_calls(out.response)
        if not fcalls:
            data, failure = llm.parse_json(out)
            if failure:
                return {}, calls, tool_calls, failure
            answers = {a["claim_id"]: a for a in data.get("claims") or [] if isinstance(a, dict) and "claim_id" in a}
            return answers, calls, tool_calls, None
        tool_calls += len(fcalls)
        convo = convo + llm.next_turn_items(out.response) + [
            {"type": "function_call_output", "call_id": fc.call_id,
             "output": json.dumps(_tool_output(fc.name, _safe_args(fc.arguments), sources, scope), ensure_ascii=False)}
            for fc in fcalls
        ]
    return {}, calls, tool_calls, "agent_turns"


def _listing(claims: list[Claim]) -> str:
    return "\n".join(f"- {c.id} [{c.kind}] (조각 {c.piece_id}) {c.quote}"
                     + (f" — 칸 후보 {c.field}" if c.field else "") + (f" / {c.field_b}" if c.field_b else "")
                     for c in claims)


def recheck_targets(verdicts: list[ClaimVerdict]) -> list[str]:
    """재확인할 주장 — 발급을 막는 단정형 추론 가운데 인과 문장이 아닌 것(1.5). 인과 문장은 데이터 대조로 옮기지 않는다."""
    return [v.claim_id for v in verdicts if v.verdict == "inference_violation" and not v.detail.get("causal")]


async def judge_faithfulness(client, pieces: list[Piece], claims: list[Claim], sources: Sources, metrics, refs, *,
                             deadline: float | None = None, clock=None) -> FaithResult:
    scope = for_module(pieces, "faithfulness")
    by_piece = {p.id: p for p in scope}
    kw = {"clock": clock} if clock else {}
    direct: dict[str, ClaimVerdict] = {}
    for c in claims:
        # 추출이 붙인 칸은 '가리키기'일 뿐 판정 근거가 아니다 — 그 칸 값과 맞으면 여기서 확정하고, 어긋나면 에이전트가 대상을
        # 다시 찾게 한다(J4 파일럿: 추출이 F1 최대 임계값의 Recall 을 Recall 값 칸에 붙여 거짓 위반이 났다).
        if c.kind == "number" and c.field and sources.number(c.field) is not None and printed_number(c):
            value = sources.number(c.field)
            if matches(printed_number(c), value):
                direct[c.id] = ClaimVerdict(c.id, c.kind, "match", c.slot, c.piece_id, c.quote,
                                            {"field": c.field, "data": value, "by": "code"})
    pending = [c for c in claims if c.id not in direct]
    answers: dict[str, dict] = {}
    calls: list[dict] = []
    tool_calls = 0
    if pending:
        answers, calls, tool_calls, failure = await _ask_agent(
            client, f"## 서술\n{render(scope)}\n\n## 조사할 주장\n{_listing(pending)}", pending,
            criteria.load_text("prompts/faithfulness.md"), "faithfulness", sources, scope, deadline=deadline, kw=kw)
        if failure:
            return FaithResult([], {}, calls, tool_calls, failure)
    verdicts = [direct.get(c.id) or judge_answer(c, answers.get(c.id), sources, metrics, refs, by_piece.get(c.piece_id))
                for c in claims]
    # 재확인(1.5, 민서 결정 2026-10-31 "서술이 맞으면 통과, 틀리거나 확인할 수 없으면 막는다"): 첫 답에서 발급을 막게 된 단정형
    # 추론만 그 조각과 함께 다시 묻는다. 같은 문장에 회차마다 답이 갈렸다(V2 1.1~1.4). 다시 물어 데이터 대조를 받으면 코드가
    # 맞다·틀리다를 정하고, 대조를 못 받으면 그대로 막는다. 재확인 호출이 실패하면 첫 판정(막음)을 그대로 둔다.
    targets = recheck_targets(verdicts)
    if targets:
        ask = [c for c in claims if c.id in targets]
        near = [p for p in scope if p.id in {c.piece_id for c in ask}]
        again, more_calls, more_tools, failure = await _ask_agent(
            client, f"## 조각\n{render(near)}\n\n## 다시 볼 주장\n{_listing(ask)}", ask,
            criteria.load_text("prompts/faithfulness_recheck.md"), "faithfulness_recheck", sources, scope,
            deadline=deadline, kw=kw)
        calls += more_calls
        tool_calls += more_tools
        by_id = {c.id: c for c in ask}
        for i, v in enumerate(verdicts):
            if v.claim_id not in by_id:
                continue
            if failure:
                v.detail["recheck"] = {"failure": failure}
                continue
            second = judge_answer(by_id[v.claim_id], again.get(v.claim_id), sources, metrics, refs, by_piece.get(v.piece_id))
            second.detail["recheck"] = {"first": v.verdict, "first_detail": v.detail}
            verdicts[i] = second
    counts = _count(verdicts)
    counts["rechecked"] = len(targets)
    return FaithResult(verdicts, counts, calls, tool_calls)


def _safe_args(arguments: str) -> dict:
    try:
        data = json.loads(arguments)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}
