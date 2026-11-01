"""app/judge/readability.py — 표기(설계 6.4, 계획 11-5). 코드 규칙 2 + 단일 호출.

받는 범위: AI 칸·규칙 문장 조각만(reading.for_module "readability"). 데이터(사실 시트)는 주지 않는다.
- 로마자 노출(코드): 허용 목록 밖 로마자 토큰(필드명·코드값·지표 번호 'M4' 등)이 나오면 위반, 나올 때마다 한 건.
  허용 = 지표 표시 이름·클래스 이름의 낱말, TP·FP·FN·TN, 고객 입력의 영어 낱말(모델 이름 포함). 허용 목록은 요청에서 만든다 —
  이름 목록일 뿐 값은 보지 않는다.
- 서술체(코드): 서술 칸 문장이 해라체로 끝나지 않거나 줄 앞에 글머리 기호가 있으면 문장마다 한 건. 권고표 칸에는 걸지 않는다.
- 어법·호응(LLM, GPT-6.1 Sol): 칸마다 이해하기 어려운 문장 있다/없다 + 인용(criteria/readability.json grammar_decision).
  "있다"에 인용이 없거나 글에 없으면 그 칸만 한 번 다시 묻고, 그래도 없으면 실패(uncited — 보류).
게이트가 아니다(설계 5.4). 부록에는 코드 규칙 위반 건수만.

상호작용
- 의존(import): app.judge.llm·config·criteria·reading
- 사용처: app.judge.core
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app.judge import config, criteria, llm
from app.judge.reading import Piece, for_module, render

CALL_TIMEOUT_SECONDS = 180.0
_LATIN = re.compile(r"(?<![%A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:[._\-][A-Za-z0-9]+)*")  # '10%p' 의 p 는 단위
_SENTENCE = re.compile(r"(?<=[.?!])\s+|\n+")
_TRAIL = " .?!)\"'’”」』]"


@dataclass
class Finding:
    rule: str
    piece_id: str
    slot: str
    text: str


@dataclass
class ReadabilityResult:
    findings: list[Finding]
    llm_flags: list[Finding]
    counts: dict
    calls: list[dict] = field(default_factory=list)
    failure: str | None = None

    def as_record(self) -> dict:
        return {"findings": [asdict(f) for f in self.findings], "llm_flags": [asdict(f) for f in self.llm_flags],
                "counts": self.counts, "calls": self.calls, "failure": self.failure}


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def allowed_latin(fs, client_inputs, source_notes: list[str] | None = None) -> set[str]:
    rules = criteria.load_json("readability.json")
    allowed = {t for r in rules["code_rules"] if r["id"] == "romanization" for t in r["always_allowed"]}
    names = [m.display_name for m in fs.metrics] + [p.label for p in fs.per_class] + [c.label for c in fs.confusion_counts]
    names += list((fs.distribution.class_distribution if fs.distribution else {}) or {})
    names += _strings(client_inputs or {})
    # 성적서에 찍히는 문제 번호(P1 — 6절 권고표 '대응 문제 번호')와 참조 기준 출처 글(기준치 v1)도 성적서 낱말이다(J4 파일럿)
    names += [p.id for p in getattr(fs, "problems", None) or []]
    names += list(source_notes or [])
    for n in names:
        allowed.update(_LATIN.findall(str(n)))
    return allowed


def romanization(pieces: list[Piece], allowed: set[str]) -> list[Finding]:
    return [Finding("romanization", p.id, p.slot, tok) for p in pieces for tok in _LATIN.findall(p.text) if tok not in allowed]


def _drop_tail_parenthetical(s: str) -> str:
    """문장 끝 괄호 설명을 걷는다 — '…이다(미달 지표: Recall(0.547, 목표 0.900)).' 의 끝은 '이다'(J4 파일럿)."""
    keep = " .?!\"'’”"  # 닫는 괄호는 남겨 둬야 걷을 괄호를 찾는다
    s = s.rstrip(keep)
    while s.endswith(")"):
        depth = 0
        for i in range(len(s) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(s[i], 0)
            if depth == 0:
                s = s[:i].rstrip(keep)
                break
        else:
            break
    return s.rstrip(_TRAIL)


def _hapnida(core: str) -> bool:
    """'…ㅂ니다'(합니다·입니다·됩니다) — '니다' 앞 글자의 받침이 ㅂ. '아니다'는 아니다."""
    if not core.endswith("니다") or len(core) < 3:
        return False
    code = ord(core[-3]) - 0xAC00
    return 0 <= code < 11172 and code % 28 == 17


def sentence_style(pieces: list[Piece]) -> list[Finding]:
    rule = next(r for r in criteria.load_json("readability.json")["code_rules"] if r["id"] == "sentence_style")
    endings, bullets = tuple(rule["endings_disallowed"]), tuple(rule["bullets"])
    found: list[Finding] = []
    for p in pieces:
        if not p.id.startswith("slot:"):
            continue
        for line in p.text.splitlines():
            if line.strip().startswith(bullets):
                found.append(Finding("sentence_style", p.id, p.slot, line.strip()[:120]))
        for s in (x.strip() for x in _SENTENCE.split(p.text)):
            if not s or s.startswith(bullets):
                continue
            core = _drop_tail_parenthetical(s.rstrip(" ."))
            if core.endswith(endings) or _hapnida(core) or not core.endswith("다"):
                found.append(Finding("sentence_style", p.id, p.slot, s[:120]))
    return found


def answer_schema(slots: list[str], piece_ids: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": ["slots"],
        "properties": {"slots": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["slot", "present", "quote", "piece_id"],
            "properties": {"slot": {"type": "string", "enum": slots}, "present": {"type": "boolean"},
                           "quote": {"type": "string"}, "piece_id": {"type": "string", "enum": [*piece_ids, ""]}},
        }}},
    }


def _norm(t: str) -> str:
    return " ".join(t.split())


def _cited(a: dict, by_id: dict[str, Piece]) -> bool:
    p = by_id.get(a.get("piece_id", ""))
    return p is not None and bool(a.get("quote", "").strip()) and _norm(a["quote"]) in _norm(p.text)


async def judge_readability(client, pieces: list[Piece], fs, client_inputs, *, source_notes: list[str] | None = None,
                            deadline: float | None = None, clock=None) -> ReadabilityResult:
    scope = for_module(pieces, "readability")
    by_id = {p.id: p for p in scope}
    findings = romanization(scope, allowed_latin(fs, client_inputs, source_notes)) + sentence_style(scope)
    slots = list(dict.fromkeys(p.slot for p in scope))
    counts = {"rules": 2, "violations": len(findings),
              "by_rule": {r: sum(1 for f in findings if f.rule == r) for r in ("romanization", "sentence_style")}}
    if not slots:
        return ReadabilityResult(findings, [], {**counts, "llm_flags": 0})
    schema = llm.json_schema_format("readability", answer_schema(slots, list(by_id)))
    kw = {"clock": clock} if clock else {}
    answers: dict[str, dict] = {}
    calls: list[dict] = []
    ask = slots
    for attempt in (1, 2):
        extra = "" if attempt == 1 else "\n\n앞 답에서 아래 칸은 '있다'고 했지만 인용이 없거나 글에 없었다. 이 칸만 다시 답하라."
        out = await llm.create_response(
            client, purpose=f"readability_{attempt}", model=config.judge_model(),
            instructions=criteria.load_text("prompts/readability.md").format(slots=", ".join(ask)) + extra,
            input=[{"role": "user", "content": render(scope)}], timeout_s=CALL_TIMEOUT_SECONDS, deadline=deadline,
            text_format=schema, **kw,
        )
        data, failure = llm.parse_json(out)
        calls.append(out.record.as_dict())
        if failure:
            return ReadabilityResult(findings, [], counts, calls, failure)
        for a in data.get("slots") or []:
            if isinstance(a, dict) and a.get("slot") in ask:
                answers[a["slot"]] = a
        if attempt == 1 and any(s not in answers for s in slots):
            return ReadabilityResult(findings, [], counts, calls, "bad_output")
        ask = [s for s in slots if answers[s].get("present") and not _cited(answers[s], by_id)]
        if not ask:
            break
    else:
        return ReadabilityResult(findings, [], counts, calls, "uncited")
    flags = [Finding("llm_hard_to_read", a["piece_id"], s, a["quote"]) for s, a in answers.items() if a.get("present")]
    return ReadabilityResult(findings, flags, {**counts, "llm_flags": len(flags)}, calls)
