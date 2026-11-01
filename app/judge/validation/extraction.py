"""app/judge/validation/extraction.py — 주장 추출 재현율·정밀도(계획 17-4, 설계 7.1·7.6).

사람 정답지 주장(claims.csv + sentences.csv, annotation.py 꼴)과 시스템 추출 주장(extract.ExtractResult 의 claims)을 짝짓는다.

짝이 될 수 있는 조건(다섯 — 결과를 보기 전에 정한 규칙, 카-4):
  ① 같은 칸(slot)
  ② 원문 위치 겹침: 같은 문장 번호. 두 쪽 모두 문장 안 글자 범위를 알면 그 범위도 겹쳐야 한다
  ③ 숫자·비교는 종류 같음(V1 결정 10③): 어느 한쪽이 number 나 comparison 주장이면 두 쪽 종류가 같아야 한다.
     number 는 더해 글에서 읽은 숫자 값이 하나 이상 같아야 한다. 이름·정도·추론끼리는 종류가 달라도 짝이 될 수 있다(덧점만 다름)
  ④ 대상 같음: 두 쪽 모두 target 이 있으면 같아야 한다. 정답지 claims.csv 에는 대상 열을 두지 않는다(V1 결정 10②) — 정답 쪽
     target 이 늘 빈칸이라 이 조건은 돌지 않고, 짝은 ①②③⑤ 로만 정해진다
  ⑤ 일대일: 짝 수가 가장 많은 짝짓기(헝가리안, scipy linear_sum_assignment). 짝 수가 같으면 같은 종류·더 많이 겹침을 고른다
재현율 = 짝 수 / 정답지 주장 수, 정밀도 = 짝 수 / 추출 주장 수. 사람이 "예"로 판정한 놓친 주장을 보탠 정답지로 다시 잰 값도
report 가 함께 낸다(V1 결정 10④ — 원 정답지 파일은 그대로).
0·1·2번 시점 = 첫 추출(round 0)까지 / 자기 점검 1번(round ≤ 1)까지 / 2번(round ≤ 2)까지의 누적 주장(extract.py 의 round).
시스템만 찾은 주장은 사람 판정용 표(MISSED_COLUMNS)로 내보낸다(Q12②) — "예" 로 판정한 것만 보탠 목록으로 정답지에 더한다.

상호작용
- 의존(import): scipy·numpy, app.judge.validation.annotation(표 열 이름), app.judge.validation.stats(touches)
- 사용처: scripts/judge_validate.py(검증 실행기, 17-3)
"""
from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from statistics import median

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.judge.validation.annotation import CLAIM_COLUMNS, SENTENCE_COLUMNS
from app.judge.validation.stats import touches

ROUNDS = (0, 1, 2)
# ③ 종류가 같아야 짝인 종류. V1 결정 10③(2026-10-29)은 ("number", "comparison") 이었다. 민서 결정(2026-10-31, 결과를 본 뒤):
# 같은 문장의 같은 부분을 짚었는데 종류만 다르면 놓친 것이 아니라 이름표를 다르게 단 것이다 → 빈 목록. 둘 다 숫자 주장이면 값은
# 같아야 한다(아래 compatible). 바꾸기 전 값은 검증 README 기록에 남긴다.
STRICT_KINDS: tuple[str, ...] = ()
MISSED_COLUMNS = ("번호", "문장번호", "칸", "종류", "인용 구절", "진짜 주장인가(예/아니오)", "메모")
ADDED_MEMO = "시스템 추출에서 보탬(Q12②)"
_SPACE = re.compile(r"\s+")
# 낱말에 붙은 숫자('F1'·'Class9')는 값으로 읽지 않는다(faithfulness 와 같은 생각)
_NUMBER = re.compile(r"(?<![A-Za-z0-9.])[+\-−]?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])")


def _norm(text) -> str:
    return _SPACE.sub(" ", str(text or "")).strip()


def numbers(text: str) -> set[float]:
    """글에서 읽은 숫자 값들(쉼표·유니코드 빼기 정리)."""
    out = set()
    for tok in _NUMBER.findall(str(text or "")):
        out.add(float(tok.replace(",", "").replace("−", "-")))
    return out


# ---------------------------------------------------------------- 표 → 짝짓기용 dict

def _sentence_table(sentence_rows) -> dict[int, dict]:
    table = {}
    for row in sentence_rows:
        r = dict(zip(SENTENCE_COLUMNS, row)) if not isinstance(row, dict) else row
        table[int(r["문장번호"])] = {"no": int(r["문장번호"]), "piece": r["조각"], "slot": r["칸"], "text": _norm(r["문장"])}
    return table


def _span(quote: str, sentence: str) -> list[int] | None:
    """인용이 문장에 나오는 글자 범위. 숫자로 시작·끝나는 인용은 다른 숫자의 일부에서 찾지 않는다('171건' 안의 '1건' —
    2026-10-31 세션 7, 짧은 숫자 인용이 앞 문장에 붙어 맞게 뽑은 주장이 놓친 것으로 셌다)."""
    q = _norm(quote)
    if not q:
        return None
    i = sentence.find(q)
    while i >= 0:
        before = sentence[i - 1] if i > 0 else ""
        after = sentence[i + len(q)] if i + len(q) < len(sentence) else ""
        if not ((q[0].isdigit() and (before.isdigit() or before == ".")) or
                (q[-1].isdigit() and (after.isdigit() or after == "."))):
            return [i, i + len(q)]
        i = sentence.find(q, i + 1)
    return None


def gold_from_csv(claim_rows: list[dict], sentence_rows) -> list[dict]:
    """정답지(claims.csv 줄들) → 짝짓기용 dict. 없는 문장 번호면 ValueError(표시 실수를 조용히 넘기지 않는다)."""
    table = _sentence_table(sentence_rows)
    out = []
    for i, row in enumerate(claim_rows, 1):
        no = int(row["문장번호"])
        if no not in table:
            raise ValueError(f"정답지 {i}번째 줄: 없는 문장 번호 {no}")
        s = table[no]
        quote = row.get("인용 구절") or ""
        out.append({"id": f"G{i}", "slot": s["slot"], "piece": s["piece"], "sentence": no, "span": _span(quote, s["text"]),
                    "kind": (row.get("종류") or "").strip(), "quote": quote, "number_text": "",
                    "target": (row.get("대상") or "").strip(), "added": bool(row.get("added"))})
    return out


def _candidates(piece: str, quote: str, table: dict[int, dict]) -> dict[int, list[int] | None]:
    """인용이 나오는 문장 후보 전부(문장 번호 → 범위). 같은 인용이 여러 문장에 나오면('1건') 정답지 문장과 맞는 후보로 짝을
    짓는다(2026-10-31 세션 7 — 처음 나온 문장 하나로 정해 다른 문장의 주장과 어긋났다). 문장 경계를 넘은 인용은 처음 걸친
    문장 하나(범위 모름)."""
    rows = [s for s in table.values() if s["piece"] == piece]
    found = {s["no"]: span for s in rows if (span := _span(quote, s["text"]))}
    if found:
        return found
    for s in rows:
        if touches(quote, s["text"]):
            return {s["no"]: None}
    return {}


def _locate(piece: str, quote: str, table: dict[int, dict]) -> tuple[int | None, list[int] | None]:
    cands = _candidates(piece, quote, table)
    if not cands:
        return None, None
    no = min(cands)
    return no, cands[no]


def pred_from_extract(claims: list, sentence_rows) -> list[dict]:
    """추출 주장(Claim 또는 as_record()["claims"] 의 dict) → 짝짓기용 dict. 문장을 못 찾으면 sentence None(짝 없음)."""
    table = _sentence_table(sentence_rows)
    out = []
    for i, c in enumerate(claims, 1):
        c = asdict(c) if is_dataclass(c) else dict(c)
        cands = _candidates(c.get("piece_id", ""), c.get("quote", ""), table)
        no = min(cands) if cands else None
        span = cands.get(no) if cands else None
        out.append({"id": c.get("id") or f"P{i}", "slot": c.get("slot", ""), "piece": c.get("piece_id", ""), "sentence": no,
                    "candidates": cands,
                    "span": span, "kind": c.get("kind", ""), "quote": c.get("quote", ""),
                    "number_text": c.get("number_text", ""), "target": c.get("target", ""), "round": int(c.get("round", 0))})
    return out


# ---------------------------------------------------------------- 짝짓기

def _values(claim: dict) -> set[float]:
    return numbers(claim.get("number_text") or claim.get("quote") or "")


def _overlap(g: dict, p: dict) -> float | None:
    """글자 범위가 겹치는 비율(0~1). 범위를 모르면 None, 안 겹치면 0."""
    a, b = g.get("span"), p.get("span")
    if not a or not b:
        return None
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def compatible(g: dict, p: dict, *, same_kind: bool = False) -> bool:
    if g.get("slot") != p.get("slot"):  # ① 같은 칸
        return False
    if g.get("sentence") is None:
        return False
    cands = p.get("candidates") or ({p["sentence"]: p.get("span")} if p.get("sentence") is not None else {})
    if g["sentence"] not in cands:  # ② 같은 문장(추출 인용이 여러 문장에 나오면 그 후보 가운데)
        return False
    p = {**p, "sentence": g["sentence"], "span": cands[g["sentence"]]}
    if _overlap(g, p) == 0.0:  # ② 글자 범위를 알면 겹쳐야
        return False
    if (g.get("kind") in STRICT_KINDS or p.get("kind") in STRICT_KINDS) and g.get("kind") != p.get("kind"):  # ③ 종류
        return False
    if g.get("kind") == "number" and p.get("kind") == "number" and not (_values(g) & _values(p)):  # 숫자끼리는 값도 같음
        return False
    if g.get("target") and p.get("target") and g["target"] != p["target"]:  # ④ 대상 같음
        return False
    if same_kind and g.get("kind") != p.get("kind"):
        return False
    return True


def match_claims(gold: list[dict], predicted: list[dict], *, same_kind: bool = False) -> dict:
    """일대일 최적 짝짓기(⑤). 돌려주는 값: pairs [(정답 dict, 추출 dict)], unmatched_gold, unmatched_pred, n_gold, n_pred."""
    out = {"pairs": [], "unmatched_gold": list(gold), "unmatched_pred": list(predicted), "n_gold": len(gold),
           "n_pred": len(predicted)}
    if not gold or not predicted:
        return out
    n = min(len(gold), len(predicted))
    weight = np.zeros((len(gold), len(predicted)))
    for i, g in enumerate(gold):
        for j, p in enumerate(predicted):
            if compatible(g, p, same_kind=same_kind):
                # 짝 하나 = 1, 덧점(같은 종류·겹침)은 모두 합쳐도 1 보다 작다 → 짝 수가 먼저다
                cands = p.get("candidates") or {}
                q = {**p, "span": cands.get(g["sentence"], p.get("span"))}
                bonus = 0.5 * (g.get("kind") == p.get("kind")) + 0.5 * (_overlap(g, q) or 0.0)
                weight[i, j] = 1.0 + bonus / (n + 1)
    rows, cols = linear_sum_assignment(weight, maximize=True)
    pairs = [(i, j) for i, j in zip(rows, cols) if weight[i, j] > 0]
    used_g, used_p = {i for i, _ in pairs}, {j for _, j in pairs}
    out["pairs"] = [(gold[i], predicted[j]) for i, j in pairs]
    out["unmatched_gold"] = [g for i, g in enumerate(gold) if i not in used_g]
    out["unmatched_pred"] = [p for j, p in enumerate(predicted) if j not in used_p]
    return out


def recall_precision(match: dict) -> dict:
    """재현율·정밀도. 분모가 0 이면 None."""
    k, ng, npred = len(match["pairs"]), match["n_gold"], match["n_pred"]
    return {"recall": k / ng if ng else None, "precision": k / npred if npred else None,
            "n_gold": ng, "n_pred": npred, "n_pairs": k}


def by_round(gold: list[dict], predicted: list[dict], rounds=ROUNDS, **kw) -> dict[int, dict]:
    """0·1·2번 시점 값 — 시점 k 까지 나온(round ≤ k) 추출 주장으로 잰 재현율·정밀도."""
    return {k: recall_precision(match_claims(gold, [p for p in predicted if p.get("round", 0) <= k], **kw)) for k in rounds}


def distribution(values: list[float | None]) -> dict:
    """여러 회차 분포(최소·중앙·최대). None 은 빼고 센다."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"min": None, "median": None, "max": None, "n": 0}
    return {"min": min(vals), "median": median(vals), "max": max(vals), "n": len(vals)}


def runs_distribution(gold: list[dict], runs: list[list[dict]], rounds=ROUNDS, **kw) -> dict:
    """같은 서술을 여러 번 추출한 결과들 → 재현율·정밀도의 회차 분포(최종과 시점별)."""
    if not runs:
        raise ValueError("회차가 없다")
    per = [by_round(gold, preds, rounds, **kw) for preds in runs]
    last = rounds[-1]
    return {
        "recall": distribution([r[last]["recall"] for r in per]),
        "precision": distribution([r[last]["precision"] for r in per]),
        "by_round": {k: {"recall": distribution([r[k]["recall"] for r in per]),
                         "precision": distribution([r[k]["precision"] for r in per])} for k in rounds},
        "runs": per,
    }


# ---------------------------------------------------------------- 사람이 놓친 주장(Q12②)

def missed_rows(match: dict) -> list[tuple]:
    """시스템만 찾은 주장 → 사람 판정용 표 줄(MISSED_COLUMNS). 판정 칸은 비워 둔다. 문장을 못 찾은 주장은 문장번호가 빈칸."""
    rows = []
    for i, p in enumerate(match["unmatched_pred"], 1):
        rows.append((i, "" if p.get("sentence") is None else p["sentence"], p.get("slot", ""), p.get("kind", ""),
                     p.get("quote", ""), "", ""))
    return rows


def confirmed_additions(judged_rows: list[dict]) -> list[dict]:
    """사람이 "예"(진짜 주장)로 판정한 줄 → 정답지 claims.csv 꼴(CLAIM_COLUMNS) 줄. 보탠 목록은 따로 파일로 남긴다."""
    verdict = MISSED_COLUMNS[5]
    out = []
    for row in judged_rows:
        v = (row.get(verdict) or "").strip()
        if v not in ("예", "아니오", ""):
            raise ValueError(f"판정은 예/아니오: {v!r}")
        if v == "예":
            if not str(row.get("문장번호") or "").strip():
                raise ValueError("문장번호가 없는 줄은 보탤 수 없다 — 사람이 문장번호를 적는다")
            out.append(dict(zip(CLAIM_COLUMNS, (row["문장번호"], row["종류"], row["인용 구절"], ADDED_MEMO))) | {"added": True})
    return out
