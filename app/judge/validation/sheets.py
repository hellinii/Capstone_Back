"""app/judge/validation/sheets.py — 사람 일치도 채점지·열쇠(계획 17-5, 설계 7.1·7.2, 계획 Q12). 순수 함수.

문항 단위(Q12 ③): 사실 일치 = 주장 하나, 필수 내용 = 항목 하나, 일관성 = 성적서 하나의 모순 유형 하나, 표기 = 칸 하나.
결함본·sham 판정도 문항에 넣고 층을 나눠 뽑는다(축 × 시스템 판정 문제 있음/없음 × 바탕/결함본/sham — "문제 있음" 사례 확보).

채점지는 블라인드다: 시스템 판정·결함 여부·변형 이름이 보이지 않는다. 성적서는 문서 번호(무작위 이름)로만 가리킨다.
결함본 문항은 7.2 역할대로 나눈다 — 사실 일치·일관성 결함은 민서가 심었으므로 인용 님이, 필수 내용·표기 결함은 인용 님이 심었으므로
민서가 채점한다. 사람 간 일치도용 겹침 문항(축마다 몇 개)은 결함본이 아닌 문항에서 뽑아 두 사람 채점지에 모두 넣는다.
열쇠(문항 → 회차·성적서·변형·시스템 판정·층)는 채점지와 따로 둔다.

상호작용
- 의존(import): app.judge.reading(칸 번호 표), app.judge.validation.variants(칸 경로), random
- 사용처: scripts/judge_validate.py sheets, 시험 tests/test_judge_validation_defects.py
"""
from __future__ import annotations

import random

from app.judge.reading import NARRATIVE_PATHS
from app.judge.validation.variants import AXES, scorer_for

PROBLEM_VERDICTS = {"violation", "inference_violation"}
QUESTION = {
    "faithfulness": "이 주장이 성적서의 데이터(표·값·고객 입력)와 맞지 않는가?",
    "coverage": "서술 전체(고정 문구 포함)가 이 내용을 독자에게 알려 주지 않는가?",
    "consistency": "서술 안에 이 유형의 모순(같은 것에 대해 두 가지 말)이 있는가?",
    "readability": "이 칸이 표기 규칙(해라체, 허용된 낱말 밖 로마자 금지, 읽히는 문장)을 어기는가?",
}
SHEET_COLUMNS = ("문항", "문서", "칸", "축", "묻는 것", "대상", "판정(예/아니오)", "메모")
KEY_COLUMNS = ("문항", "문서", "run_id", "성적서", "변형", "변형_축", "축", "단위", "시스템_문제", "층", "채점")
_PATH_TO_SLOT = {f"{g}.{n}": sid for sid, (g, n) in NARRATIVE_PATHS.items()}


def slot_of_path(path: str) -> str:
    """결함 칸 경로 → 칸 번호(7.1 … 9)."""
    if path.startswith(("recommendation_narrative.", "recommendations.")):
        return "9"
    return _PATH_TO_SLOT.get(path, "")


def items_from_run(run: dict, contradiction_types: list[dict]) -> list[dict]:
    """결과 원본 하나(judge_validate 회차) → 문항 후보. 채점이 실패한 회차는 문항이 없다."""
    rec = run.get("record") or {}
    if not run.get("scorecard"):
        return []
    key = run["key"]
    meta = run.get("variant_meta") or {}
    common = {"run_id": run["run_id"], "report": key["report"], "variant": key["variant"],
              "kind": "base" if key["variant"] == "base" else ("defect" if key["variant"].startswith("D") else "sham"),
              "variant_axis": meta.get("axis", ""), "variant_slot": slot_of_path(meta.get("slot", "")),
              "variant_round": meta.get("round", 1) if isinstance(meta.get("round", 1), int) else 1}
    out = []
    for v in (rec.get("faithfulness") or {}).get("verdicts", []):
        out.append({**common, "axis": "faithfulness", "unit": v["claim_id"], "slot": v["slot"], "target": v["quote"],
                    "problem": v["verdict"] in PROBLEM_VERDICTS})
    for i in (rec.get("coverage") or {}).get("items", []):
        out.append({**common, "axis": "coverage", "unit": i["id"], "slot": i.get("slot", ""), "target": i["id"],
                    "problem": not i["described"]})
    found = {c["type"] for c in (rec.get("consistency") or {}).get("contradictions", [])}
    for t in contradiction_types:
        out.append({**common, "axis": "consistency", "unit": t["id"], "slot": "", "target": f"{t['name']}: {t['what']}",
                    "problem": t["id"] in found})
    rea = rec.get("readability") or {}
    flagged = {f["slot"] for f in [*rea.get("findings", []), *rea.get("llm_flags", [])]}
    for slot in sorted({p["slot"] for p in rec.get("pieces", []) if p.get("source") in ("ai", "template")}):
        out.append({**common, "axis": "readability", "unit": slot, "slot": slot, "target": f"칸 {slot}",
                    "problem": slot in flagged})
    return out


def stratum(item: dict) -> str:
    return f"{item['axis']}|{'문제' if item['problem'] else '없음'}|{item['kind']}"


def _planted_item(item: dict) -> bool:
    """결함본·sham 에서 심은 자리의 문항 — 심은 축(sham 은 모든 축)이고 바꾼 칸이거나 칸이 없는 단위(일관성·필수 내용)."""
    if item["kind"] == "base":
        return False
    axis_ok = item["kind"] == "sham" or item["axis"] == item["variant_axis"]
    return axis_ok and (not item["slot"] or item["slot"] == item["variant_slot"])


def sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """층 나눠 뽑기. 축마다 n/4 개: 결함본·sham 의 심은 자리 문항을 먼저(축 몫의 절반까지), 나머지는 '문제'·'없음' 층을 번갈아."""
    rng = random.Random(seed)
    per_axis = {a: n // len(AXES) + (1 if i < n % len(AXES) else 0) for i, a in enumerate(AXES)}
    chosen: list[dict] = []
    for axis in AXES:
        pool = sorted((i for i in items if i["axis"] == axis), key=lambda i: (i["run_id"], i["unit"]))
        want = per_axis[axis]
        planted = [i for i in pool if _planted_item(i)]
        rng.shuffle(planted)
        planted.sort(key=lambda i: not i["problem"])  # 심은 자리에서 시스템이 문제라고 한 문항부터(탐지한 결함 사례)
        take = planted[: max(want // 2, 1) if planted else 0]
        rest = [i for i in pool if i not in take]
        groups = {True: [i for i in rest if i["problem"]], False: [i for i in rest if not i["problem"]]}
        for g in groups.values():
            rng.shuffle(g)
        flip = True
        while len(take) < want and (groups[True] or groups[False]):
            g = groups[flip] or groups[not flip]
            take.append(g.pop())
            flip = not flip
        chosen += take
    return chosen


def assign(sampled: list[dict], *, overlap_per_axis: int, seed: int) -> list[dict]:
    """채점자 나누기. 결함본 문항은 7.2 역할대로(심은 사람이 아닌 쪽), 겹침은 결함본이 아닌 문항에서 축마다."""
    rng = random.Random(seed + 1)
    out = []
    for item in sampled:
        scorer = scorer_for(item["variant_axis"], item["variant_round"]) if item["kind"] == "defect" else "인용"
        out.append({**item, "scorers": [scorer]})
    for axis in AXES:
        cand = [i for i in out if i["axis"] == axis and i["kind"] != "defect"]
        for item in rng.sample(cand, min(overlap_per_axis, len(cand))):
            item["scorers"] = ["인용", "민서"]
    return out


def build(assigned: list[dict], seed: int) -> tuple[dict[str, list[tuple]], list[tuple], dict[str, tuple[str, str]]]:
    """(채점지 {채점자: 줄}, 열쇠 줄, 문서 번호 {문서: (run_id, 성적서)}). 문항 순서·문서 번호는 시드로 섞는다."""
    rng = random.Random(seed + 2)
    runs = sorted({(i["run_id"], i["report"]) for i in assigned})
    labels = [f"문서-{n:02d}" for n in range(1, len(runs) + 1)]
    rng.shuffle(labels)
    doc_of = dict(zip(runs, labels))
    order = list(assigned)
    rng.shuffle(order)
    sheets: dict[str, list[tuple]] = {"인용": [], "민서": []}
    key = []
    for n, item in enumerate(order, start=1):
        qid = f"Q{n:03d}"
        doc = doc_of[(item["run_id"], item["report"])]
        for who in item["scorers"]:
            sheets[who].append((qid, doc, item["slot"] or "-", item["axis"], QUESTION[item["axis"]], item["target"], "", ""))
        key.append((qid, doc, item["run_id"], item["report"], item["variant"], item["variant_axis"], item["axis"], item["unit"],
                    "예" if item["problem"] else "아니오", stratum(item), "+".join(item["scorers"])))
    return sheets, key, {label: run for run, label in doc_of.items()}


def human_pairs(key_rows: list[dict], filled: dict[str, list[dict]]) -> dict:
    """채운 채점지(채점자 → 줄) + 열쇠 → 일치도 짝(축별).

    system_human[축] = [(시스템, 사람) …] — 각 채점자의 판정마다 한 짝(겹침 문항은 두 짝). human_human[축] = [(인용, 민서) …]
    — 두 사람이 모두 판정한 겹침 문항. 판정은 예/아니오, 빈칸은 짝에서 빼고 blank 로 센다. 그 밖의 글이나 열쇠에 없는 문항이면 ValueError.
    """
    verdict_col = SHEET_COLUMNS[6]
    key = {r["문항"]: r for r in key_rows}
    judged: dict[str, dict[str, str]] = {}
    blank = 0
    for who, rows in filled.items():
        for row in rows:
            qid = row["문항"]
            if qid not in key:
                raise ValueError(f"열쇠에 없는 문항: {qid} ({who})")
            v = (row.get(verdict_col) or "").strip()
            if v not in ("예", "아니오", ""):
                raise ValueError(f"판정은 예/아니오: {qid} ({who}) {v!r}")
            if not v:
                blank += 1
                continue
            judged.setdefault(qid, {})[who] = v
    system_human: dict[str, list] = {a: [] for a in AXES}
    human_human: dict[str, list] = {a: [] for a in AXES}
    for qid, by in sorted(judged.items()):
        axis, system = key[qid]["축"], key[qid]["시스템_문제"]
        for who in sorted(by):
            system_human[axis].append((system, by[who]))
        if len(by) == 2:
            human_human[axis].append((by["인용"], by["민서"]))
    return {"system_human": system_human, "human_human": human_human, "blank": blank}
