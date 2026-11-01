"""app/judge/validation/sheets.py — 사람 일치도 채점지·열쇠(계획 17-5, 설계 7.1·7.2, 계획 Q12). 순수 함수.

문항 단위(Q12 ③): 사실 일치 = 주장 하나, 필수 내용 = 항목 하나, 일관성 = 성적서 하나의 모순 유형 하나, 표기 = 칸 하나.
결함본·sham 판정도 문항에 넣고 층을 나눠 뽑는다(축 × 시스템 판정 문제 있음/없음 × 바탕/결함본/sham — "문제 있음" 사례 확보).

채점지는 블라인드다: 시스템 판정·결함 여부·변형 이름이 보이지 않는다. 성적서는 문서 번호(무작위 이름)로만 가리킨다.
결함본 문항은 7.2 역할대로 나눈다 — 사실 일치·일관성 결함은 민서가 심었으므로 인용 님이, 필수 내용·표기 결함은 인용 님이 심었으므로
민서가 채점한다. 사람 간 일치도용 겹침 문항(모두 몇 개 — 축마다 고르게 나눔)은 결함본이 아닌 문항에서 뽑아 두 사람 채점지에
모두 넣는다. V1 결정 7(2026-10-29): 40문항(축마다 10), 그중 겹침 10문항(축마다 3·3·2·2).
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


def overlap_quota(overlap: int, available: dict[str, int]) -> dict[str, int]:
    """겹침 문항 수(모두)를 축마다 나눈다 — 고르게(앞 축부터 하나씩 더), 후보가 모자란 축의 몫은 후보가 남은 축으로 넘긴다."""
    quota = {a: 0 for a in AXES}
    left = max(overlap, 0)
    while left:
        moved = False
        for axis in AXES:
            if left and quota[axis] < available.get(axis, 0):
                quota[axis] += 1
                left -= 1
                moved = True
        if not moved:
            break
    return quota


def assign(sampled: list[dict], *, overlap: int, seed: int) -> list[dict]:
    """채점자 나누기. 결함본 문항은 7.2 역할대로(심은 사람이 아닌 쪽), 겹침(모두 overlap 문항)은 결함본이 아닌 문항에서
    축마다 고르게(overlap_quota)."""
    rng = random.Random(seed + 1)
    out = []
    for item in sampled:
        scorer = scorer_for(item["variant_axis"], item["variant_round"]) if item["kind"] == "defect" else "인용"
        out.append({**item, "scorers": [scorer]})
    cands = {axis: [i for i in out if i["axis"] == axis and i["kind"] != "defect"] for axis in AXES}
    quota = overlap_quota(overlap, {a: len(c) for a, c in cands.items()})
    for axis in AXES:
        for item in rng.sample(cands[axis], quota[axis]):
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


# ── 8절 블라인드 읽기 묶음(P1) — "심판 없음 = 사람이 직접 읽고 결함을 찾음" ─────────────────────────
# 읽는 사람은 그 결함을 심지 않은 사람이다(심은 사람은 답을 안다): 결함본은 심은 사람이 아닌 쪽에게만 간다. 한 사람에게는 **한 성적서의
# 한 판본만** 준다(J10 검토 — 바탕본과 결함본을 같이 읽으면 두 글을 맞대 차이를 찾는 일이 된다). 성적서마다 그 사람이 읽을 수 있는
# 결함본이 있으면 그중 하나, 없으면 그 사람이 심지 않은 sham, 그것도 없으면 바탕본을 준다. 이번 묶음에 자리가 없는 결함본은
# unplaced 로 돌려준다(다른 판본을 이미 받은 사람에게 줄 수 없다 — 효용성 표에서 '사람이 읽은 결함' 수로 따로 보인다).
# 읽는 사람은 문서마다 문제라고 본 문장을 글자 그대로 옮겨 적고 시간을 적는다.
READING_COLUMNS = ("문서", "문제 문장(글자 그대로)", "축(사실 일치/필수 내용/일관성/표기)", "메모")
READING_KEY_COLUMNS = ("문서", "성적서", "변형", "변형_축", "심은_문장", "항목", "읽는_사람")
READERS = ("인용", "민서")


def build_reading(bases: list[str], variants: dict[str, dict], seed: int,
                  ) -> tuple[dict[str, list[tuple]], list[tuple], list[str]]:
    """(읽기 목록 {읽는 사람: [(문서, 성적서, 변형)]}, 열쇠 줄, 자리가 없는 결함본). variants = {변형 id: meta}.
    사람마다 성적서 하나에 문서 하나. 문서 번호는 사람마다 따로 섞는다."""
    rng = random.Random(seed + 3)
    lists: dict[str, list[tuple]] = {}
    key: list[tuple] = []
    placed: set[str] = set()
    for reader in READERS:
        docs = []
        for tag in sorted(set(bases)):
            mine = sorted(v for v, m in variants.items() if m.get("base") == tag and m.get("planted_by") != reader)
            defects = [v for v in mine if v.startswith("D") and v not in placed]
            shams = [v for v in mine if v.startswith("S")]
            pick = rng.choice(defects) if defects else (rng.choice(shams) if shams else "base")
            placed.add(pick)
            docs.append((tag, pick))
        rng.shuffle(docs)
        labels = [f"읽기-{reader}-{n:02d}" for n in range(1, len(docs) + 1)]
        lists[reader] = [(label, tag, vid) for label, (tag, vid) in zip(labels, docs)]
        for label, tag, vid in lists[reader]:
            meta = variants.get(vid, {})
            key.append((label, tag, vid, meta.get("axis", ""), meta.get("after") or meta.get("before") or "",
                        meta.get("item_id", ""), reader))
    unplaced = sorted(v for v in variants if v.startswith("D") and v not in placed)
    return lists, key, unplaced


def reading_hits(key_rows: list[dict], filled: list[dict], touches) -> dict[str, dict]:
    """읽은 결과 → 결함본마다 사람이 심은 문장을 짚었나. touches = 인용·문장 겹침 판정(stats.touches 를 넘긴다).
    필수 내용(지운 문장)은 적은 글이 지운 문장을 짚었거나 메모에 항목 번호를 적었으면 찾은 것으로 본다. 결함 없는 문서의 지적은 false_alarms."""
    found_col = READING_COLUMNS[1]
    by_doc: dict[str, list[dict]] = {}
    for row in filled:
        by_doc.setdefault(row["문서"], []).append(row)
    out: dict[str, dict] = {}
    false_alarms = 0
    for k in key_rows:
        notes = [r for r in by_doc.get(k["문서"], []) if (r.get(found_col) or "").strip() or (r.get("메모") or "").strip()]
        if k["변형"] in ("base",) or k["변형"].startswith("S"):
            false_alarms += sum(1 for r in notes if (r.get(found_col) or "").strip())
            continue
        hit = any(touches(r.get(found_col) or "", k["심은_문장"]) for r in notes if k["심은_문장"])
        if not hit and k["항목"]:
            hit = any(k["항목"] in (r.get("메모") or "") + (r.get(found_col) or "") for r in notes)
        out[k["변형"]] = {"axis": k["변형_축"], "reader": k["읽는_사람"], "found": hit}
    return {"defects": out, "false_alarms": false_alarms}
