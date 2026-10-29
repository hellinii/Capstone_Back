"""app/judge/validation/variants.py — 결함본·sham(변형) 파일 꼴과 적용(계획 17-5, 설계 7.2). 순수 함수.

변형 하나 = 폴더 하나(`02_결함/D01/`, `S01/` …):
- meta.json : 바탕 성적서 태그(base)·축(axis)·결함 종류(kind)·심은 사람(planted_by)·바꾼 칸(slot)·의도(intent)
- slots.json: {바꾼 칸: 그 칸의 바뀐 글 전체} — 칸 하나만
- diff.txt  : 바탕 글 ↔ 바뀐 글의 문장 단위 차이(도구가 만든다 — make_diff)

칸(slot)은 서술 결과(작업 결과의 result) 안 글 칸의 점 경로다: interpretation.confusion_analysis, conclusion.narrative,
recommendation_narrative.data_quality, recommendations.0.action 처럼. 심판이 읽는 서술 칸만 받는다(app.judge.reading 의 표와 같다).

상호작용
- 의존(import): app.judge.reading(서술 칸 표), difflib, json, re
- 사용처: scripts/judge_validate.py
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from app.judge.reading import NARRATIVE_PATHS, RECOMMEND_PATHS

AXES = ("faithfulness", "coverage", "consistency", "readability")
SHAM = "sham"
META_FIELDS = ("id", "base", "axis", "kind", "planted_by", "slot", "intent", "before", "after")
# before = 바탕 칸에서 바꾸거나 지운 문장(글자 그대로, 끼워 넣기면 ""), after = 바꾸거나 끼운 문장(지우기면 "").
# 설계 7.2(결정): 사실 일치·일관성 결함은 민서가 심고 인용 님이 채점, 필수 내용·표기 결함은 인용 님이 심고 민서가 채점.
PLANTER = {"faithfulness": "민서", "consistency": "민서", "coverage": "인용", "readability": "인용"}
SCORER = {"faithfulness": "인용", "consistency": "인용", "coverage": "민서", "readability": "민서"}
# 다시 잴 때(7.7, Q12 ④ — meta 의 "round" 가 2 이상): 모듈을 고친 사람(민서)이 아닌 인용 님이 모든 결함을 심고 민서가 채점한다.
RETRY_PLANTER, RETRY_SCORER = "인용", "민서"
VARIANT_ID = re.compile(r"(D|S)\d{2}")


def planter_for(axis: str, round_no: int = 1) -> str:
    return RETRY_PLANTER if round_no >= 2 else PLANTER[axis]


def scorer_for(axis: str, round_no: int = 1) -> str:
    return RETRY_SCORER if round_no >= 2 else SCORER[axis]
REC_FIELDS = ("action", "expected_impact")
# 문장 나누기 — 사람 정답지 표시지(annotation)와 같은 규칙.
_SPLIT = re.compile(r"(?<=[.?!])\s+|\n+")


def slot_paths(result: dict) -> list[str]:
    """서술 결과(dict)에서 결함을 심을 수 있는 글 칸 경로 전부."""
    paths = [f"{g}.{n}" for g, n in NARRATIVE_PATHS.values()]
    paths += [f"recommendation_narrative.{n}" for _, n in RECOMMEND_PATHS]
    paths += [f"recommendations.{i}.{f}" for i, _ in enumerate(result.get("recommendations") or []) for f in REC_FIELDS]
    return paths


def _walk(result: dict, path: str):
    parts = path.split(".")
    node = result
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node[p]
    return node, (int(parts[-1]) if isinstance(node, list) else parts[-1])


def read_slot(result: dict, path: str) -> str:
    if path not in slot_paths(result):
        raise KeyError(f"서술 칸이 아니다: {path}")
    node, last = _walk(result, path)
    return node[last] or ""


def apply_slots(result: dict, slots: dict[str, str]) -> dict:
    """바탕 서술 결과에 바뀐 칸을 끼운 사본(바탕은 그대로)."""
    out = json.loads(json.dumps(result))
    for path, text in slots.items():
        read_slot(out, path)  # 칸 이름 확인
        node, last = _walk(out, path)
        node[last] = text
    return out


def sentences(text: str) -> list[str]:
    return [s for s in (x.strip() for x in _SPLIT.split(text or "")) if s]


def make_diff(base_text: str, new_text: str, slot: str) -> str:
    """문장 한 줄씩의 차이(unified). diff.txt 의 정본 — 사람이 손으로 쓰지 않는다."""
    lines = difflib.unified_diff([s + "\n" for s in sentences(base_text)], [s + "\n" for s in sentences(new_text)],
                                 fromfile=f"바탕/{slot}", tofile=f"변형/{slot}", n=1)
    return "".join(lines)


def load_variant(folder: Path) -> tuple[dict, dict[str, str]]:
    """(meta, slots). 꼴 검사는 check_variant 가 한다."""
    meta = json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))
    slots = json.loads((Path(folder) / "slots.json").read_text(encoding="utf-8"))
    return meta, slots


def variant_folders(root: Path, prefix: str) -> list[Path]:
    """D·S 로 시작하는 변형 폴더(이름 순)."""
    return sorted(p for p in Path(root).iterdir() if p.is_dir() and VARIANT_ID.fullmatch(p.name) and p.name[0] == prefix)


def sentence_change(base_text: str, new_text: str) -> tuple[str, list[str], list[str]] | None:
    """문장 단위로 바뀐 곳이 정확히 한 곳이고 문장 하나(바꿈 1→1·지움 1·끼움 1)면 (종류, 지운 문장, 넣은 문장). 아니면 None."""
    a, b = sentences(base_text), sentences(new_text)
    ops = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if len(ops) != 1:
        return None
    tag, i1, i2, j1, j2 = ops[0]
    if (i2 - i1, j2 - j1) not in ((1, 1), (1, 0), (0, 1)):
        return None
    return tag, a[i1:i2], b[j1:j2]


def _round(meta: dict) -> int:
    value = meta.get("round", 1)
    return value if isinstance(value, int) and value >= 1 else 1


def check_variant(folder_name: str, meta: dict, slots: dict, diff_text: str | None, base_result: dict, *,
                  check_diff: bool = True) -> list[str]:
    """결함본·sham 하나의 꼴 검사 — 문제 문장 목록(빈 목록 = 통과). 문장에 서술 글을 옮기지 않는다(화면에 찍히므로)."""
    problems = [f"meta.json 에 '{f}' 칸이 없다" for f in META_FIELDS if f not in meta]
    problems += [f"meta.json 의 '{f}' 가 비었다" for f in META_FIELDS[:7] if f in meta and not str(meta[f]).strip()]
    if problems:
        return problems
    if not VARIANT_ID.fullmatch(folder_name) or meta["id"] != folder_name:
        return [f"폴더 이름과 meta.id 가 다르거나 꼴(D01·S01)이 아니다: {folder_name} / {meta['id']}"]
    if folder_name.startswith("D"):
        if meta["axis"] not in AXES:
            problems.append(f"결함본의 축이 네 축 중 하나가 아니다: {meta['axis']}")
        elif meta["planted_by"] != planter_for(meta["axis"], _round(meta)):
            problems.append(f"{meta['axis']} 결함은 {planter_for(meta['axis'], _round(meta))} 이 심는다(설계 7.2 역할"
                            f"{', 다시 잼' if _round(meta) >= 2 else ''}) — 적힌 것: {meta['planted_by']}")
    elif meta["axis"] != SHAM:
        problems.append(f"sham 의 축은 '{SHAM}' 이다: {meta['axis']}")
    if meta["axis"] == "coverage" and not str(meta.get("item_id") or "").strip():
        problems.append("필수 내용 결함은 meta.item_id(지운 체크리스트 항목, 예: 분석-7)가 있어야 탐지를 셀 수 있다")
    if meta["planted_by"] not in ("민서", "인용"):
        problems.append(f"심은 사람은 민서·인용 중 하나다: {meta['planted_by']}")
    if list(slots) != [meta["slot"]]:
        return problems + [f"slots.json 은 meta.slot 칸 하나만 담는다(담긴 칸 {len(slots)}개)"]
    try:
        base_text = read_slot(base_result, meta["slot"])
    except (KeyError, IndexError, ValueError):
        return problems + [f"바탕 성적서에 없는 서술 칸이다: {meta['slot']}"]
    new_text = slots[meta["slot"]]
    if not isinstance(new_text, str):
        return problems + ["slots.json 의 값은 칸 글 전체(문자열)다"]
    change = sentence_change(base_text, new_text)
    if change is None:
        problems.append("바탕 칸 글에서 문장 하나만 바뀐 것이 아니다(바뀐 곳이 없거나 둘 이상)")
    else:
        _, removed, added = change
        if removed != ([meta["before"]] if meta["before"] else []):
            problems.append("meta.before 가 바탕 칸에서 바뀐(지운) 문장과 글자 그대로 같지 않다")
        if added != ([meta["after"]] if meta["after"] else []):
            problems.append("meta.after 가 새로 들어간 문장과 글자 그대로 같지 않다")
    if not check_diff:
        pass
    elif diff_text is None:
        problems.append("diff.txt 가 없다")
    elif diff_text != make_diff(base_text, new_text, meta["slot"]):
        problems.append("diff.txt 가 다시 만든 차이와 다르다(도구로 다시 쓴다)")
    return problems


def replace_sentence(text: str, index: int, new: str | None) -> str:
    """칸 글의 index 번째 문장을 new 로(None 이면 지움). 합성 결함(dev)·시험용 — 나머지 문장은 공백 하나로 잇는다."""
    parts = sentences(text)
    parts[index: index + 1] = [] if new is None else [new]
    return " ".join(parts)


def write_variant(folder: Path, meta: dict, base_result: dict, new_text: str) -> Path:
    """변형 폴더 하나를 쓴다(meta.json·slots.json·diff.txt). 꼴 검사는 하지 않는다 — check_variant 로 따로."""
    target = Path(folder) / meta["id"]
    target.mkdir(parents=True, exist_ok=True)
    (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (target / "slots.json").write_text(json.dumps({meta["slot"]: new_text}, ensure_ascii=False, indent=1) + "\n",
                                       encoding="utf-8")
    (target / "diff.txt").write_text(make_diff(read_slot(base_result, meta["slot"]), new_text, meta["slot"]), encoding="utf-8")
    return target
