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
META_FIELDS = ("id", "base", "axis", "kind", "planted_by", "slot", "intent")
# 설계 7.2(결정): 사실 일치·일관성 결함은 민서가 심고 인용 님이 채점, 필수 내용·표기 결함은 인용 님이 심고 민서가 채점.
PLANTER = {"faithfulness": "민서", "consistency": "민서", "coverage": "인용", "readability": "인용"}
SCORER = {"faithfulness": "인용", "consistency": "인용", "coverage": "민서", "readability": "민서"}
VARIANT_ID = re.compile(r"(D|S)\d{2}")
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
