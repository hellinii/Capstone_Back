"""scripts/judge_paths.py(검증 도구의 경로 비교 한 곳, J10 검토) 시험 — 두 겹(글자 비교 · samefile)을 따로 본다.

두 겹이 서로를 가리지 않게: 글자 비교 겹은 root 가 없는 경로로(samefile 이 끼지 못하게), samefile 겹은 글자 비교를 끈 채로 본다.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("judge_paths", ROOT / "scripts" / "judge_paths.py")
P = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("judge_paths", P)
_SPEC.loader.exec_module(P)


def test_text_layer_catches_nfd_and_case_without_samefile(tmp_path):
    root = tmp_path / "없는" / "검증"  # 없는 폴더 — samefile 겹은 쓰이지 않는다
    assert not root.exists()
    child = root / "00_봉인" / "a.json"
    for spelled in (unicodedata.normalize("NFD", str(child)), str(child).upper(), str(root)):
        assert P.inside(spelled, root), spelled
    assert not P.inside(tmp_path / "없는" / "검증2" / "a", root)  # 앞부분만 같은 이름은 안이 아니다
    assert P.has_part(unicodedata.normalize("NFD", str(child)), "00_봉인")
    assert P.has_part(str(child).replace("00_봉인", "00_봉인".upper()), "00_봉인")
    assert not P.has_part(tmp_path / "00_봉인x" / "a", "00_봉인")


def _case_insensitive(folder: Path) -> bool:
    return Path(str(folder).upper()).exists() and str(folder).upper() != str(folder)


def test_samefile_layer_alone_catches_other_spelling(tmp_path, monkeypatch):
    root = tmp_path / "검증"
    (root / "00_봉인").mkdir(parents=True)
    calls = iter(range(10**6))
    monkeypatch.setattr(P, "key", lambda p: f"글자{next(calls)}")  # 글자 비교 겹을 끈다(늘 다르다고 답함)
    assert P.inside(root / "00_봉인", root)
    link = tmp_path / "링크"
    link.symlink_to(root, target_is_directory=True)
    assert P.inside(link / "00_봉인", root)
    if _case_insensitive(tmp_path):  # macOS 기본 파일 시스템 — 대소문자 다른 철자를 samefile 이 같은 폴더로 본다
        assert P.inside(Path(str(root / "00_봉인").upper()), root)
    assert not P.inside(tmp_path / "다른", root)
