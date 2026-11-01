"""scripts/judge_paths.py — 검증 도구들의 경로 비교 한 곳(심판 라운드 J10 검토 반영, 오염 방지 규칙 2·계획 §카-3).

macOS 파일 시스템은 대소문자를 가리지 않고(`검증` 과 `검증`(자모 분해, NFD)·`00_봉인` 을 같은 폴더로 연다) 링크도 따라간다.
그래서 글자 비교만으로 "검증 폴더 안인가·봉인 폴더인가"를 보면 다른 철자로 우회된다. 여기서는 경로를 resolve(링크·..)한 뒤
NFC 로 모으고 casefold 해서 비교하고, 두 경로가 실제로 있으면 samefile 로도 본다.

표준 라이브러리만 쓴다(judge_cost_report.py 가 앱을 import 하지 않는다).

상호작용
- 사용처: scripts/judge_validate.py(dev 허용 목록·검증 폴더 거부·결과 원본 경로), scripts/judge_cost_report.py(봉인 거부)
"""
from __future__ import annotations

import os
import unicodedata
from pathlib import Path


def key(path: Path | str) -> str:
    """비교용 글자 — resolve → NFC → casefold."""
    return unicodedata.normalize("NFC", str(Path(path).resolve())).casefold()


def _name_key(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def inside(path: Path | str, root: Path | str) -> bool:
    """path 가 root 자신이거나 그 안인가(철자·대소문자·자모 분해·링크 우회까지)."""
    p, r = key(path), key(root)
    if p == r or p.startswith(r.rstrip(os.sep) + os.sep):
        return True
    root_path = Path(root)
    if root_path.exists():
        here = Path(path).resolve()
        for q in (here, *here.parents):
            try:
                if q.exists() and os.path.samefile(q, root_path):
                    return True
            except OSError:
                continue
    return False


def has_part(path: Path | str, name: str) -> bool:
    """경로 마디 중 하나가 name 인가(resolve 한 경로와 준 경로 둘 다 — NFC·casefold 로)."""
    want = _name_key(name)
    for parts in (Path(path).parts, Path(path).resolve().parts):
        if any(_name_key(part) == want for part in parts):
            return True
    return False
