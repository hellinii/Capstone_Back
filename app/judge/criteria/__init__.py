"""app/judge/criteria — 심판 기준 고정 묶음(설계 7.4·7.5)과 그 지문.

이 폴더의 파일(프롬프트·체크리스트·규칙·모델 설정·이 로더)이 심판의 기준이다. `MANIFEST.sha256` 에 파일마다 sha256 을 적고,
그 MANIFEST 글 전체의 sha256 을 "기준 지문"으로 쓴다. 고정한 버전의 지문은 `VERSIONS.json` 에 덧붙이기만 한다.
시험(tests/test_judge_criteria_manifest.py)이 파일 ↔ MANIFEST, 버전 기록 ↔ 지금 지문을 맞댄다 — 몰래 바꾸면 빨개지고, MANIFEST 만
다시 써도 고정한 버전의 기록과 달라 빨개진다(버전을 올리고 다시 검증해야 한다).

MANIFEST 다시 쓰기: `python scripts/judge_criteria_manifest.py`(기준을 고친 커밋에서, 이유를 CHANGELOG 에 적고).

상호작용
- 의존(import): hashlib, json, pathlib
- 사용처: app.judge 의 모듈들(기준 읽기), scripts/judge_criteria_manifest.py, scripts/judge_live.py(장부의 기준 버전·지문)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 지문에 들지 않는 것: 지문 자신, 버전 기록, 변경 기록(설명 글이 바뀌어도 기준은 같다), 파이썬 캐시.
NOT_IN_MANIFEST = {"MANIFEST.sha256", "VERSIONS.json", "CHANGELOG.md"}


def bundle_files() -> list[Path]:
    """지문에 드는 파일(경로 순)."""
    return sorted(
        p for p in HERE.rglob("*")
        if p.is_file() and p.name not in NOT_IN_MANIFEST and "__pycache__" not in p.parts and not p.name.startswith(".")
    )


def manifest_text() -> str:
    """지금 파일들로 만든 MANIFEST 글(`<sha256>  <상대 경로>` 줄, 끝 줄바꿈)."""
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(HERE).as_posix()}" for p in bundle_files()]
    return "\n".join(lines) + "\n"


def recorded_manifest() -> str:
    return (HERE / "MANIFEST.sha256").read_text(encoding="utf-8")


def fingerprint(text: str | None = None) -> str:
    """기준 지문 = MANIFEST 글의 sha256. 기본은 적어 둔 MANIFEST."""
    return hashlib.sha256((recorded_manifest() if text is None else text).encode("utf-8")).hexdigest()


def version() -> str:
    return (HERE / "VERSION").read_text(encoding="utf-8").strip()


def recorded_versions() -> dict[str, str]:
    return json.loads((HERE / "VERSIONS.json").read_text(encoding="utf-8"))


def load_json(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def load_text(name: str) -> str:
    return (HERE / name).read_text(encoding="utf-8")


def version_problem(ver: str | None = None, recorded: dict | None = None, fp: str | None = None) -> str | None:
    """버전 기록과 지금 지문이 어긋나면 그 사유, 맞으면 None. 고정 전(-dev)은 기록에 없어야 한다."""
    ver = version() if ver is None else ver
    recorded = recorded_versions() if recorded is None else recorded
    fp = fingerprint() if fp is None else fp
    if ver.endswith("-dev"):
        return f"개발 중 버전 {ver} 이 고정 기록에 있다" if ver in recorded else None
    if ver not in recorded:
        return f"고정 버전 {ver} 의 지문 기록이 없다"
    if recorded[ver] != fp:
        return f"고정 버전 {ver} 의 기록된 지문과 지금 지문이 다르다 — 기준을 바꿨으면 VERSION 을 올리고 다시 검증한다(설계 7.4)"
    return None
