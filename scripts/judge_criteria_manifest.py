"""
judge_criteria_manifest.py — 심판 기준 묶음(app/judge/criteria)의 지문 도구.

하위 명령(레포 루트에서)
- (없음) 또는 write: MANIFEST.sha256 을 지금 파일로 다시 쓴다. 기준 파일을 고친 커밋에서 돌리고, 바꾼 이유를 criteria/CHANGELOG.md
  에 적는다. 고정한 버전(VERSIONS.json 에 기록된 VERSION)의 지문은 바뀌면 시험이 빨개진다 — 그때는 VERSION 을 올리고 다시
  검증한다(설계 7.4).
    python scripts/judge_criteria_manifest.py
- freeze <버전>: 기준을 고정한다(V1, 계획 17-1 — 예: freeze 1.0). VERSION 을 그 버전으로 바꾸고 MANIFEST 를 다시 쓴 뒤
  VERSIONS.json 끝에 "버전 → 지문" 한 줄을 덧붙인다. 아래면 거부한다(아무 파일도 바꾸지 않음):
    - 버전 이름이 "숫자.숫자" 꼴이 아니다(-dev 등)
    - 그 버전이 이미 VERSIONS.json 에 있다(기록은 덧붙이기만 — 다시 고정할 수 없다. 기준을 바꿨으면 다음 버전으로)
    - 지금 파일과 MANIFEST 가 다르다(기록하지 않은 변경 — 먼저 write 하고 CHANGELOG 에 이유를 적어 커밋)
    - CHANGELOG.md 에 "## <버전>" 절이 없다(고정 이유·날짜를 먼저 사람이 적는다)
  끝나면 시험(tests/test_judge_criteria_manifest.py KNOWN)에 덧붙일 줄과 태그 명령(정8)을 찍는다. 태그는 사람이 붙인다.
    python scripts/judge_criteria_manifest.py freeze 1.0
  --criteria-dir 로 다른 폴더(시험의 임시 사본)를 다룰 수 있다.

상호작용
- 의존(import): app.judge.criteria(지문 계산은 한 곳 — 여기서 다시 짜지 않는다)
- 사용처: 사람이 직접 실행. 시험은 tests/test_judge_criteria_manifest.py
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.judge import criteria  # noqa: E402

FROZEN_NAME = re.compile(r"\d+\.\d+")
TAG_PREFIX = "judge-criteria-v"


class FreezeError(RuntimeError):
    """고정을 거부할 때."""


@contextlib.contextmanager
def bundle_at(path: Path | None):
    """criteria 의 지문 함수가 다른 폴더(시험의 임시 사본)를 보게 잠시 바꾼다."""
    if path is None:
        yield criteria.HERE
        return
    old = criteria.HERE
    criteria.HERE = Path(path).resolve()
    try:
        yield criteria.HERE
    finally:
        criteria.HERE = old


def write_manifest(path: Path | None = None) -> str:
    with bundle_at(path) as here:
        text = criteria.manifest_text()
        (here / "MANIFEST.sha256").write_text(text, encoding="utf-8")
        return text


def freeze(version: str, path: Path | None = None) -> str:
    """기준 고정 — 지문(MANIFEST 해시)을 돌려준다. 거부 조건은 머리말."""
    with bundle_at(path) as here:
        if not FROZEN_NAME.fullmatch(version):
            raise FreezeError(f"고정 버전 이름은 '숫자.숫자' 꼴이다(받은 것: {version!r})")
        recorded = criteria.recorded_versions()
        if version in recorded:
            raise FreezeError(f"버전 {version} 은 이미 고정 기록에 있다 — 기록은 덧붙이기만 한다(기준을 바꿨으면 다음 버전으로)")
        if criteria.recorded_manifest() != criteria.manifest_text():
            raise FreezeError("지금 기준 파일과 MANIFEST 가 다르다 — 먼저 write 로 다시 쓰고 CHANGELOG 에 이유를 적어 커밋한다")
        if f"## {version}\n" not in (here / "CHANGELOG.md").read_text(encoding="utf-8"):
            raise FreezeError(f"CHANGELOG.md 에 '## {version}' 절이 없다 — 고정 날짜·이유를 먼저 적는다")
        (here / "VERSION").write_text(version + "\n", encoding="utf-8")
        text = criteria.manifest_text()
        (here / "MANIFEST.sha256").write_text(text, encoding="utf-8")
        fp = criteria.fingerprint(text)
        # 덧붙이기만 — 앞 기록의 순서·글자를 그대로 두고 끝에 한 줄.
        (here / "VERSIONS.json").write_text(json.dumps({**recorded, version: fp}, indent=1) + "\n", encoding="utf-8")
        return fp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="심판 기준 묶음 지문 도구")
    parser.add_argument("command", nargs="?", default="write", choices=["write", "freeze"])
    parser.add_argument("version", nargs="?")
    parser.add_argument("--criteria-dir")
    args = parser.parse_args(argv)
    path = Path(args.criteria_dir) if args.criteria_dir else None
    if args.command == "write":
        text = write_manifest(path)
        with bundle_at(path):
            print(f"MANIFEST {len(text.splitlines())}개 파일, 지문 {criteria.fingerprint(text)}, VERSION {criteria.version()}")
        return 0
    if not args.version:
        print("멈춤: freeze 에는 버전이 필요하다(예: freeze 1.0)", file=sys.stderr)
        return 2
    try:
        fp = freeze(args.version, path)
    except FreezeError as exc:
        print(f"멈춤: {exc}", file=sys.stderr)
        return 2
    print(f"고정: VERSION {args.version}, 지문 {fp}")
    print("다음(사람):")
    print(f'  1) tests/test_judge_criteria_manifest.py 의 KNOWN 에 덧붙인다:  "{args.version}": "{fp}",')
    print("  2) 전체 시험 → 커밋(VERSION·MANIFEST·VERSIONS.json·CHANGELOG·시험)")
    print(f'  3) git tag -a {TAG_PREFIX}{args.version} -m "심판 기준 {args.version} 고정 — 지문 {fp}"  (정8, 시각·지문을 라운드 기록에)')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
