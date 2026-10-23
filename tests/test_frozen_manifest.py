"""test_frozen_manifest.py — 판 4 라운드 S0: 고정 정답 폴더의 파일 목록과 sha256 을 MANIFEST.sha256 과 대조한다.

tests/golden_frozen(README·MANIFEST 제외)과 tests/data/edition3_issued 의 모든 파일이 목록에 있고, 지문이 같아야 한다.
정답을 다시 뜨거나 입력 사본을 고치면 이 목록도 같이 바뀌어야 한다 — 몰래 바뀌는 것을 막는다(README 의 규칙).
"""
import hashlib
from pathlib import Path

import frozen_s0 as fz

# MANIFEST.sha256 에 실린 파일 수 — 고정 폴더 63(사본 12·평가 기준 4·입력 6(계약 4·합성 2)·판 3 정답 41) + 발급본 자료 12.
# 판 3 정답 41 = 작업 경로 11예 × 3(규칙 문장·지시문·사실 시트 보기) + 옛 경로 지시문 2 + 발급본 검문 6.
MANIFEST_COUNT = 75


def _manifest_files() -> dict[str, str]:
    tests = Path(__file__).parent
    files = [
        *(p for p in fz.FROZEN.rglob("*") if p.is_file() and p.name not in ("README.md", "MANIFEST.sha256")),
        *(p for p in fz.ISSUED.rglob("*") if p.is_file()),
    ]
    files = [p for p in files if not p.name.startswith(".")]  # .DS_Store 같은 숨은 파일은 보지 않는다
    return {p.relative_to(tests).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def test_manifest_lists_every_frozen_file_with_its_sha256():
    manifest_path = fz.FROZEN / "MANIFEST.sha256"
    assert manifest_path.is_file(), manifest_path
    listed = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    assert len(listed) == MANIFEST_COUNT
    assert _manifest_files() == listed
