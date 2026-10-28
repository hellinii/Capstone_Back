"""
judge_criteria_manifest.py — 심판 기준 묶음(app/judge/criteria)의 MANIFEST.sha256 을 지금 파일로 다시 쓴다.

기준 파일을 고친 커밋에서 돌리고, 바꾼 이유를 criteria/CHANGELOG.md 에 적는다. 고정한 버전(VERSIONS.json 에 기록된 VERSION)의
지문은 바뀌면 시험이 빨개진다 — 그때는 VERSION 을 올리고 다시 검증한다(설계 7.4). 레포 루트에서: python scripts/judge_criteria_manifest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.judge import criteria  # noqa: E402

text = criteria.manifest_text()
(criteria.HERE / "MANIFEST.sha256").write_text(text, encoding="utf-8")
print(f"MANIFEST {len(text.splitlines())}개 파일, 지문 {criteria.fingerprint(text)}, VERSION {criteria.version()}")
