"""app/judge/criteria_lock.py — 검증 기록 잠금(심판 라운드 J9, 계획 14-10·정9·§카-2).

"배포되는 기준 = 검증한 기준"을 코드로 지킨다. 검증(V2)을 마친 기준의 지문을 `app/judge/validated.json` 에 적고(J11 이 쓴다),
판 4 작업은 지금 기준 묶음 파일로 계산한 지문이 그 기록과 같을 때만 심판 통과로 나갈 수 있다. 기록이 없거나·읽을 수 없거나·
버전이나 지문이 다르면 심판을 부르지 않고 발급 보류(judge_error, 다시 만들기 없음)다 — 다시 만들어도 같은 기준이라 풀리지 않는다.

기록 파일은 기준 묶음 폴더(criteria/) 밖에 둔다 — 안에 두면 기록을 쓰는 순간 지문이 바뀐다.

    {"version": "1.0", "fingerprint": "<MANIFEST 글의 sha256>", "tag": "judge-criteria-v1.0", "validated_at": "...", "note": "..."}

판정에 쓰는 것은 version·fingerprint 둘뿐이다(나머지는 사람이 읽는 기록). 지문은 적어 둔 MANIFEST 가 아니라 **지금 파일로 다시
만든** MANIFEST 의 sha256 이다 — 서버에서 기준 파일 하나가 몰래 바뀌어도 잡힌다. 개발 중 버전(-dev)은 기록이 있어도 잠긴다.

닫힌 쪽 기본값: 잠금은 늘 켜져 있다. 끄는 길은 환경변수 `JUDGE_CRITERIA_LOCK=off` 하나뿐이고(글자 그대로, 대소문자·앞뒤 공백만
너그럽게), 개발 서버(scripts/dev_fake_llm_server.py)와 pytest 장치(tests/conftest.py)만 쓴다. 운영 환경에 두면 안 된다 — 꺼진
상태는 작업 내부 기록(judge.criteria_lock = "off")과 서버 로그에 남는다.

상호작용
- 의존(import): app.judge.criteria(version·manifest_text·fingerprint)
- 사용처: app.narrative.jobs(_judge_branch — 판 4 심판 갈래의 한 곳), tests
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from app.judge import criteria

RECORD_PATH = Path(__file__).resolve().parent / "validated.json"
ENV = "JUDGE_CRITERIA_LOCK"

# 잠금 사유(내부 기록·로그용 — 고객에게는 보류 종류 judge_error 로만 보인다).
UNVALIDATED = "unvalidated"  # 기록 없음 — 아직 검증 전
UNREADABLE = "record_unreadable"  # 기록이 JSON 이 아니거나 모양이 틀림
DEV_VERSION = "dev_version"  # 지금 기준이 개발 중 버전(-dev) — 검증한 기준일 수 없다
VERSION_MISMATCH = "version_mismatch"
FINGERPRINT_MISMATCH = "fingerprint_mismatch"


def enabled() -> bool:
    """잠금이 켜져 있나. `off` 만 끈다 — 그 밖의 값(빈 값·0·false·오타)은 모두 켜짐."""
    return os.getenv(ENV, "").strip().lower() != "off"


def problem(record_path: Path | None = None) -> str | None:
    """잠금 사유(위 상수), 받으면 None. 꺼져 있으면 None — 부르는 쪽이 enabled() 로 'off' 를 따로 적는다."""
    if not enabled():
        return None
    path = RECORD_PATH if record_path is None else record_path
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return UNVALIDATED
    except (OSError, ValueError):
        return UNREADABLE
    if not isinstance(record, dict) or not isinstance(record.get("version"), str) \
            or not isinstance(record.get("fingerprint"), str):
        return UNREADABLE
    current = criteria.version()
    if current.endswith("-dev"):
        return DEV_VERSION
    if record["version"] != current:
        return VERSION_MISMATCH
    if record["fingerprint"] != criteria.fingerprint(criteria.manifest_text()):
        return FINGERPRINT_MISMATCH
    return None
