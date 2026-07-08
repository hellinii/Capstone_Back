"""tests/golden_utils.py — 골든 스냅샷(characterization) 테스트 헬퍼.

계층화 리팩토링(DECOMPOSITION_PLAN.md) 시 "동작 불변"을 기계적으로 보증하기 위한 장치.
엔드포인트의 현재 응답을 tests/golden/<name>.json 에 고정해 두고, 리팩토링 후에도
같은 스냅샷이 나오는지 비교한다.

- 최초 실행 또는 UPDATE_GOLDEN=1 환경변수 → 골든 파일을 (재)생성하고 통과.
- 이후 실행 → 저장된 골든과 비교. 다르면 실패(의도된 변경이면 UPDATE_GOLDEN=1 로 갱신).

부동소수는 플랫폼별 말단 비트 차이를 흡수하기 위해 비교 시 소수점 6자리로 정규화한다
(6자리를 넘는 값 변화 = 실제 동작 변화로 간주).
"""
import json
import os
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "golden"


def _canon(o):
    """비교용 정규화: float 는 6자리 반올림, dict/list 는 재귀."""
    if isinstance(o, bool):
        return o
    if isinstance(o, float):
        return round(o, 6)
    if isinstance(o, dict):
        return {k: _canon(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_canon(v) for v in o]
    return o


def assert_golden(name: str, actual) -> None:
    """actual(JSON 직렬화 가능 객체)을 tests/golden/<name>.json 골든과 비교.

    UPDATE_GOLDEN 이 설정됐거나 골든이 없으면 생성 후 통과(부트스트랩).
    """
    GOLDEN_DIR.mkdir(exist_ok=True)
    path = GOLDEN_DIR / f"{name}.json"

    if os.environ.get("UPDATE_GOLDEN") or not path.exists():
        path.write_text(
            json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return

    expected = json.loads(path.read_text(encoding="utf-8"))
    if _canon(expected) != _canon(actual):
        # 실패 시 어디가 다른지 빠르게 보이도록 요약
        raise AssertionError(
            f"golden mismatch for '{name}'. "
            f"의도된 변경이면 `UPDATE_GOLDEN=1 pytest`로 갱신하세요.\n"
            f"expected keys={_keys(expected)}\n  actual keys={_keys(actual)}"
        )


def _keys(o):
    return sorted(o.keys()) if isinstance(o, dict) else type(o).__name__
