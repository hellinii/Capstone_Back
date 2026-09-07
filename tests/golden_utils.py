"""tests/golden_utils.py — 골든 스냅샷(characterization) 테스트 헬퍼.

계층화 리팩토링(DECOMPOSITION_PLAN.md) 시 "동작 불변"을 기계적으로 보증하기 위한 장치.
엔드포인트의 현재 응답을 tests/golden/<name>.json 에 고정해 두고, 리팩토링 후에도
같은 스냅샷이 나오는지 비교한다.

- 최초 실행 또는 UPDATE_GOLDEN=1 환경변수 → 골든 파일을 (재)생성하고 통과.
- 이후 실행 → 저장된 골든과 비교. 다르면 실패(의도된 변경이면 UPDATE_GOLDEN=1 로 갱신).

부동소수는 플랫폼별 말단 비트 차이를 흡수하기 위해 비교 시 소수점 6자리로 정규화한다
(6자리를 넘는 값 변화 = 실제 동작 변화로 간주).

**비결정적 필드는 값 대신 존재만 고정한다**(REDACTED_PATHS). 평가 수행 시각·실행
환경의 라이브러리 버전은 실행할 때마다·기계마다 달라서 그대로 두면 골든이 항상 깨진다.
값을 자리표시자로 바꾸되 **키가 사라지면 여전히 잡히도록** 남겨 둔다 — 필드를 통째로
빼먹는 회귀는 골든이 잡아야 할 대상이다.
"""
import json
import os
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "golden"

# 값이 실행마다 달라지는 경로. 점 표기의 접두어로 대조한다.
# 값은 자리표시자로 대체하고 **키 존재 여부는 그대로 비교**한다.
REDACTED_PATHS = ("environment.evaluated_at", "environment.libraries")
_REDACTED = "<비결정적 값 — 존재만 고정>"


def _redact(o, path=""):
    """비결정적 경로의 값을 자리표시자로 바꾼다(구조는 보존)."""
    if any(path == p or path.startswith(p + ".") for p in REDACTED_PATHS):
        return _REDACTED
    if isinstance(o, dict):
        return {k: _redact(v, f"{path}.{k}" if path else str(k)) for k, v in o.items()}
    if isinstance(o, list):
        return [_redact(v, f"{path}[{i}]") for i, v in enumerate(o)]
    return o


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

    redacted = _redact(actual)

    if os.environ.get("UPDATE_GOLDEN") or not path.exists():
        path.write_text(
            json.dumps(redacted, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return

    expected = json.loads(path.read_text(encoding="utf-8"))
    exp_canon, act_canon = _canon(_redact(expected)), _canon(redacted)
    if exp_canon != act_canon:
        raise AssertionError(
            f"golden mismatch for '{name}'. "
            f"의도된 변경이면 `UPDATE_GOLDEN=1 pytest`로 갱신하세요.\n"
            + _format_diff(exp_canon, act_canon)
        )


# ── 진단 ──────────────────────────────────────────────────────────────────────
# ISSUES.md H-02 — 이전 구현은 최상위 키 목록만 출력해서, 중첩된 값 하나가 바뀐 경우
# expected/actual 두 줄이 글자 그대로 같았다(A-08 패치 시 실제로 그랬다). 골든이 흔들리는
# 변경(A-08·D-01·D-06·C-03)에서 회귀 신호가 판독 불가능했으므로 경로 단위 diff 로 교체한다.

_MAX_DIFFS = 25
_MISSING = object()  # "이쪽에는 키/원소가 없음" 을 None 과 구분하기 위한 sentinel


def _diff_paths(expected, actual, path="", out=None):
    """expected/actual 의 차이를 (경로, 기대값, 실제값) 목록으로 재귀 수집.

    비교는 호출부에서 _canon 을 거친 값에 대해 수행한다(6자리 반올림 허용 오차 유지).
    """
    if out is None:
        out = []
    if len(out) >= _MAX_DIFFS:
        return out

    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            _diff_paths(
                expected.get(key, _MISSING),
                actual.get(key, _MISSING),
                f"{path}.{key}" if path else str(key),
                out,
            )
        return out

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append((f"{path} (len)", len(expected), len(actual)))
        for i in range(max(len(expected), len(actual))):
            _diff_paths(
                expected[i] if i < len(expected) else _MISSING,
                actual[i] if i < len(actual) else _MISSING,
                f"{path}[{i}]",
                out,
            )
        return out

    if expected != actual:
        out.append((path or "$", expected, actual))
    return out


def _show(v):
    if v is _MISSING:
        return "<없음>"
    return json.dumps(v, ensure_ascii=False, default=str)


def _format_diff(expected, actual) -> str:
    diffs = _diff_paths(expected, actual)
    if not diffs:  # 최상위 타입 자체가 다른 경우 등
        return f"expected={_show(expected)}\n  actual={_show(actual)}"

    lines = [f"차이 {len(diffs)}건" + (f" (앞 {_MAX_DIFFS}건만 표시)" if len(diffs) >= _MAX_DIFFS else "") + ":"]
    for p, exp, act in diffs:
        lines.append(f"  {p}: expected={_show(exp)}  actual={_show(act)}")
    return "\n".join(lines)
