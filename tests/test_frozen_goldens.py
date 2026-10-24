"""test_frozen_goldens.py — 판 4 라운드 S0: 지금 골든을 '갱신 경로가 없는' 고정 사본으로 지킨다.

- 서술(narrative_*)·검증(validate_*) 골든은 이 라운드에서 바뀌면 안 된다. tests/golden 원본은 UPDATE_GOLDEN 이나 파일
  삭제로 다시 만들어질 수 있어, 사본을 tests/golden_frozen/ 에 두고 원래 테스트와 같은 계산을 다시 해서 글자 그대로 비교한다.
  파일이 없으면 실패한다. 원래 테스트의 요청 조립은 tests/frozen_s0.py 에 옮겨 두었다.
- 평가(evaluate_*) 골든은 다음 세션에서 키가 늘어난다(판 4 새 값 — 계획 2.6). 그래서 같음이 아니라 "옛 골든 ⊆ 지금 골든"
  (옛 키·값이 모두 그대로 있다)을 본다.
- 고정 폴더의 파일 목록과 sha256 은 test_frozen_manifest.py 가 본다.
"""
import json
from pathlib import Path

import pytest

import frozen_s0 as fz
from app.main import app

GOLDEN = Path(__file__).parent / "golden"

FROZEN_COPIES = (
    "narrative_binary_fallback",
    "narrative_binary_fallback_problems",
    "narrative_legacy_main_fallback",
    "narrative_job_fallback_binary",
    "narrative_job_fallback_multiclass",
    "narrative_job_fallback_multilabel",
    "validate_binary_clean",
    "validate_binary_errors",
    "validate_multiclass_clean",
    "validate_multiclass_errors",
    "validate_multilabel_clean",
    "validate_multilabel_errors",
)
EVALUATE_BASES = ("evaluate_binary_csv", "evaluate_binary_json", "evaluate_multiclass_csv", "evaluate_multilabel_csv")


@pytest.fixture
def _no_client():
    prev = getattr(app.state, "anthropic_client", None)
    app.state.anthropic_client = None
    try:
        yield
    finally:
        app.state.anthropic_client = prev


@pytest.fixture(scope="module")
def legacy_goldens():
    prev = getattr(app.state, "anthropic_client", None)
    app.state.anthropic_client = None
    try:
        return fz.legacy_route_goldens()
    finally:
        app.state.anthropic_client = prev


# ── 1. 서술·검증 골든 고정 사본 ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [n for n in FROZEN_COPIES if not n.startswith("narrative_job_")])
def test_legacy_route_and_validate_outputs_equal_frozen_copy(name, legacy_goldens):
    fz.assert_frozen(name, legacy_goldens[name])


async def test_job_fallback_outputs_equal_frozen_copy():
    computed = await fz.job_fallback_goldens()
    assert sorted(computed) == sorted(n for n in FROZEN_COPIES if n.startswith("narrative_job_"))
    for name, value in computed.items():
        fz.assert_frozen(name, value)


def test_every_frozen_copy_is_computed(legacy_goldens):
    """고정 사본마다 위 비교가 실제로 돈다(목록에서 빠진 사본이 비교 없이 남지 않게)."""
    names = {p.stem for p in fz.FROZEN.glob("*.json") if not p.name.endswith(".base.json")}
    assert names == set(FROZEN_COPIES)
    assert set(legacy_goldens) | {f"narrative_job_fallback_{t}" for t in fz.CONTRACT_CASES} == set(FROZEN_COPIES)


# ── 2. 평가 골든 — 옛 키·값이 지금 골든에 그대로 있다 ───────────────────────────────────


def _missing_or_changed(base, current, path="$") -> list[str]:
    """base 의 모든 키·값이 current 에 같은 값으로 있는지. 목록은 길이와 원소를 차례로 본다. 어긋난 경로 목록."""
    if isinstance(base, dict):
        if not isinstance(current, dict):
            return [f"{path}: dict 가 아님"]
        out = []
        for k, v in base.items():
            if k not in current:
                out.append(f"{path}.{k}: 빠짐")
            else:
                out.extend(_missing_or_changed(v, current[k], f"{path}.{k}"))
        return out
    if isinstance(base, list):
        if not isinstance(current, list) or len(current) != len(base):
            return [f"{path}: 목록 길이·형식이 다름"]
        out = []
        for i, (b, c) in enumerate(zip(base, current)):
            out.extend(_missing_or_changed(b, c, f"{path}[{i}]"))
        return out
    if type(base) is not type(current) or base != current:
        return [f"{path}: {base!r} → {current!r}"]
    return []


@pytest.mark.parametrize("name", EVALUATE_BASES)
def test_evaluate_golden_keeps_every_base_key_and_value(name):
    base_path = fz.FROZEN / f"{name}.base.json"
    current_path = GOLDEN / f"{name}.json"
    assert base_path.is_file(), base_path
    assert current_path.is_file(), current_path
    base = json.loads(base_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    problems = _missing_or_changed(base, current)
    assert problems == [], problems[:30]


def test_subset_check_has_teeth():
    base = {"a": 1, "b": {"c": [1, {"d": 0.5}]}, "e": True}
    assert _missing_or_changed(base, {**base, "new": 9}) == []
    assert _missing_or_changed(base, {"a": 1, "e": True, "b": {"c": [1, {"d": 0.5}], "x": 1}}) == []
    assert _missing_or_changed(base, {"a": 1, "b": {"c": [1, {"d": 0.5}]}}) == ["$.e: 빠짐"]
    assert _missing_or_changed(base, {**base, "b": {"c": [1, {"d": 0.50001}]}})
    assert _missing_or_changed(base, {**base, "b": {"c": [1]}})
    assert _missing_or_changed(base, {**base, "e": 1})  # True 와 1 은 다르다
    assert _missing_or_changed(base, {**base, "a": 1.0})  # 1 과 1.0 도 다르다(직렬화가 바뀐다)
