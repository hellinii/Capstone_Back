"""test_p4_evaluate_additive.py — 판 4 라운드 S1 ⑤: 평가 응답은 키를 **더하기만** 했다.

기준(옛 응답):
- `tests/golden_frozen/evaluate_*.base.json`(S0 에 고정한 평가 골든 4개),
- `tests/golden_ed4/evaluate_bc_strong.edition3_base.json`(판 3 발급 때 실제 응답 원문).
지금 응답(엔드포인트를 실제로 불러 얻음)에 대해:
1. 옛 키·값이 모두 같은 값으로 있다(목록은 원소별, 타입까지).
2. 새로 생긴 경로는 판 4 자리(계획 2.6)뿐이다 — 최상위 `source_file`·`engine_version`, `f1_threshold_gain`·
   `confidence_intervals[*]`·`imbalance_effect` 안의 정해진 키.
3. 판 3 데이터에 통째로 저장되는 객체(`environment`·`derived_prediction`·`class_report`·`M22`·`class_distribution`·
   `warnings`)는 키 하나 늘지 않았다(값도 같다 — environment 는 실행마다 달라 키만 본다).
tests/golden 파일과 기준의 포함 관계는 test_frozen_goldens.py 가 따로 본다.
"""
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from golden_utils import _redact
from router_cases import CASES, request_json

TESTS = Path(__file__).parent
client = TestClient(app)

# 판 4 새 경로(계획 2.6). 지표 ID 는 `*` 로 바꿔 대조한다.
_F1_NEW = (
    "best_recall", "best_precision", "current_recall", "current_precision",
    "max_f1_raw", "current_f1_raw", "gain_raw", "best_threshold_raw",
    "best_recall_raw", "best_precision_raw", "current_recall_raw", "current_precision_raw",
    "best_threshold_text",
)
_IMBALANCE_NEW = tuple(f"{k}_raw" for k in (
    "imbalance_ratio", "accuracy_minus_macro_f1", "majority_minus_minority_f1",
    "accuracy", "macro_f1", "majority_f1", "minority_f1",
))
ALLOWED_NEW = frozenset(
    {"$.source_file", "$.engine_version"}
    | {f"$.results.success_metrics.f1_threshold_gain.{k}" for k in _F1_NEW}
    | {"$.results.success_metrics.confidence_intervals.*.low_raw",
       "$.results.success_metrics.confidence_intervals.*.high_raw"}
    | {f"$.results.success_metrics.imbalance_effect.{k}" for k in _IMBALANCE_NEW}
)
# 판 3 저장본에 통째로 들어가는 객체 — 키를 더하면 판 3 데이터가 바뀐다.
WHOLE_STORED = (
    ("environment",),
    ("results", "success_metrics", "derived_prediction"),
    ("results", "success_metrics", "class_report"),
    ("results", "success_metrics", "M22"),
    ("class_distribution",),
    ("warnings",),
)


def _bc_strong_response():
    data = TESTS / "data" / "bc_strong"
    return client.post(
        "/api/evaluate",
        files={"file": ("bc_strong.csv", (data / "bc_strong.csv").read_bytes(), "text/csv")},
        data={"data": (data / "bc_strong.payload.json").read_text(encoding="utf-8")},
    )


def _router_response(name: str):
    task, kind = re.fullmatch(r"evaluate_(\w+?)_(csv|json)", name).groups()
    c = CASES[task]
    path = c[kind]
    return client.post(
        "/api/evaluate",
        files={"file": (path.name, path.read_bytes(), "text/csv" if kind == "csv" else "application/json")},
        data={"data": request_json(task)},
    )


CASE_NAMES = ("evaluate_binary_csv", "evaluate_binary_json", "evaluate_multiclass_csv", "evaluate_multilabel_csv",
              "evaluate_bc_strong")


def _pair(name: str):
    if name == "evaluate_bc_strong":
        base_path = TESTS / "golden_ed4" / "evaluate_bc_strong.edition3_base.json"
        resp = _bc_strong_response()
    else:
        base_path = TESTS / "golden_frozen" / f"{name}.base.json"
        resp = _router_response(name)
    assert base_path.is_file(), base_path
    assert resp.status_code == 200, resp.text
    return json.loads(base_path.read_text(encoding="utf-8")), resp.json()


def _missing_or_changed(base, current, path="$"):
    if isinstance(base, dict):
        if not isinstance(current, dict):
            return [f"{path}: dict 가 아님"]
        out = []
        for k, v in base.items():
            out.extend([f"{path}.{k}: 빠짐"] if k not in current else _missing_or_changed(v, current[k], f"{path}.{k}"))
        return out
    if isinstance(base, list):
        if not isinstance(current, list) or len(base) != len(current):
            return [f"{path}: 목록 길이·형식이 다름"]
        return [p for i, (b, c) in enumerate(zip(base, current)) for p in _missing_or_changed(b, c, f"{path}[{i}]")]
    if type(base) is not type(current) or base != current:
        return [f"{path}: {base!r} → {current!r}"]
    return []


def _added_paths(base, current, path="$"):
    """current 에만 있는 키 경로(지표 ID 는 *). 목록 원소 안은 같은 자리끼리 본다."""
    out = set()
    if isinstance(base, dict) and isinstance(current, dict):
        for k, v in current.items():
            p = f"{path}.{k}"
            if k not in base:
                out.add(re.sub(r"confidence_intervals\.M\d+", "confidence_intervals.*", p))
            else:
                out |= _added_paths(base[k], v, p)
    elif isinstance(base, list) and isinstance(current, list):
        for i, (b, c) in enumerate(zip(base, current)):
            out |= _added_paths(b, c, f"{path}[{i}]")
    return out


def _dig(obj, keys):
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def _environment_free(base, current):
    """environment 값은 실행마다 다르다 — 양쪽을 자리표시자로 바꿔 비교한다(키는 남는다)."""
    return _redact(base), _redact(current)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_old_keys_and_values_are_kept(name):
    base, current = _environment_free(*_pair(name))
    problems = _missing_or_changed(base, current)
    assert problems == [], problems[:30]


@pytest.mark.parametrize("name", CASE_NAMES)
def test_new_paths_are_only_the_edition4_places(name):
    base, current = _pair(name)
    added = _added_paths(base, current)
    assert added <= ALLOWED_NEW, sorted(added - ALLOWED_NEW)
    # 최상위 두 칸과 구간 원값은 어느 응답에나 있다.
    assert {"$.source_file", "$.engine_version",
            "$.results.success_metrics.confidence_intervals.*.low_raw"} <= added


def test_binary_cases_add_every_threshold_gain_key():
    for name in ("evaluate_binary_csv", "evaluate_bc_strong"):
        base, current = _pair(name)
        added = _added_paths(base, current)
        assert {f"$.results.success_metrics.f1_threshold_gain.{k}" for k in _F1_NEW} <= added, name
        assert {f"$.results.success_metrics.imbalance_effect.{k}" for k in _IMBALANCE_NEW} <= added, name


@pytest.mark.parametrize("name", CASE_NAMES)
def test_whole_stored_objects_gain_no_key(name):
    base, current = _pair(name)
    for keys in WHOLE_STORED:
        b, c = _dig(base, keys), _dig(current, keys)
        assert (b is None) == (c is None), keys
        if b is None:
            continue
        if keys == ("environment",):
            assert set(c) == set(b) == {"libraries", "evaluated_at"}
            # S0 골든 사본은 libraries 를 자리표시자로 저장했다 — 그때는 지금 코드의 키 목록과 대조한다.
            libraries = {"python", "scikit-learn", "pandas", "numpy"}
            assert set(c["libraries"]) == libraries
            if isinstance(b["libraries"], dict):
                assert set(b["libraries"]) == libraries
            continue
        assert c == b, keys


def test_added_paths_has_teeth():
    base = {"a": {"M1": {"low": 1}}, "environment": {"x": 1}}
    cur = {"a": {"M1": {"low": 1, "low_raw": 1.0}}, "environment": {"x": 1, "engine_version": "1"}, "n": 1}
    assert _added_paths(base, cur) == {"$.a.M1.low_raw", "$.environment.engine_version", "$.n"}
    cur2 = {"results": {"success_metrics": {"confidence_intervals": {"M12": {"low_raw": 0}}}}}
    assert _added_paths({"results": {"success_metrics": {"confidence_intervals": {"M12": {}}}}}, cur2) == {
        "$.results.success_metrics.confidence_intervals.*.low_raw"}
