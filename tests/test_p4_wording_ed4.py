"""tests/test_p4_wording_ed4.py — 판 4 숫자 표기 함수(app.narrative.wording_ed4)가 두 저장소 공통 대조 표와 같은가.

대조 표(tests/contract/number_wording_ed4.json)는 S2 ① 공통 계약이 만든 것이고 같은 바이트 사본이 프론트에 있다(프론트 시험이
같은 표로 ed4Plain 등을 본다). 여기서는 표의 **모든 줄**을 백엔드 함수로 통과시킨다 — 규칙 이름마다 줄 수를 단정해 빈 갈래를
돌지 않게 한다. 표가 규칙과 맞는지(기준 구현)는 tests/test_p4_contract_ed4.py 가 본다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.narrative import wording_ed4 as w
from app.narrative.wording import josa

TABLE = json.loads((Path(__file__).parent / "contract" / "number_wording_ed4.json").read_text(encoding="utf-8"))


def run(case: dict) -> str:
    rule = case["rule"]
    if rule == "plain":
        return w.ed4_plain(case["value"])
    if rule == "judged":
        return w.ed4_judged(case["value"], case["threshold"], case["higher_is_better"], case["meets"])
    if rule == "signed_judged":
        return w.ed4_signed_judged(case["value"], case["threshold"], case["meets"])
    if rule == "ci_bound":
        return w.ed4_ci_bound(case["raw"], case["target"])
    if rule == "percent":
        return w.ed4_percent(case["text"])
    if rule == "percent_point":
        return w.ed4_percent_point(case["text"])
    if rule == "signed_count":
        return w.ed4_signed_count(case["value"])
    if rule == "given":
        return w.ed4_given(case["value"])
    if rule == "target":
        return w.ed4_target(case["value"])
    if rule == "latency":
        return w.ed4_latency(case["value"])
    if rule == "distinct":
        return w.ed4_distinct(case["value"], case["others"])
    raise AssertionError(rule)


RULE_COUNTS = {
    "plain": 35, "judged": 35, "ci_bound": 15, "percent": 11, "signed_judged": 7, "given": 12,
    "percent_point": 5, "signed_count": 4, "target": 11, "latency": 7, "distinct": 7,
}


def test_table_rows_by_rule():
    counts: dict[str, int] = {}
    for c in TABLE["number_cases"]:
        counts[c["rule"]] = counts.get(c["rule"], 0) + 1
    assert counts == RULE_COUNTS


@pytest.mark.parametrize("case", TABLE["number_cases"], ids=lambda c: f"{c['rule']}:{c['expected']}")
def test_backend_function_matches_table(case):
    assert run(case) == case["expected"]


@pytest.mark.parametrize("chain", TABLE["chains"], ids=lambda c: c["expected"])
def test_chain_number_then_josa(chain):
    text = run(chain["number"])
    assert text + josa(text, chain["pair"]) == chain["expected"]


@pytest.mark.parametrize("case", TABLE["josa_cases"], ids=lambda c: f"{c['word']}{c['expected']}")
def test_josa_is_the_shared_function(case):
    assert w.josa is josa
    assert w.josa(case["word"], case["pair"]) == case["expected"]


def test_trap_row_is_not_what_the_report_prints():
    """함정 줄 — 백엔드 기준표 모델 값(넷째 자리 0.9825)을 다시 반올림하면 0.983. 판 4 는 원값 0.98246 → 0.982 를 쓴다."""
    [trap] = [c for c in TABLE["number_cases"] if c.get("trap")]
    assert w.ed4_plain(trap["value"]) == "0.983"
    assert w.ed4_plain(0.9824561403508771) == "0.982"


def test_rounding_is_half_away_from_zero_not_python_round():
    """자기 검문 — 파이썬 round() 로 찍었다면 표의 0.0625 줄이 갈린다(이 시험이 반올림 규칙을 정말 고정한다)."""
    assert f"{round(0.0625, 3):.3f}" == "0.062"
    assert w.ed4_plain(0.0625) == "0.063" and w.ed4_plain(-0.0625) == "-0.063"


def test_text_rules_give_number_shaped_texts():
    import re

    num = re.compile(r"^-?\d+(\.\d+)?$")
    rows = [c for c in TABLE["number_cases"] if c["rule"] in ("plain", "judged", "signed_judged", "ci_bound")]
    assert len(rows) == 92
    assert [run(c) for c in rows if not num.match(run(c))] == []


def test_ratio_percent():
    assert w.ed4_ratio_percent(3, 171) == "1.8%"
    assert w.ed4_ratio_percent(1, 3) == "33.3%"
    assert w.ed4_ratio_percent(0, 3) == "0.0%"
    assert w.ed4_ratio_percent(0, 0) == "0.0%"


# ── 참조 범위 위치의 모델 값 ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "low", "high", "base", "target", "expected"),
    [
        (0.9824561403508771, 0.85, 0.92, "0.982", (0.95, True, True), "0.982"),  # 예시본 4.6 Accuracy
        (0.9692307692307692, 0.80, 0.92, None, None, "0.969"),  # 예시본 4.6 Precision(참고 지표 — plain)
        (0.9204, 0.85, 0.92, None, None, "0.9204"),  # 0.920 은 범위 끝(안)으로 읽힌다 — 실제는 위
        (0.79996, 0.80, 0.92, None, None, "0.79996"),  # 0.800 은 범위 안으로 읽힌다 — 실제는 아래
        (0.8, 0.80, 0.92, None, None, "0.800"),  # 범위 끝은 안
        (0.03, 0.05, 0.20, None, None, "0.030"),  # Hamming Loss(낮을수록 좋음)도 위치는 수치로만
        (0.9200004, 0.85, 0.92, None, None, "0.920001"),  # 여섯째 자리로도 같아 보이면 위치 쪽으로 자름(위 → 올림)
    ],
)
def test_range_value_text(value, low, high, base, target, expected):
    assert w.ed4_range_value_text(value, low, high, base, target) == expected


def test_position_words_are_neutral():
    """위치 말은 수치 위치만 — 우수·미흡 같은 평가 말이 없다(대응표 22, 부록 A10)."""
    assert w.POSITION_KO == {"above": "참조 범위보다 높음", "within": "참조 범위 안", "below": "참조 범위보다 낮음"}
    for word in ("우수", "미흡", "양호", "부족", "좋", "나쁨"):
        assert all(word not in v for v in w.POSITION_KO.values())
