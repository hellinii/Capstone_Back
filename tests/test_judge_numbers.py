"""tests/test_judge_numbers.py — 숫자 대조 규칙(criteria/rules_numbers.py). 설계 6.1 예를 그대로 표로."""
import pytest

from app.judge.criteria.rules_numbers import matches, parse_printed, same_quantity


@pytest.mark.parametrize("text, data, ok", [
    ("0.7", 0.700, True),       # 설계 예
    ("0.71", 0.700, False),     # 설계 예
    ("94%", 0.944, True),       # 설계 예
    ("95%", 0.944, False),      # 설계 예
    ("94.4%", 0.944, True),
    ("0.912", 0.91249, True),
    ("0.912", 0.9126, False),
    ("0.9720", 0.97195, True),  # 판 4 늘린 자릿수
    ("33.5%", 33.5, True),      # 이미 백분율인 데이터
    ("1,234", 1234, True),
    ("1,234", 1235, False),
    ("+13", 13, True),
    ("−0.275", -0.275, True),   # U+2212
    ("-0.275", 0.275, False),
    ("0.066", -0.06588, True),  # 부호 없이 쓴 차이 — 크기로(방향은 말로)
    ("+0.066", -0.06588, False),  # 부호를 붙였으면 부호까지
    ("6.6%p", 0.066, True),
    ("12", 12.4, True),
    ("12", 12.6, False),
    ("0.5", 0.55, True),        # 반 칸 경계(올림·짝수 맞춤 어느 쪽도)
])
def test_matches(text, data, ok):
    assert matches(text, data) is ok


@pytest.mark.parametrize("text", ["약 0.9", "0.9배", "abc", "", "1,23", "0.9.1"])
def test_not_a_number(text):
    assert parse_printed(text) is None and matches(text, 0.9) is None


@pytest.mark.parametrize("a, b, same", [
    ("0.7", "0.700", True),
    ("0.71", "0.700", False),
    ("94.4%", "0.944", True),
    ("94%", "0.944", True),
    ("95%", "0.944", False),
    ("1,234", "1234", True),
    ("54", "55", False),
])
def test_same_quantity(a, b, same):
    assert same_quantity(a, b) is same
    assert same_quantity(b, a) is same


def test_touching_intervals_are_different_quantities():
    assert same_quantity("0.5", "0.6") is False
    assert same_quantity("0.55", "0.5") is True


def test_same_quantity_one_signed_compares_magnitude():
    assert same_quantity("16", "−16") is True and same_quantity("+16", "−16") is False and same_quantity("15", "−16") is False
