"""tests/test_judge_calc.py — 사실 일치 계산 도구(criteria/rules_calc.py). 식 문자열을 eval 하지 않는다."""
import builtins

import pytest

from app.judge.criteria.rules_calc import CalcError, calculate


@pytest.mark.parametrize("expr, value", [
    ("38 / (38 + 11)", 38 / 49), ("1+2*3", 7.0), ("-(2-5)", 3.0), ("0.944 * 100", 94.4), ("7 × 3 ÷ 2", 10.5), ("10 − 4", 6.0),
])
def test_arithmetic(expr, value):
    assert calculate(expr) == pytest.approx(value)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('x')", "abs(-1)", "2 ** 10", "x + 1", "1/0", "", "1" * 201, "[1,2]", "1 if 1 else 2", "1e400*10",
])
def test_rejects_everything_else(expr):
    with pytest.raises(CalcError):
        calculate(expr)


def test_does_not_call_eval(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("eval 을 불렀다")

    monkeypatch.setattr(builtins, "eval", boom)
    monkeypatch.setattr(builtins, "exec", boom)
    assert calculate("2*3") == 6.0
