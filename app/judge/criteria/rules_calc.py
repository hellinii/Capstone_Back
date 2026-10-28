"""criteria/rules_calc.py — 사실 일치 에이전트의 계산 도구(설계 6.1 "도구 3개" 중 계산).

식 문자열을 eval 하지 않는다. 숫자와 + − × ÷, 괄호, 단항 부호만 받는 작은 계산기다(ast 를 직접 걷는다).
기준 묶음에 둔다 — 계산 규칙이 바뀌면 판정이 바뀌므로 기준 지문에 든다.
"""
from __future__ import annotations

import ast
import math
import operator

MAX_EXPR_CHARS = 200
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalcError(ValueError):
    """계산할 수 없는 식."""


def calculate(expr: str) -> float:
    """사칙연산 식의 값. 받지 않는 꼴(이름·호출·거듭제곱·0으로 나눔·너무 긴 식)은 CalcError."""
    if not isinstance(expr, str) or not expr.strip() or len(expr) > MAX_EXPR_CHARS:
        raise CalcError("식이 비었거나 너무 길다")
    try:
        tree = ast.parse(expr.replace("×", "*").replace("÷", "/").replace("−", "-"), mode="eval")
    except SyntaxError as exc:
        raise CalcError("식을 읽을 수 없다") from exc

    def walk(node) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise CalcError("0으로 나눔")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        raise CalcError(f"받지 않는 식 요소: {type(node).__name__}")

    value = walk(tree.body)
    if not math.isfinite(value):
        raise CalcError("값이 유한하지 않다")
    return value
