"""The calculator accepts abs(x) and nothing else that looks like a call.

"How far apart" questions ask for an absolute difference, and their gold
derivations use abs(). A018 found both correct operands and was declined only
because the calculator rejected abs(...).
"""

import pytest

from app.calculator import CalculatorError, calculate


def test_a018s_rejected_formula_now_gives_the_gold_answer():
    formula = "abs((442147+398227+397133+469285)-(75188+85539+101235+103645))"

    assert calculate(formula) == 1341185.0


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("abs(-3.5)", 3.5),
        ("abs(9,447-314,258)", 304811.0),
        ("abs(100-250)/2", 75.0),
        ("abs(abs(-2)-5)", 3.0),
    ],
)
def test_abs_of_arithmetic(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "max(1, 2)",
        "abs(1, 2)",
        "abs()",
        "abs(x=1)",
        "abs(*[1])",
        "abs.__call__(1)",
        "abs(__import__('os'))",
        "round(2.5)",
    ],
)
def test_other_calls_are_still_rejected(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)
