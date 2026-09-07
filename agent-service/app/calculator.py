"""
Deterministic calculator tool.

Per spec: "Arithmetic must go through the calculator tool, never be produced
by the LLM from memory." This module evaluates a restricted arithmetic AST
(numbers, + - * / **, unary -, parentheses only) -- no names, no calls, no
attribute access -- so it is safe to run on any string the LLM proposes.
"""
import ast
import operator as op

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


class CalculatorError(ValueError):
    pass


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise CalculatorError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        try:
            return _ALLOWED_BINOPS[op_type](left, right)
        except ZeroDivisionError as e:
            raise CalculatorError("Division by zero") from e
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise CalculatorError(f"Unsupported unary operator: {op_type.__name__}")
        return _ALLOWED_UNARYOPS[op_type](_eval_node(node.operand))
    raise CalculatorError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> float:
    """Safely evaluate a pure-arithmetic expression string, e.g.
    "(3875-3410)/3410*100", and return a float result.

    Raises CalculatorError on anything outside +-*/%** and parentheses
    (no identifiers, function calls, subscripts, comprehensions, etc.).
    """
    if not expression or not expression.strip():
        raise CalculatorError("Empty expression")
    # Normalize a couple of common LLM output quirks.
    cleaned = expression.replace("×", "*").replace("÷", "/").replace(",", "")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"Invalid expression syntax: {e}") from e
    result = _eval_node(tree)
    return float(result)
