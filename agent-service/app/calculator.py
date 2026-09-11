"""
Deterministic calculator tool.

Per spec: "Arithmetic must go through the calculator tool, never be produced
by the LLM from memory." This module evaluates a restricted arithmetic AST
(numbers, + - * / ** %, unary -, parentheses and abs(x) only) -- no other
names, no other calls, no attribute access -- so it is safe to run on any
string the LLM proposes.
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
# "How far apart" questions ask for an absolute difference, and their gold
# derivations use abs() (8 practice questions). A018 found both correct
# operands and was declined only because abs(...) was rejected.
_ALLOWED_FUNCTIONS = {
    "abs": abs,
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
    if isinstance(node, ast.Call):
        # Exactly abs(x): a plain name, one positional argument, no keywords.
        # The argument goes through this same restricted evaluator.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _ALLOWED_FUNCTIONS
            and len(node.args) == 1
            and not node.keywords
        ):
            return _ALLOWED_FUNCTIONS[node.func.id](_eval_node(node.args[0]))
        raise CalculatorError("Unsupported function call: only abs(x) is allowed")
    raise CalculatorError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> float:
    """Safely evaluate a pure-arithmetic expression string, e.g.
    "(3875-3410)/3410*100" or "abs(9447-314258)", and return a float result.

    Raises CalculatorError on anything outside +-*/%**, parentheses and
    abs(x) (no identifiers, other function calls, subscripts,
    comprehensions, etc.).
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
