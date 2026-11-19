"""Safe Decimal arithmetic evaluator (SPEC section 5).

`evaluate(expr)` parses with `ast` and walks the tree itself; nothing is ever passed to eval().
Allowed: + - * / // % ** (integer exponent, |e| <= 12), unary + -, numeric literals (converted from
their source text to Decimal, never through float), parentheses, and the calls round(x, n)
(half_even), abs, min, max.  Everything else raises CalculatorError.
"""
from __future__ import annotations

import ast
from decimal import (ROUND_HALF_EVEN, Context, Decimal, DecimalException, DivisionByZero,
                     InvalidOperation, Overflow, localcontext)

from .constants import MAX_CALC_EXPR_CHARS

MAX_DEPTH = 50
MAX_POW = 12
MAX_ROUND_DIGITS = 30
MAX_CALL_ARGS = 32
_EXP_BOUND = 100          # literal / result magnitude bound: 10**-100 .. 10**100
_PREC = 40


class CalculatorError(ValueError):
    """Raised for any expression that is not plain, safe arithmetic."""


def _ctx() -> Context:
    return Context(prec=_PREC, rounding=ROUND_HALF_EVEN, Emax=_EXP_BOUND + 20,
                   Emin=-(_EXP_BOUND + 20), traps=[InvalidOperation, DivisionByZero, Overflow])


def _check_range(d: Decimal) -> Decimal:
    if d.is_finite() and d != 0 and abs(d.adjusted()) > _EXP_BOUND:
        raise CalculatorError("number out of range")
    return d


def _no_underflow(result: Decimal, *operands: Decimal) -> Decimal:
    """A product / quotient / power of non-zero operands that collapsed to 0 underflowed: out of range,
    never a silent '0'."""
    if result == 0 and all(o != 0 for o in operands):
        raise CalculatorError("number out of range")
    return result


def _as_int(d: Decimal, what: str) -> int:
    if not d.is_finite() or d != d.to_integral_value():
        raise CalculatorError(f"{what} must be an integer")
    return int(d)


def _floordiv(a: Decimal, b: Decimal) -> Decimal:
    if b == 0:
        raise CalculatorError("division by zero")
    q = a // b                      # truncates toward zero
    r = a - q * b
    if r != 0 and ((r < 0) != (b < 0)):
        q -= 1
    return q


class _Evaluator:
    def __init__(self, source: str):
        self.src = source

    def eval(self, node: ast.AST, depth: int = 0) -> Decimal:
        if depth > MAX_DEPTH:
            raise CalculatorError("expression nested too deeply")
        d = depth + 1
        if isinstance(node, ast.Expression):
            return self.eval(node.body, d)
        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.UnaryOp):
            v = self.eval(node.operand, d)
            if isinstance(node.op, ast.USub):
                return _check_range(-v)
            if isinstance(node.op, ast.UAdd):
                return +v
            raise CalculatorError("operator not allowed")
        if isinstance(node, ast.BinOp):
            return _check_range(self._binop(node, d))
        if isinstance(node, ast.Call):
            return _check_range(self._call(node, d))
        raise CalculatorError("expression element not allowed")

    def _constant(self, node: ast.Constant) -> Decimal:
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise CalculatorError("only numeric literals are allowed")
        text = ast.get_source_segment(self.src, node)
        if text is None:
            raise CalculatorError("bad literal")
        try:
            dec = Decimal(text.strip())
        except (InvalidOperation, ValueError):
            raise CalculatorError("bad numeric literal") from None
        if not dec.is_finite():
            raise CalculatorError("bad numeric literal")
        return _check_range(dec)

    def _binop(self, node: ast.BinOp, d: int) -> Decimal:
        op = node.op
        a = self.eval(node.left, d)
        b = self.eval(node.right, d)
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return _no_underflow(a * b, a, b)
        if isinstance(op, ast.Div):
            if b == 0:
                raise CalculatorError("division by zero")
            return _no_underflow(a / b, a)
        if isinstance(op, ast.FloorDiv):
            return _floordiv(a, b)
        if isinstance(op, ast.Mod):
            q = _floordiv(a, b)
            return a - q * b
        if isinstance(op, ast.Pow):
            e = _as_int(b, "exponent")
            if abs(e) > MAX_POW:
                raise CalculatorError("exponent too large")
            if a == 0 and e < 0:
                raise CalculatorError("division by zero")
            return _no_underflow(a ** e, a)
        raise CalculatorError("operator not allowed")

    def _call(self, node: ast.Call, d: int) -> Decimal:
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise CalculatorError("function call not allowed")
        name = node.func.id
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise CalculatorError("function call not allowed")
        if len(node.args) > MAX_CALL_ARGS:
            raise CalculatorError("too many arguments")
        if name == "round":
            if len(node.args) not in (1, 2):
                raise CalculatorError("round takes 1 or 2 arguments")
            x = self.eval(node.args[0], d)
            n = 0
            if len(node.args) == 2:
                n = _as_int(self.eval(node.args[1], d), "round digits")
                if abs(n) > MAX_ROUND_DIGITS:
                    raise CalculatorError("round digits out of range")
            return x.quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_EVEN)
        if name == "abs":
            if len(node.args) != 1:
                raise CalculatorError("abs takes 1 argument")
            return abs(self.eval(node.args[0], d))
        if name in ("min", "max"):
            if not node.args:
                raise CalculatorError(f"{name} needs arguments")
            vals = [self.eval(a, d) for a in node.args]
            return min(vals) if name == "min" else max(vals)
        raise CalculatorError("function not allowed")


def _format(d: Decimal) -> str:
    if d == 0:
        return "0"
    s = format(d.normalize(), "f")
    return s


def evaluate(expr: str) -> str:
    """Evaluate an arithmetic expression exactly; return a plain decimal string."""
    if not isinstance(expr, str):
        raise CalculatorError("expression must be a string")
    if len(expr) > MAX_CALC_EXPR_CHARS:
        raise CalculatorError("expression too long")
    text = expr.strip()
    if not text:
        raise CalculatorError("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise CalculatorError("syntax error") from None
    try:
        with localcontext(_ctx()):
            result = _Evaluator(text).eval(tree)
            result = _check_range(result)
            return _format(result)
    except CalculatorError:
        raise
    except RecursionError:
        raise CalculatorError("expression nested too deeply") from None
    except (DecimalException, ArithmeticError, ValueError):
        raise CalculatorError("arithmetic error") from None
