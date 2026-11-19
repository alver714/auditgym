"""Calculator safety and exactness (SPEC section 5)."""
from __future__ import annotations

import ast
import os
import random

import pytest

from auditgym import calculator
from auditgym.calculator import CalculatorError, evaluate
from auditgym.constants import MAX_CALC_EXPR_CHARS


# ------------------------------------------------------------------ exact results
@pytest.mark.parametrize("expr,expected", [
    ("1+2", "3"),
    ("0.1+0.2", "0.3"),                      # float math would give 0.30000000000000004
    ("0.1*3", "0.3"),
    ("1.1*1.1", "1.21"),
    ("0.07*100", "7"),
    ("1.50+1.50", "3"),                      # trailing zeros normalised away
    ("10/4", "2.5"),
    ("1/3", "0." + "3" * 40),                # prec=40 significant digits
    ("2/3", "0." + "6" * 39 + "7"),
    ("2**10", "1024"),
    ("2**12", "4096"),
    ("2**-12", "0.000244140625"),
    ("(-2)**2", "4"),
    ("-2**2", "-4"),                         # python precedence
    ("2**3**2", "512"),                      # right associative: 2**(3**2)
    ("7//2", "3"),
    ("-7//2", "-4"),                         # floor, not truncation
    ("7//-2", "-4"),
    ("7%3", "1"),
    ("-7%3", "2"),                           # python sign convention
    ("7%-3", "-2"),
    ("7.5%2", "1.5"),
    ("-(3)", "-3"),
    ("+(3)", "3"),
    ("--3", "3"),
    ("(1+2)*(3+4)", "21"),
    ("123456789012*987654321098", str(123456789012 * 987654321098)),
    ("12345678901234567890123456789*3", str(12345678901234567890123456789 * 3)),
    ("1e3", "1000"),
    ("1.5e2", "150"),
    ("1e-3", "0.001"),
    ("0.00", "0"),
    ("-0", "0"),
    ("0*-1", "0"),
    ("10**12", "1000000000000"),
    ("1/10**12", "0.000000000001"),          # never exponent notation
    ("1e-30", "0." + "0" * 29 + "1"),
    ("  42  ", "42"),
    ("12345*0.12", "1481.4"),
    ("1234.5678-1234.5678", "0"),
])
def test_exact_results(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize("expr,expected", [
    ("round(2.5)", "2"),                     # half even
    ("round(3.5)", "4"),
    ("round(-2.5)", "-2"),
    ("round(-3.5)", "-4"),
    ("round(0.5)", "0"),
    ("round(1.5)", "2"),
    ("round(2.675, 2)", "2.68"),
    ("round(2.665, 2)", "2.66"),
    ("round(1234.5678, 2)", "1234.57"),
    ("round(1234.5678, 0)", "1235"),
    ("round(1234.5678, -2)", "1200"),
    ("round(1481.4, 0)", "1481"),
    ("round(1/3, 4)", "0.3333"),
    ("round(12.5)", "12"),
    ("abs(-3.5)", "3.5"),
    ("abs(3)", "3"),
    ("abs(-0)", "0"),
    ("min(3, 1, 2)", "1"),
    ("max(3, 1, 2)", "3"),
    ("min(1.5, 1.25)", "1.25"),
    ("max(-1, -2)", "-1"),
    ("max(7)", "7"),
    ("max(1+1, 3-2, abs(-5))", "5"),
    ("round(abs(-2.5) * 3, 1)", "7.5"),
])
def test_whitelisted_functions(expr, expected):
    assert evaluate(expr) == expected


def test_result_is_plain_decimal_string_never_exponent():
    for expr in ("1e5", "1e-5", "10**12", "1/10**9", "123456789*10**12", "1e20", "5e-20"):
        out = evaluate(expr)
        assert isinstance(out, str)
        assert "e" not in out.lower()
        float_free = out.lstrip("-")
        assert float_free.replace(".", "", 1).isdigit()


def test_money_style_computation_matches_integer_decimal_math():
    # amount_minor / 10**exp * rate * 10**exp  with half-even rounding, as in the policy
    assert evaluate("round(1250000 / 10**2 * 475.123456 * 10**2, 0)") == "593904320"
    assert evaluate("round(10 * 0.5)") == "5"
    assert evaluate("round(2.5 * 1)") == "2"


def test_literals_not_converted_through_float():
    # 0.1 as a float is 0.1000000000000000055...; exact Decimal arithmetic keeps it clean
    assert evaluate("0.1*10-1") == "0"
    assert evaluate("0.3-0.1-0.1-0.1") == "0"
    assert evaluate("3*0.1-0.3") == "0"
    # a literal with more digits than a float can hold survives exactly
    assert evaluate("1234567890123456789.123456789+0") == "1234567890123456789.123456789"


def test_surrounding_whitespace_and_parentheses_ok():
    assert evaluate(" ( ( 1 + 2 ) ) ") == "3"
    assert evaluate("\t7\t") == "7"


# ------------------------------------------------------------------ division by zero
@pytest.mark.parametrize("expr", [
    "1/0", "1//0", "1%0", "1/(2-2)", "0/0", "1/0.0", "5//0.0", "5%0.00",
    "round(1/0)", "abs(1/0)", "max(1, 1/0)", "0**-1", "1/(1-1)*5", "2/(3-3.0)",
])
def test_division_by_zero_is_calculator_error(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)


def test_division_by_zero_message_says_so():
    with pytest.raises(CalculatorError, match="division by zero"):
        evaluate("1/0")


# ------------------------------------------------------------------ rejected syntax
@pytest.mark.parametrize("expr", [
    # names
    "x", "x+1", "pi", "__import__", "__builtins__", "True", "False", "None", "Infinity", "nan",
    "inf", "e", "len", "os", "abs", "round", "min", "max",       # bare whitelisted names are not values
    # calls other than the whitelist
    "pow(2,3)", "int(3)", "float(1)", "str(1)", "Decimal(1)", "sum(1,2)", "len('a')", "divmod(1,2)",
    "eval('1')", "exec('1')", "compile('1','a','eval')", "open('/etc/passwd')", "print(1)",
    "input()", "getattr(1,'real')", "globals()", "locals()", "vars()", "dir()", "type(1)",
    "__import__('os')", "__import__('os').system('true')", "sorted([1])", "range(3)", "ROUND(1)",
    "Abs(1)", "round_(1)", "float('nan')", "float('inf')", "int('1')", "complex(1,2)", "hash(1)",
    "id(1)", "repr(1)", "chr(65)", "ord('a')", "bool(1)", "all([1])", "any([1])",
    # call forms that are not plain Name(args)
    "abs.__call__(1)", "(lambda: 1)()", "(lambda x: x)(1)", "round(1, ndigits=2)",
    "max(*[1, 2])", "max(**{'a': 1})", "round(1)(2)", "abs(1)(2)", "[abs][0](1)",
    # attributes
    "(1).real", "(1).__class__", "(1).__class__.__mro__", "1 .real", "abs.__name__",
    "().__class__", "(1).bit_length()", "'a'.upper()", "(1.5).is_integer()",
    "abs(1).real", "round.__globals__",
    # strings / bytes / f-strings
    "'abc'", "\"1\"", "'1'+'2'", "'1'*3", "b'x'", "f'{1}'", "f'{1+1}'", "'%d' % 1", "'a' 'b'",
    "'1'", "round('1')", "abs('1')", "min('a', 'b')",
    # subscripts, containers, comprehensions, lambdas
    "[1,2][0]", "(1,2)[0]", "{1:2}[1]", "[1]", "(1,2)", "{1}", "{1:2}", "[x for x in [1]]",
    "{x for x in [1]}", "(x for x in [1])", "{x: x for x in [1]}", "lambda: 1", "lambda x: x",
    "[1,2,3][1:2]", "abs[1]", "max([1, 2])", "min((1, 2))", "round([1])", "1,2", "(1,)",
    # other operators / expressions
    "1<2", "1==1", "1<2<3", "1 in [1]", "1 is 1", "not 1", "1 and 2", "1 or 2", "1 if 1 else 2",
    "~1", "1&1", "1|1", "1^1", "1<<2", "1>>1", "1@2", "(x:=1)", "...", "1j", "2+3j", "await 1",
    "yield 1", "*1", "**1",
    # statements / syntax errors
    "x=1", "import os", "1;2", "1 2", "1 +", "(", ")", "((", "1+*2", "1..2", "1__0", "0x10", "0b11",
    "0o7", "$", "?", "1 +\n 2", "1\n+2", "# comment", "\\", "`1`", "1 = 1",
    "def f(): pass", "if 1: 2", "pass", "del x",
    # empty
    "", "   ", "\n", "\t",
])
def test_rejected(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)


@pytest.mark.parametrize("bad", [None, 5, 5.0, b"1+1", ["1+1"], {"e": "1"}, True, object()])
def test_non_string_input_rejected(bad):
    with pytest.raises(CalculatorError):
        evaluate(bad)


@pytest.mark.parametrize("expr", [
    "round()", "round(1,2,3)", "round(1, 0.5)", "round(1, 1/3)", "round(1, 31)", "round(1, -31)",
    "round(1, 10**20)", "abs()", "abs(1, 2)", "min()", "max()", "min(1, 'a')",
])
def test_bad_arity_or_arguments_rejected(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)


def test_too_many_function_args_rejected():
    with pytest.raises(CalculatorError):
        evaluate("max(" + ",".join(["1"] * 200) + ")")


# ------------------------------------------------------------------ exponent / size limits
@pytest.mark.parametrize("expr", [
    "2**13", "2**-13", "10**1000", "9**9**9", "10**100000", "2**(10**9)", "(10**12)**12",
    "99999999999999999999**12", "2**0.5", "2**(1/2)", "4**0.5", "2**1.5", "2**(1/3)",
    "(1/3)**(1/3)", "10**13", "0**-1",
    "2**(2**4)",                              # exponent 16 > 12
    "1e1000", "1e-1000", "1" + "0" * 200, "0." + "0" * 200 + "1",
    "10**12*10**12*10**12*10**12*10**12*10**12*10**12*10**12*10**12*10**12",
    "(10**12)*(10**12)*(10**12)*(10**12)*(10**12)*(10**12)*(10**12)*(10**12)*(10**12)",
    "1/10**12/10**12/10**12/10**12/10**12/10**12/10**12/10**12/10**12/10**12",
])
def test_huge_or_non_integer_exponent_and_out_of_range_rejected(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)


def test_exponent_boundary_accepted():
    assert evaluate("2**12") == "4096"
    assert evaluate("2**1.0") == "2"          # integral-valued exponent is accepted
    assert evaluate("2**(6*2)") == "4096"
    assert evaluate("(-3)**-3") == "-" + evaluate("1/27").lstrip("-")


def test_input_length_limit():
    ok = "1" + "+0" * 20 + " " * 400                          # long but shallow
    assert len(ok) <= MAX_CALC_EXPR_CHARS
    assert evaluate(ok) == "1"
    exactly = "1" + " " * (MAX_CALC_EXPR_CHARS - 1)
    assert len(exactly) == MAX_CALC_EXPR_CHARS
    assert evaluate(exactly) == "1"
    with pytest.raises(CalculatorError):
        evaluate("1" + " " * MAX_CALC_EXPR_CHARS)             # 513 chars
    with pytest.raises(CalculatorError):
        evaluate("1+" * 300 + "1")
    with pytest.raises(CalculatorError):
        evaluate("1" * 100000)
    with pytest.raises(CalculatorError):
        evaluate("(" * 10000 + "1" + ")" * 10000)


def test_nesting_depth_limit():
    ok_depth = 20
    assert evaluate("(1+" * ok_depth + "1" + ")" * ok_depth) == str(ok_depth + 1)
    deep = 80
    expr = "(1+" * deep + "1" + ")" * deep
    assert len(expr) <= MAX_CALC_EXPR_CHARS
    with pytest.raises(CalculatorError):
        evaluate(expr)
    with pytest.raises(CalculatorError):
        evaluate("-" * 120 + "1")
    with pytest.raises(CalculatorError):
        evaluate("abs(" * 80 + "1" + ")" * 80)
    with pytest.raises(CalculatorError):
        evaluate("-" * 450 + "1")                              # may also be a parser limit; either way an error


def test_calculator_error_is_a_value_error():
    assert issubclass(CalculatorError, ValueError)


def test_rejected_expression_has_no_side_effects(tmp_path):
    marker = tmp_path / "pwned"
    for expr in (f"__import__('os').system('touch {marker}')",
                 f"open('{marker}', 'w')",
                 f"exec(\"open('{marker}','w')\")"):
        with pytest.raises(CalculatorError):
            evaluate(expr)
    assert not marker.exists()


def test_error_message_is_short_and_does_not_echo_input():
    secret = "SECRETTOKEN_" + "z" * 300
    for expr in (secret, f"'{secret}'", f"{secret}()", "(1)." + secret):
        with pytest.raises(CalculatorError) as ei:
            evaluate(expr)
        assert secret not in str(ei.value)
        assert len(str(ei.value)) <= 120


# ------------------------------------------------------------------ implementation hygiene
def test_source_never_uses_eval_exec_or_compile():
    path = calculator.__file__
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"eval", "exec", "compile", "__import__", "float"}


def test_no_float_in_calculator_module():
    with open(calculator.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            assert n.func.id != "float"


# ------------------------------------------------------------------ fuzz: only str or CalculatorError
_TOKENS = ["1", "2", "0", "0.5", "3.14", "10", "100", "1e3", "+", "-", "*", "/", "//", "%", "**",
           "(", ")", ",", "round", "abs", "min", "max", "x", "'a'", ".", "[", "]", " ", "round(",
           "abs(", "min(", "max(", "__import__", "lambda", ":", "=", "j", "True", "None"]


def test_fuzz_never_raises_anything_but_calculator_error():
    rng = random.Random(1234)
    n_ok = 0
    for _ in range(3000):
        expr = "".join(rng.choice(_TOKENS) for _ in range(rng.randint(1, 14)))
        try:
            out = evaluate(expr)
        except CalculatorError:
            continue
        n_ok += 1
        assert isinstance(out, str) and out
        assert "e" not in out.lower() and "n" not in out.lower()   # no exponent / nan / inf
    assert n_ok > 20          # the fuzzer does produce some valid expressions


def test_fuzz_random_byte_strings():
    rng = random.Random(99)
    alphabet = [chr(c) for c in range(0, 128)] + ["é", "١", "−", "１", "퟿"]
    for _ in range(1500):
        expr = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        try:
            evaluate(expr)
        except CalculatorError:
            pass


def test_unicode_digits_and_minus_do_not_sneak_in():
    for expr in ("１+１", "١+1", "1−2", "1×2", "1÷2"):
        with pytest.raises(CalculatorError):
            evaluate(expr)


def test_evaluation_is_fast_on_adversarial_input():
    import time
    t0 = time.perf_counter()
    for expr in ("9**9**9", "(10**12)**12", "10**12*10**12*10**12*10**12*10**12*10**12*10**12",
                 "1" * 500, "1/3*3/3*3/3*3/3*3", "2**12**2"):
        try:
            evaluate(expr)
        except CalculatorError:
            pass
    assert time.perf_counter() - t0 < 2.0


def test_evaluate_is_deterministic_and_context_independent():
    import decimal
    old = decimal.getcontext().prec
    try:
        decimal.getcontext().prec = 5
        assert evaluate("1/3") == "0." + "3" * 40       # caller's context must not leak in
        decimal.getcontext().prec = 100
        assert evaluate("1/3") == "0." + "3" * 40
        assert decimal.getcontext().prec == 100          # and ours must not leak out
    finally:
        decimal.getcontext().prec = old
