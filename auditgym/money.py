"""Exact money / FX arithmetic. Integers in minor units, Decimal rates. No floats."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP, localcontext

from .constants import CURRENCY_EXPONENT

ROUNDING_MODES = {"half_even": ROUND_HALF_EVEN, "half_up": ROUND_HALF_UP}


def exponent(currency: str) -> int:
    return CURRENCY_EXPONENT[currency]


def convert_minor(amount_minor: int, source: str, target: str, rate: str | Decimal,
                  rounding: str = "half_even") -> int:
    """Convert integer minor units of `source` into integer minor units of `target`.

    target_minor = round(amount_minor / 10**exp_s * rate * 10**exp_t)
    """
    if source == target:
        return int(amount_minor)
    with localcontext() as ctx:
        ctx.prec = 50
        major = Decimal(int(amount_minor)).scaleb(-exponent(source))
        value = (major * Decimal(str(rate))).scaleb(exponent(target))
        return int(value.quantize(Decimal(1), rounding=ROUNDING_MODES[rounding]))


def format_rate(rate: Decimal, places: int = 6) -> str:
    return str(rate.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN))
