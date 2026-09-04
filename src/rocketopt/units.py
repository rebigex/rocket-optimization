"""Length, pressure and mass-flux conversions, and parsing of shop fractions.

The motor this project started from is imperial to its last digit -- cores of
exactly 1.600 / 1.900 / 2.200 in, a 6.000 in grain, a 1.300 in throat -- so the
app has to speak inches as fluently as metres. Everything internal stays in SI;
these helpers exist only at the edges where a person types a number.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Union

MM_PER_IN = 25.4
M_PER_IN = 0.0254
PA_PER_PSI = 6894.757293168361
#: kg/(m^2 s) per lb/(in^2 s). Matches openMotor's own conversion table.
KG_M2S_PER_LB_IN2S = 703.0696

LENGTH_UNITS = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": M_PER_IN}
PRESSURE_UNITS = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "psi": PA_PER_PSI}
MASS_FLUX_UNITS = {"kg/(m^2*s)": 1.0, "lb/(in^2*s)": KG_M2S_PER_LB_IN2S}


def parse_number(text: Union[str, float, int]) -> float:
    """Reads a number the way a machinist would write one.

    Accepts ``0.0625``, ``1/16``, ``1 1/16`` and ``2-1/2`` as well as plain
    floats, because a step size on a shop drawing is far more often a fraction
    than a decimal.
    """
    if isinstance(text, (int, float)):
        return float(text)
    cleaned = str(text).strip().replace("″", "").replace('"', "")
    if not cleaned:
        raise ValueError("empty number")
    cleaned = cleaned.replace("-", " ") if "/" in cleaned else cleaned
    parts = cleaned.split()
    total = 0.0
    for part in parts:
        total += float(Fraction(part)) if "/" in part else float(part)
    return total


def to_si(value: Union[str, float], unit: str, kind: str = "length") -> float:
    """Converts a displayed value into SI."""
    table = _table(kind)
    if unit not in table:
        raise ValueError("unknown {} unit {!r}".format(kind, unit))
    return parse_number(value) * table[unit]


def from_si(value: float, unit: str, kind: str = "length") -> float:
    """Converts an SI value into the requested display unit."""
    table = _table(kind)
    if unit not in table:
        raise ValueError("unknown {} unit {!r}".format(kind, unit))
    return float(value) / table[unit]


def _table(kind: str):
    if kind == "length":
        return LENGTH_UNITS
    if kind == "pressure":
        return PRESSURE_UNITS
    if kind == "mass_flux":
        return MASS_FLUX_UNITS
    raise ValueError("unknown quantity kind {!r}".format(kind))


def snap(value, step: float, low: float = None, high: float = None):
    """Rounds to the nearest whole multiple of ``step``, anchored at zero.

    Anchoring at zero rather than at the lower bound is deliberate: tooling
    comes in whole fractions of an inch measured from nothing, so a 1/16 in grid
    should offer 1.6250 in, not 1.6250 in plus whatever the lower bound happened
    to be. When bounds are given the result is nudged *inward* so snapping can
    never push a value outside a range it started inside.
    """
    import numpy as np

    if not step or step <= 0:
        result = np.asarray(value, dtype=float)
    else:
        result = np.round(np.asarray(value, dtype=float) / step) * step
        if low is not None:
            below = result < low - 1e-12
            result = np.where(below, np.ceil(np.asarray(low) / step) * step, result)
        if high is not None:
            above = result > high + 1e-12
            result = np.where(above, np.floor(np.asarray(high) / step) * step, result)
    if low is not None or high is not None:
        result = np.clip(result, low, high)
    return result if np.ndim(value) else float(result)


def round_up_to_step(value: float, step: float) -> float:
    """Smallest whole multiple of ``step`` that is at least ``value``.

    Used on the minimum core increment so that walking a ladder of cores keeps
    every rung on the machining grid.
    """
    import math

    if not step or step <= 0:
        return float(value)
    return math.ceil(value / step - 1e-9) * step


def format_length(value_m: float, unit: str = "mm", digits: int = None) -> str:
    shown = from_si(value_m, unit)
    if digits is None:
        digits = 4 if unit == "in" else 2
    return "{:.{}f} {}".format(shown, digits, unit)
