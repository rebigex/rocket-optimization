"""How many distinct motors a configuration actually admits.

Not a plain product of the per-variable counts. Two things shrink it, and one
thing makes it awkward:

* **Cores are stored sorted**, so a set of six core diameters is one motor, not
  720. The count is a multiset coefficient rather than N⁶ -- for a 0.01 in grid
  over 4 in that is the difference between 10²⁴ and 10¹⁵.
* **Ordering rules cut further.** Requiring each core to exceed the one ahead of
  it by a fixed step removes whole regions; grouping grains onto shared mandrels
  removes independent dimensions outright.
* **The exit is bounded by the throat**, so how many exit diameters exist
  depends on which throat you picked. That sum is taken exactly rather than
  approximated, because the two are not independent.

The point of showing the number is context: a search that evaluates fourteen
thousand designs out of 10¹⁵ is not being lazy, it is doing the only thing that
was ever possible.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .spec import RunSpec
from .units import M_PER_IN as IN


def _steps(low: float, high: float, step: float) -> int:
    """Grid points in [low, high], inclusive of both ends."""
    if step <= 0:
        return 0
    span = high - low
    if span < 0:
        return 0
    return int(math.floor(span / step + 1e-9)) + 1


def _multiset(values: int, slots: int) -> int:
    """Non-decreasing sequences of ``slots`` drawn from ``values`` options."""
    if values <= 0 or slots <= 0:
        return 0
    return math.comb(values + slots - 1, slots)


def core_combinations(spec: RunSpec, n_grains: int) -> Dict:
    """How many distinct core arrangements the ordering rule leaves."""
    cores = [v for v in spec.variables if v.name.startswith("core")][:n_grains]
    if not cores:
        return {"count": 1, "exact": True, "note": ""}

    free = [c for c in cores if c.free]
    if not free:
        return {"count": 1, "exact": True, "grid_values": 1,
                "note": "every core held at a fixed value"}
    if any(c.step <= 0 for c in free):
        return {"count": None, "exact": False, "grid_values": None,
                "note": "core diameters are continuous — no step set"}

    first = free[0]
    uniform = all((c.low, c.high, c.step) == (first.low, first.high, first.step)
                  for c in free)
    values = _steps(first.low, first.high, first.step)
    k = len(free)

    if not uniform:
        # Per-grain bounds differ, so the tidy multiset formula does not apply.
        # The product is an upper bound before ordering collapses it.
        total = 1
        for c in free:
            total *= max(_steps(c.low, c.high, c.step), 1)
        return {"count": total, "exact": False, "grid_values": None,
                "note": "cores have different bounds — this is an upper bound, "
                        "before the ordering rule collapses it"}

    mode = spec.ordering.mode
    if mode == "paired" and spec.ordering.groups:
        k = len(spec.ordering.groups)
        count = _multiset(values, k)
        note = "{} mandrel size{} shared across {} grains".format(
            k, "" if k == 1 else "s", n_grains)
    elif mode == "strict":
        gap = int(round((spec.ordering.min_step or 0.0) / first.step))
        gap = max(gap, 1)  # strictly increasing means at least one grid step
        available = values - (k - 1) * gap
        count = _multiset(available, k) if available > 0 else 0
        note = ("each core at least {} grid step{} above the one ahead"
                .format(gap, "" if gap == 1 else "s"))
    elif mode == "none":
        count = values ** k
        note = "order is free, so every permutation counts separately"
    else:
        count = _multiset(values, k)
        note = "sorted smallest-forward, so a set of cores is one motor"

    return {"count": count, "exact": True, "grid_values": values, "note": note}


def nozzle_combinations(spec: RunSpec) -> Dict:
    """Throat and exit counted together, because the exit range follows the throat."""
    by_name = {v.name: v for v in spec.variables}
    throat = by_name.get("throat")
    exit_var = by_name.get("exit")
    if throat is None:
        return {"count": 1, "exact": True, "note": ""}

    ratio = 1.15  # SpaceConfig.exit_min_ratio
    if not throat.free:
        throat_values = [throat.fixed_value if throat.fixed_value is not None
                         else throat.low]
    elif throat.step <= 0:
        return {"count": None, "exact": False,
                "note": "throat diameter is continuous — no step set"}
    else:
        n = _steps(throat.low, throat.high, throat.step)
        throat_values = [throat.low + i * throat.step for i in range(n)]

    if exit_var is None or not exit_var.free:
        return {"count": len(throat_values), "exact": True,
                "note": "exit held, so only the throat varies"}
    if exit_var.step <= 0:
        return {"count": None, "exact": False,
                "note": "exit diameter is continuous — no step set"}

    # For each throat, the exit runs from the larger of its own floor and
    # 1.15 x throat, up to the airframe limit.
    total = 0
    for t in throat_values:
        low = max(exit_var.low, t * ratio)
        total += max(_steps(low, exit_var.high, exit_var.step), 1)
    return {"count": total, "exact": True,
            "throat_values": len(throat_values),
            "note": "exit range depends on the throat, so these are counted together"}


def size_space(spec: RunSpec, n_grains: int,
               evaluated: Optional[int] = None,
               rate: float = 32.0) -> Dict:
    """Total distinct motors, plus what the search will actually look at."""
    cores = core_combinations(spec, n_grains)
    nozzle = nozzle_combinations(spec)

    others: List[Dict] = []
    total = 1
    continuous = False
    for var in spec.variables:
        if var.name.startswith("core") or var.name in ("throat", "exit"):
            continue
        if not var.free:
            others.append({"name": var.label or var.name, "values": 1, "held": True})
            continue
        if var.step <= 0:
            continuous = True
            others.append({"name": var.label or var.name, "values": None,
                           "held": False})
            continue
        n = _steps(var.low, var.high, var.step)
        others.append({"name": var.label or var.name, "values": n, "held": False})
        total *= max(n, 1)

    parts = [cores.get("count"), nozzle.get("count")]
    if any(p is None for p in parts) or continuous:
        grand: Optional[int] = None
    else:
        grand = total
        for p in parts:
            grand *= p

    budget = spec.budget
    if evaluated is None:
        evaluated = budget["total"]
        if spec.mode == "pareto":
            evaluated += budget["samples"]

    result = {
        "total": grand,
        "total_text": _humanise(grand),
        "continuous": grand is None,
        "cores": cores,
        "nozzle": nozzle,
        "others": others,
        "evaluated": evaluated,
        "free_variables": sum(1 for v in spec.variables if v.free),
        "held_variables": sum(1 for v in spec.variables if not v.free),
    }
    if grand and grand > 0:
        result["fraction"] = evaluated / grand
        result["fraction_text"] = _one_in(grand / evaluated) if evaluated else ""
        result["brute_force_text"] = _duration(grand / max(rate, 1e-9))
    return result


def _humanise(value: Optional[int]) -> str:
    """A count a person can hold in their head."""
    if value is None:
        return "unbounded — at least one dimension has no step"
    if value < 1_000_000:
        return "{:,}".format(value)
    digits = len(str(value)) - 1
    lead = value / (10 ** digits)
    return "{:.1f} × 10^{}".format(lead, digits)


def _one_in(ratio: float) -> str:
    if ratio < 10:
        return "about 1 in {:.1f}".format(ratio)
    return "about 1 in {}".format(_humanise(int(ratio)))


def _duration(seconds: float) -> str:
    """Wall clock for simulating every configuration once."""
    minute, hour, day, year = 60, 3600, 86400, 31_557_600
    if seconds < minute:
        return "{:.0f} seconds".format(seconds)
    if seconds < hour:
        return "{:.0f} minutes".format(seconds / minute)
    if seconds < day:
        return "{:.1f} hours".format(seconds / hour)
    if seconds < year:
        return "{:.0f} days".format(seconds / day)
    years = seconds / year
    if years > 1e6:
        return "{} years".format(_humanise(int(years)))
    return "{:,.0f} years".format(years)


# ---------------------------------------------------------------------------
# Using the limits to shrink the space
# ---------------------------------------------------------------------------

#: The closed-form peak Kn below runs up to ~1% above what openMotor reports,
#: because the simulator samples regression on a timestep and stops a grain a
#: hair before its web is gone. Rejecting only past this margin means the screen
#: can never discard a design the simulator would have accepted.
KN_SCREEN_MARGIN = 0.02


def peak_burn_area(cores: np.ndarray, diameter: float, length: float,
                   samples: int = 160) -> np.ndarray:
    """Largest burning area reached during the burn, uninhibited BATES.

    openMotor gives, for each grain at regression r,
    ``pi*(d+2r)*(L-2r) + (pi/2)*(D^2 - (d+2r)^2)`` while any web is left. That is
    closed form, so the whole burn can be evaluated without simulating it -- and
    the peak of the sum is what the Kn limit actually constrains.
    """
    cores = np.atleast_2d(np.asarray(cores, dtype=float))
    web = np.minimum((diameter - cores) / 2.0, length / 2.0)
    steps = np.linspace(0.0, 1.0, samples)
    # One regression axis shared by every design, scaled per design by its own
    # longest-burning grain, so each is sampled across its whole burn.
    r = steps[None, :, None] * web.max(axis=1)[:, None, None]
    d = cores[:, None, :] + 2 * r
    live = r < web[:, None, :]
    area = (math.pi * d * (length - 2 * r)
            + (math.pi / 2) * (diameter**2 - d**2)) * live
    return area.sum(axis=2).max(axis=1)


def _core_grid(spec: RunSpec) -> Optional[np.ndarray]:
    cores = [v for v in spec.variables if v.name.startswith("core") and v.free]
    if not cores or cores[0].step <= 0:
        return None
    first = cores[0]
    n = _steps(first.low, first.high, first.step)
    return first.low + np.arange(n) * first.step


def tighten_bounds(spec: RunSpec, base_motor: Dict) -> Dict:
    """Bounds the limits rule out outright, before any simulation.

    Two of them are exact rather than heuristic:

    * **Throat floor.** Burning area is smallest when every core is at its
      minimum, so if even that area needs a bigger throat to hold Kn, no design
      with a smaller throat can exist at all.
    * **Core ceiling.** Holding every other core at its minimum, the widest a
      single core can be before the largest allowed throat still cannot hold Kn.

    Both are necessary conditions, so narrowing to them throws nothing away.
    """
    grains = base_motor["grains"]
    diameter = grains[0]["properties"]["diameter"]
    length = grains[0]["properties"]["length"]
    n_grains = len(grains)
    by_name = {v.name: v for v in spec.variables}
    throat = by_name.get("throat")
    grid = _core_grid(spec)

    kn_limit = next((c.value for c in spec.enabled_constraints
                     if c.metric == "peak_kn" and c.op == "<="), None)
    port_throat = next((c.value for c in spec.enabled_constraints
                        if c.metric == "port_throat" and c.op == ">="), None)

    out: Dict = {"changes": [], "throat_low": None, "core_high": None}
    if kn_limit is None or grid is None or throat is None or not throat.free:
        return out

    allowed = kn_limit * (1 + KN_SCREEN_MARGIN)
    smallest = np.full((1, n_grains), grid[0])
    area_min = float(peak_burn_area(smallest, diameter, length)[0])
    throat_floor = math.sqrt(4 * area_min / (allowed * math.pi))
    if throat_floor > throat.low + 1e-9:
        out["throat_low"] = min(throat_floor, throat.high)
        out["changes"].append({
            "variable": "throat",
            "from": throat.low, "to": out["throat_low"],
            "why": "Burning area is smallest with every core at {} in, and even "
                   "that needs a {} in throat to hold Kn at {:g}. Nothing narrower "
                   "can be legal.".format(
                       "{:.2f}".format(grid[0] / IN),
                       "{:.2f}".format(throat_floor / IN), kn_limit),
        })

    # Widest a single core can be, with every other core at its minimum.
    throat_area_max = math.pi * throat.high**2 / 4
    budget = allowed * throat_area_max
    trials = np.tile(grid[0], (len(grid), n_grains))
    trials[:, -1] = grid
    areas = peak_burn_area(trials, diameter, length)
    usable = np.flatnonzero(areas <= budget)
    core_var = next(v for v in spec.variables if v.name.startswith("core"))
    if len(usable) and usable[-1] < len(grid) - 1:
        out["core_high"] = float(grid[usable[-1]])
        out["changes"].append({
            "variable": "cores",
            "from": core_var.high, "to": out["core_high"],
            "why": "Past {} in a single core pushes burning area beyond what even "
                   "the largest allowed throat ({} in) can hold at Kn {:g}.".format(
                       "{:.2f}".format(out["core_high"] / IN),
                       "{:.2f}".format(throat.high / IN), kn_limit),
        })

    if port_throat is not None:
        out["port_throat_note"] = (
            "Port/throat ≥ {:g} also ties the aft core to the throat: the widest "
            "core must be at least √{:g} × the throat.".format(port_throat, port_throat))
    return out


def _uniform_multisets(grid: np.ndarray, slots: int, samples: int,
                       rng) -> np.ndarray:
    """Uniform samples of the sorted core vectors, not of the ordered ones.

    Sorting uniformly-drawn tuples is *not* uniform over sorted vectors -- it
    over-weights sets with many distinct values, because those have more
    orderings that collapse onto them. Since the count being reported is a count
    of sorted vectors, the sampling has to match it, so this uses the
    stars-and-bars bijection: choose ``slots`` distinct positions from
    ``len(grid) + slots - 1`` and subtract their index.
    """
    n = len(grid)
    picks = np.array([rng.choice(n + slots - 1, size=slots, replace=False)
                      for _ in range(samples)])
    picks.sort(axis=1)
    return grid[picks - np.arange(slots)[None, :]]


def estimate_feasible(spec: RunSpec, base_motor: Dict, samples: int = 4000,
                      seed: int = 0) -> Dict:
    """What share of the grid actually satisfies the closed-form limits.

    Sampled rather than enumerated -- the space is far too large to walk -- but
    every screen applied here is exact arithmetic on the same formulas openMotor
    uses, so this is measuring the constraints and not a model of them. Only the
    limits that are closed-form are applied, so the true legal share is at most
    what comes back.
    """
    grains = base_motor["grains"]
    diameter = grains[0]["properties"]["diameter"]
    length = grains[0]["properties"]["length"]
    n_grains = len(grains)
    grid = _core_grid(spec)
    by_name = {v.name: v for v in spec.variables}
    throat = by_name.get("throat")
    if grid is None or throat is None:
        return {"available": False}

    kn_limit = next((c.value for c in spec.enabled_constraints
                     if c.metric == "peak_kn" and c.op == "<="), None)
    port_throat = next((c.value for c in spec.enabled_constraints
                        if c.metric == "port_throat" and c.op == ">="), None)
    if kn_limit is None and port_throat is None:
        return {"available": False}

    rng = np.random.default_rng(seed)
    cores = _uniform_multisets(grid, n_grains, samples, rng)
    if throat.free and throat.step > 0:
        t_grid = throat.low + np.arange(_steps(throat.low, throat.high, throat.step)) * throat.step
        throats = rng.choice(t_grid, size=samples)
    else:
        throats = np.full(samples, throat.fixed_value if throat.fixed_value is not None
                          else throat.low)

    throat_area = math.pi * throats**2 / 4
    keep = np.ones(samples, dtype=bool)
    screens = []
    if kn_limit is not None:
        kn = peak_burn_area(cores, diameter, length) / throat_area
        passed = kn <= kn_limit * (1 + KN_SCREEN_MARGIN)
        screens.append({"name": "Peak Kn ≤ {:g}".format(kn_limit),
                        "pass": float(passed.mean())})
        keep &= passed
    if port_throat is not None:
        ratio = (math.pi * cores[:, -1] ** 2 / 4) / throat_area
        passed = ratio >= port_throat
        screens.append({"name": "Port/throat ≥ {:g}".format(port_throat),
                        "pass": float(passed.mean())})
        keep &= passed

    hits = int(keep.sum())
    fraction = hits / samples
    # Wilson interval, so a zero-hit sample still reports an honest ceiling.
    z = 1.96
    denom = 1 + z**2 / samples
    centre = (fraction + z**2 / (2 * samples)) / denom
    half = z * math.sqrt(fraction * (1 - fraction) / samples
                         + z**2 / (4 * samples**2)) / denom
    return {"available": True, "samples": samples, "hits": hits,
            "fraction": fraction,
            "low": max(centre - half, 0.0), "high": min(centre + half, 1.0),
            "screens": screens}


def equivalent_limits(spec: RunSpec, base_motor: Dict) -> List[Dict]:
    """Limits that are secretly the same limit.

    Chamber pressure is a monotone function of Kn, so a pressure ceiling and a
    Kn ceiling are one constraint wearing two hats. Knowing which of the two
    actually binds -- and by how little -- is worth more than either number on
    its own, because tightening the slack one changes nothing at all.
    """
    from motorlib.propellant import Propellant

    kn_limit = next((c.value for c in spec.enabled_constraints
                     if c.metric == "peak_kn" and c.op == "<="), None)
    psi_limit = next((c.value for c in spec.enabled_constraints
                      if c.metric == "max_pressure" and c.op == "<="), None)
    if kn_limit is None or psi_limit is None:
        return []

    propellant = Propellant(base_motor["propellant"])
    low, high = 1.0, 5000.0
    for _ in range(80):                     # invert the Saint-Robert solve
        mid = 0.5 * (low + high)
        if propellant.getPressureFromKn(mid) < psi_limit:
            low = mid
        else:
            high = mid
    kn_at_psi = high
    psi_at_kn = propellant.getPressureFromKn(kn_limit)
    psi = psi_limit / 6894.757293168361

    if kn_at_psi < kn_limit:
        binding, slack = "pressure", "Kn"
        detail = ("{:.0f} psi is reached at Kn {:.1f}, below your Kn ceiling of "
                  "{:g}. Pressure is the real limit; raising Kn on its own would "
                  "change nothing.".format(psi, kn_at_psi, kn_limit))
    else:
        binding, slack = "Kn", "pressure"
        detail = ("Kn {:g} produces {:.0f} psi, under your {:.0f} psi ceiling. Kn "
                  "is the real limit; raising the pressure ceiling on its own "
                  "would change nothing.".format(
                      kn_limit, psi_at_kn / 6894.757293168361, psi))
    return [{
        "kind": "equivalent",
        "title": "Peak Kn and peak pressure are one constraint",
        "binding": binding, "slack": slack,
        "kn_at_pressure": kn_at_psi, "pressure_at_kn": psi_at_kn,
        "detail": detail,
    }]


def reduction_chain(spec: RunSpec, base_motor: Dict,
                    samples: int = 6000) -> Dict:
    """Total, then what the limits provably remove, then what they likely remove.

    Kept as three separate numbers because they have different standing: the
    first is exact combinatorics, the second is exact arithmetic on the limits,
    and the third is a sampled estimate with an interval on it. Collapsing them
    into one figure would hide which is which.
    """
    import copy

    n_grains = len(base_motor["grains"])
    total = size_space(spec, n_grains)
    tightened = tighten_bounds(spec, base_motor)

    narrowed = copy.deepcopy(spec)
    for var in narrowed.variables:
        if var.name == "throat" and tightened.get("throat_low"):
            var.low = max(var.low, tightened["throat_low"])
        if var.name.startswith("core") and tightened.get("core_high"):
            var.high = min(var.high, tightened["core_high"])
    after_bounds = size_space(narrowed, n_grains)

    screened = estimate_feasible(narrowed, base_motor, samples=samples)
    legal: Optional[int] = None
    if screened.get("available") and after_bounds.get("total"):
        legal = int(after_bounds["total"] * screened["fraction"])

    return {
        "total": total.get("total"),
        "total_text": total.get("total_text"),
        "after_bounds": after_bounds.get("total"),
        "after_bounds_text": after_bounds.get("total_text"),
        "legal": legal,
        "legal_text": _humanise(legal) if legal is not None else None,
        "tightened": tightened,
        "screened": screened,
        "equivalences": equivalent_limits(spec, base_motor),
        "shrink_bounds": (total["total"] / after_bounds["total"]
                          if total.get("total") and after_bounds.get("total") else None),
        "shrink_total": (total["total"] / legal
                         if total.get("total") and legal else None),
    }
