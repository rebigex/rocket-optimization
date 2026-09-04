"""What a design does when it is built rather than drawn.

Every design the optimiser returns sits hard against whatever limits it was
given -- that is what optimising means. A motor at 497 psi against a 500 psi
ceiling is not 3 psi of margin, it is a coin flip once the throat is a few
thousandths off and the propellant is from a different batch.

This propagates that. Uncertainty is declared against the *hardware and the
propellant*, once, and applies unchanged to every optimisation -- it never needs
to know what was being optimised, which is why nothing here is per-run.

What varies together matters as much as how much:

* **Core diameters vary independently.** Each is a separate pass with a separate
  reamer or mandrel, so their errors do not cancel or reinforce.
* **The propellant batch is one draw.** Burn rate and density are properties of
  the mix, shared by every grain in the motor.
* **Nozzle dimensions are one draw each.** One throat, machined once.

Getting that wrong would flatter the answer: six independent core errors partly
cancel, while one shared batch error does not.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .ric import clone
from .simulate import PA_PER_PSI, Metrics, simulate_motor
from .units import M_PER_IN as IN

#: How each quantity is drawn. "per_grain" quantities get one draw per grain;
#: everything else gets one draw for the whole motor.
SCOPE_PER_GRAIN = "per_grain"
SCOPE_MOTOR = "motor"

#: The quantities that can vary, and where they live in a .ric. Named rather
#: than addressed by dotted path so a typo cannot silently perturb nothing.
TOLERANCE_FIELDS: Dict[str, Dict] = {
    "core_diameter": {
        "label": "Grain core diameter", "scope": SCOPE_PER_GRAIN, "unit": "in",
        "kind": "absolute",
        "help": "Reamer or mandrel error. One draw per grain — separate passes."},
    "grain_length": {
        "label": "Grain length", "scope": SCOPE_PER_GRAIN, "unit": "in",
        "kind": "absolute",
        "help": "Casting and cut-off error, independent per grain."},
    "throat": {
        "label": "Nozzle throat", "scope": SCOPE_MOTOR, "unit": "in",
        "kind": "absolute",
        "help": "Machining error on the throat. The single most sensitive dimension."},
    "exit": {
        "label": "Nozzle exit", "scope": SCOPE_MOTOR, "unit": "in",
        "kind": "absolute", "help": "Machining error on the exit cone."},
    "throat_length": {
        "label": "Throat length", "scope": SCOPE_MOTOR, "unit": "in",
        "kind": "absolute", "help": "Machining error on throat length."},
    "burn_rate_a": {
        "label": "Burn-rate coefficient", "scope": SCOPE_MOTOR, "unit": "%",
        "kind": "relative",
        "help": "Batch-to-batch variation in the propellant's a. One draw for "
                "the whole motor — every grain is from the same mix."},
    "burn_rate_n": {
        "label": "Burn-rate exponent", "scope": SCOPE_MOTOR, "unit": "%",
        "kind": "relative", "help": "Variation in the pressure exponent n."},
    "density": {
        "label": "Propellant density", "scope": SCOPE_MOTOR, "unit": "%",
        "kind": "relative", "help": "Packing and mix variation."},
    "nozzle_efficiency": {
        "label": "Nozzle efficiency", "scope": SCOPE_MOTOR, "unit": "%",
        "kind": "relative", "help": "How closely the nozzle performs to its model."},
    "ambient_pressure": {
        "label": "Ambient pressure", "scope": SCOPE_MOTOR, "unit": "%",
        "kind": "relative", "help": "Field elevation and weather on the day."},
}


@dataclass
class ToleranceSpec:
    """One quantity that varies, and by how much.

    ``sigma`` is one standard deviation: inches for an absolute quantity, a
    fraction for a relative one. For a uniform distribution it is the half-width
    instead, because that is how a tolerance is written on a drawing.
    """

    field: str
    sigma: float = 0.0
    distribution: str = "normal"      # "normal" | "uniform"
    enabled: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ToleranceSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def default_tolerances() -> List[ToleranceSpec]:
    """A starting point, not a measurement.

    These are plausible for a home shop and a hand-mixed batch. They are
    assumptions and are meant to be replaced with what your lathe and your
    propellant actually do.
    """
    return [
        ToleranceSpec("core_diameter", 0.005 * IN),
        ToleranceSpec("throat", 0.005 * IN),
        ToleranceSpec("burn_rate_a", 0.03),
        ToleranceSpec("grain_length", 0.020 * IN, enabled=False),
        ToleranceSpec("exit", 0.010 * IN, enabled=False),
        ToleranceSpec("throat_length", 0.010 * IN, enabled=False),
        ToleranceSpec("burn_rate_n", 0.01, enabled=False),
        ToleranceSpec("density", 0.01, enabled=False),
        ToleranceSpec("nozzle_efficiency", 0.02, enabled=False),
        ToleranceSpec("ambient_pressure", 0.02, enabled=False),
    ]


def _draw(rng, spec: ToleranceSpec, size) -> np.ndarray:
    if spec.distribution == "uniform":
        return rng.uniform(-spec.sigma, spec.sigma, size)
    return rng.normal(0.0, spec.sigma, size)


def perturb(motor: Dict, tolerances: Sequence[ToleranceSpec], rng) -> Dict:
    """One motor as it might actually come out of the shop."""
    built = clone(motor)
    grains = built["grains"]
    nozzle = built["nozzle"]
    propellant = built["propellant"]
    n_grains = len(grains)

    for spec in tolerances:
        if not spec.enabled or spec.sigma <= 0:
            continue
        meta = TOLERANCE_FIELDS.get(spec.field)
        if meta is None:
            continue
        size = n_grains if meta["scope"] == SCOPE_PER_GRAIN else None
        delta = _draw(rng, spec, size)

        if spec.field == "core_diameter":
            for grain, d in zip(grains, np.atleast_1d(delta)):
                props = grain["properties"]
                # A core can never eat the whole grain, however unlucky the draw.
                largest = props["diameter"] - 2 * 0.02 * IN
                props["coreDiameter"] = float(
                    min(max(props["coreDiameter"] + d, 0.05 * IN), largest))
        elif spec.field == "grain_length":
            for grain, d in zip(grains, np.atleast_1d(delta)):
                props = grain["properties"]
                props["length"] = float(max(props["length"] + d, 0.1 * IN))
        elif spec.field in ("throat", "exit", "throat_length"):
            key = {"throat": "throat", "exit": "exit",
                   "throat_length": "throatLength"}[spec.field]
            nozzle[key] = float(max(nozzle[key] + float(delta), 1e-4))
            if spec.field == "throat":
                # openMotor rejects a nozzle whose exit is under its throat.
                nozzle["exit"] = float(max(nozzle["exit"], nozzle["throat"] * 1.01))
        elif spec.field == "burn_rate_a":
            for tab in propellant["tabs"]:
                tab["a"] = float(tab["a"] * (1 + float(delta)))
        elif spec.field == "burn_rate_n":
            for tab in propellant["tabs"]:
                tab["n"] = float(np.clip(tab["n"] * (1 + float(delta)), 0.05, 0.95))
        elif spec.field == "density":
            propellant["density"] = float(propellant["density"] * (1 + float(delta)))
        elif spec.field == "nozzle_efficiency":
            nozzle["efficiency"] = float(
                np.clip(nozzle["efficiency"] * (1 + float(delta)), 0.05, 1.5))
        elif spec.field == "ambient_pressure":
            built["config"] = dict(built["config"])
            built["config"]["ambPressure"] = float(max(
                built["config"]["ambPressure"] * (1 + float(delta)), 1.0))
    return built


# --------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------


def _limit_check(metrics: Metrics, constraints) -> Dict[str, bool]:
    """Which limits this particular build broke."""
    out = {}
    for spec in constraints:
        if not getattr(spec, "enabled", True):
            continue
        value = getattr(metrics, spec.metric, None)
        if value is None:
            continue
        out[spec.metric] = (value <= spec.value if spec.op == "<="
                            else value >= spec.value)
    return out


def propagate(motor: Dict, tolerances: Sequence[ToleranceSpec], constraints,
              samples: int = 400, timestep: float = 0.01, seed: int = 0,
              workers: Optional[int] = None) -> Dict:
    """Simulates the same design many times, built slightly differently each time.

    A coarser timestep than final verification is deliberate: this is a
    distribution, and several hundred runs at 0.002 s would cost minutes to
    sharpen numbers that the tolerance assumptions dominate anyway. Peak mass
    flux does drift with timestep, so its exceedance rate here is read against
    the same timestep throughout and compared like for like.
    """
    from .sampling import simulate_many

    active = [t for t in tolerances if t.enabled and t.sigma > 0]
    nominal = simulate_motor(motor, timestep=timestep)
    if not active:
        return {"available": False, "reason": "no tolerances set"}

    rng = np.random.default_rng(seed)
    builds = [perturb(motor, active, rng) for _ in range(samples)]
    results = simulate_many(builds, timestep=timestep, workers=workers)

    good = [m for m in results if m.ok]
    if not good:
        return {"available": False, "reason": "every perturbed build failed to simulate"}

    checks = [_limit_check(m, constraints) for m in good]
    per_limit = []
    for spec in constraints:
        if not getattr(spec, "enabled", True):
            continue
        passes = [c.get(spec.metric) for c in checks if spec.metric in c]
        if not passes:
            continue
        values = np.array([getattr(m, spec.metric) for m in good], dtype=float)
        per_limit.append({
            "metric": spec.metric,
            "label": getattr(spec, "label", "") or spec.metric,
            "op": spec.op,
            "limit": float(spec.value),
            "exceed_probability": float(1.0 - np.mean(passes)),
            "nominal": float(getattr(nominal, spec.metric, float("nan"))),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p05": float(np.percentile(values, 5)),
            "worst": float(values.max() if spec.op == "<=" else values.min()),
            "samples": values.tolist(),
        })

    all_pass = np.array([all(c.values()) for c in checks], dtype=bool)
    pass_rate = float(all_pass.mean())
    # Wilson interval, so a clean sweep still reports an honest floor.
    z, n = 1.96, len(all_pass)
    denom = 1 + z**2 / n
    centre = (pass_rate + z**2 / (2 * n)) / denom
    half = z * math.sqrt(pass_rate * (1 - pass_rate) / n + z**2 / (4 * n**2)) / denom

    spread = {}
    for name in ("initial_thrust", "total_impulse", "isp", "burn_time",
                 "max_pressure", "peak_kn", "peak_mass_flux"):
        values = np.array([getattr(m, name) for m in good], dtype=float)
        spread[name] = {
            "nominal": float(getattr(nominal, name)),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "sd": float(values.std()),
        }

    return {
        "available": True,
        "samples": int(n),
        "simulated_ok": len(good),
        "pass_rate": pass_rate,
        "pass_low": max(centre - half, 0.0),
        "pass_high": min(centre + half, 1.0),
        "per_limit": sorted(per_limit, key=lambda r: -r["exceed_probability"]),
        "spread": spread,
        "tolerances": [t.to_dict() for t in active],
    }


def summarise(report: Dict) -> str:
    """One sentence a builder can act on."""
    if not report.get("available"):
        return report.get("reason", "no robustness data")
    rate = report["pass_rate"]
    worst = report["per_limit"][0] if report["per_limit"] else None
    if worst is None or worst["exceed_probability"] < 0.005:
        return "{:.0f}% of builds stay inside every limit.".format(100 * rate)
    return ("{:.0f}% of builds stay inside every limit; {} is the one that goes "
            "over, on {:.0f}% of them.".format(
                100 * rate, worst["label"], 100 * worst["exceed_probability"]))
