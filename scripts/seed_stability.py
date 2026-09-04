"""Does the same search, run again, give the same answer?

NSGA-II is stochastic. Nothing in it proves it found the global front, so the
practical question is whether independent runs land in the same place. Spread
across seeds is an empirical stand-in for the guarantee the algorithm cannot
offer.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rocketopt.ric import motor_path, load_ric
from rocketopt.runner import default_spec, run
from rocketopt.simulate import PA_PER_PSI
from rocketopt.spec import ConstraintSpec
from rocketopt.units import KG_M2S_PER_LB_IN2S as LB
from rocketopt.units import M_PER_IN as IN

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 101, 2027)


def main() -> None:
    base = load_ric(motor_path(ROOT))
    out = []
    for seed in SEEDS:
        spec = default_spec(base)
        for var in spec.variables:
            var.step = 0.01 * IN
            var.free = True
        spec.constraints = [
            ConstraintSpec("peak_kn", "<=", 225.0, label="Peak Kn"),
            ConstraintSpec("max_pressure", "<=", 500 * PA_PER_PSI, label="Peak chamber pressure"),
            ConstraintSpec("peak_mass_flux", "<=", 1.05 * LB, label="Peak mass flux"),
            ConstraintSpec("port_throat", ">=", 1.4, label="Port/throat ratio"),
        ]
        for obj in spec.objectives:
            obj.enabled = obj.metric in ("initial_thrust", "total_impulse")
        spec.effort = "standard"
        spec.mode = "fast"
        spec.seed = seed
        result = run(spec, base)
        designs = result.to_dict()["designs"]
        out.append({"seed": seed, "n": len(designs),
                    "front": [{"initial_thrust": d["initial_thrust"],
                               "total_impulse": d["total_impulse"]} for d in designs]})
        print("seed {:>5}: {:>3} designs, thrust {:,.0f}..{:,.0f} N, impulse {:,.0f}..{:,.0f} N.s".format(
            seed, len(designs),
            min(d["initial_thrust"] for d in designs), max(d["initial_thrust"] for d in designs),
            min(d["total_impulse"] for d in designs), max(d["total_impulse"] for d in designs)),
            flush=True)
    (ROOT / "outputs" / "final" / "seed_stability.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
