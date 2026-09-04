"""The two optimisations requested: cores only, and cores plus the nozzle.

Both maximise initial thrust (mean over the first 0.35 s) and total impulse at
the same time, so the answer is a trade-off curve rather than a single motor.
Everything the .ric fixes -- 5.00 in grain outer diameter, 6.00 in length, six
grains, uninhibited ends, propellant, and every nozzle property except the three
being searched -- is carried through untouched.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rocketopt.ric import study_motor, load_ric, save_ric
from rocketopt.runner import default_spec, run
from rocketopt.spec import ConstraintSpec
from rocketopt.units import KG_M2S_PER_LB_IN2S as LB
from rocketopt.units import M_PER_IN as IN
from rocketopt.units import PA_PER_PSI as PSI

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final"
GRID = 0.01 * IN


def build(base, free_nozzle: bool):
    spec = default_spec(base)
    spec.constraints = [
        ConstraintSpec("peak_kn", "<=", 225.0, label="Peak Kn"),
        ConstraintSpec("max_pressure", "<=", 500 * PSI, label="Peak chamber pressure"),
        ConstraintSpec("peak_mass_flux", "<=", 1.05 * LB, label="Peak mass flux"),
        ConstraintSpec("port_throat", ">=", base["config"]["minPortThroat"],
                       label="Port/throat ratio"),
    ]
    for var in spec.variables:
        var.step = GRID
        if var.name.startswith("core"):
            var.free = True
        else:
            var.free = free_nozzle
    for obj in spec.objectives:
        obj.enabled = obj.metric in ("initial_thrust", "total_impulse")
    spec.effort = "thorough"
    spec.mode = "fast"
    return spec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_ric(study_motor(ROOT))

    for key, free_nozzle in (("A_cores_only", False), ("B_cores_and_nozzle", True)):
        spec = build(base, free_nozzle)
        label = "cores only, nozzle fixed" if not free_nozzle else "cores + nozzle"
        searched = [v.name for v in spec.variables if v.free]
        print("\n=== {} ({} variables: {}) ===".format(label, len(searched), ", ".join(searched)))
        result = run(spec, base, on_progress=lambda s, f, m: (
            print("   {:>5.0f}%  {}".format(100 * f, m), flush=True)
            if s == "search" and abs(f * 100 % 10) < 1.2 else None))
        payload = result.to_dict()
        (OUT / "{}.json".format(key)).write_text(json.dumps(payload))
        front = sorted(payload["designs"], key=lambda d: -d["initial_thrust"])
        print("   -> {} legal designs, {} simulations, {:.0f}s".format(
            len(front), payload["stats"]["simulations"], payload["stats"]["seconds"]))
        if front:
            print("      thrust {:.0f}..{:.0f} N   impulse {:.0f}..{:.0f} N.s".format(
                min(d["initial_thrust"] for d in front),
                max(d["initial_thrust"] for d in front),
                min(d["total_impulse"] for d in front),
                max(d["total_impulse"] for d in front)))


if __name__ == "__main__":
    main()
