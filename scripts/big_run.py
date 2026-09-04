"""The largest search run so far: 100,000 simulations across 8 seeds.

Same motor, same limits, same nine variables as the earlier study. The only
change is budget and the number of independent searches, so the result is
directly comparable to the 14,400-simulation runs before it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rocketopt.ric import study_motor, load_ric, save_ric
from rocketopt.runner import build_space, default_spec, run
from rocketopt.simulate import PA_PER_PSI
from rocketopt.spec import ConstraintSpec
from rocketopt.units import KG_M2S_PER_LB_IN2S as LB
from rocketopt.units import M_PER_IN as IN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "big"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_ric(study_motor(ROOT))

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
    spec.mode = "fast"
    spec.budget_simulations = 100_000
    spec.seeds = 8

    budget = spec.budget
    print("{} seeds x ({} pop x {} gen = {:,}) = {:,} simulations".format(
        budget["seeds"], budget["pop"], budget["gen"], budget["per_seed"],
        budget["total"]), flush=True)

    last = {"msg": ""}

    def progress(stage, fraction, message):
        if message != last["msg"] and "generation" in message:
            head = message.split(":")[0]
            if head != last["msg"]:
                print("  {:>5.1f}%  {}".format(100 * fraction, message), flush=True)
                last["msg"] = head
        elif stage in ("verify", "done"):
            print("  {:>5.1f}%  {}".format(100 * fraction, message), flush=True)

    result = run(spec, base, on_progress=progress)
    payload = result.to_dict()
    (OUT / "result.json").write_text(json.dumps(payload))

    designs = sorted(payload["designs"], key=lambda d: -d["initial_thrust"])
    print("\n{:,} simulations in {:.0f}s -> {} designs on the merged front".format(
        payload["stats"]["simulations"], payload["stats"]["seconds"], len(designs)))
    if designs:
        print("  initial thrust {:,.0f} .. {:,.0f} N".format(
            min(d["initial_thrust"] for d in designs),
            max(d["initial_thrust"] for d in designs)))
        print("  total impulse  {:,.0f} .. {:,.0f} N.s".format(
            min(d["total_impulse"] for d in designs),
            max(d["total_impulse"] for d in designs)))
        space = build_space(spec, base)
        import numpy as np
        for tag, key in (("max_thrust", "initial_thrust"), ("max_impulse", "total_impulse")):
            best = max(designs, key=lambda d: d[key])
            save_ric(OUT / "{}.ric".format(tag), space.to_motor(np.array(best["x"])))


if __name__ == "__main__":
    main()
