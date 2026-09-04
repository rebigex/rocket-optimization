"""Does a genetic search actually beat drawing designs at random?

Same motor, same limits, same simulation budget as optimisation B: 14,400 runs.
The only difference is how the next design gets chosen. Sobol is used rather
than uniform random because it is the stronger version of the argument -- it
covers the box more evenly than pseudo-random draws, so if it still loses, plain
Monte Carlo loses by more.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rocketopt.optimize import Objective, pareto_indices, scale_constraints
from rocketopt.ric import motor_path, load_ric
from rocketopt.runner import build_objective, build_space, default_spec, timestep_bias
from rocketopt.sampling import evaluate_batch, sobol_designs
from rocketopt.simulate import PA_PER_PSI, simulate_motor
from rocketopt.spec import ConstraintSpec
from rocketopt.units import KG_M2S_PER_LB_IN2S as LB
from rocketopt.units import M_PER_IN as IN

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 14400


def main() -> None:
    base = load_ric(motor_path(ROOT))
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

    space = build_space(spec, base)
    baseline = simulate_motor(base, timestep=spec.verify_timestep)
    bias = timestep_bias(base, spec.search_timestep, spec.verify_timestep)
    objective = build_objective(spec, baseline, bias).strict()

    print("drawing {:,} designs from a Sobol sequence...".format(BUDGET), flush=True)
    started = time.time()
    X = sobol_designs(space, BUDGET, seed=5)
    frame = evaluate_batch(space, X, timestep=spec.search_timestep, workers=12)
    elapsed = time.time() - started

    violation = scale_constraints(frame, objective, space).max(axis=1)
    legal = frame[frame["ok"].to_numpy(dtype=bool) & (violation <= 0)].reset_index(drop=True)
    print("  {:,} simulations in {:.0f}s -> {:,} legal ({:.2f}%)".format(
        len(frame), elapsed, len(legal), 100 * len(legal) / len(frame)))

    out = {"budget": BUDGET, "legal": int(len(legal)), "seconds": elapsed}
    if len(legal):
        axes = -objective.matrix(legal)
        front = legal.iloc[pareto_indices(axes)]
        out["front"] = front[["initial_thrust", "total_impulse"]].to_dict("records")
        out["best_thrust"] = float(front["initial_thrust"].max())
        out["best_impulse"] = float(front["total_impulse"].max())
        out["front_size"] = int(len(front))
        print("  best initial thrust {:,.0f} N   best impulse {:,.0f} N.s   front {} wide".format(
            out["best_thrust"], out["best_impulse"], out["front_size"]))
    else:
        print("  nothing legal at all")
    (ROOT / "outputs" / "final" / "random_search.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
