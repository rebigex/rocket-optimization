"""Optimises inside the tightened operating envelope.

Ceiling: 500 psi peak chamber pressure, peak Kn 225, peak mass flux
1.05 lb/(in^2 s). The baseline motor already sits at 98-100% of all three, so
unlike the earlier study there is no pressure headroom to spend -- the gain has
to come from flattening the Kn curve so that *initial* Kn can sit near the
ceiling instead of 7% below it, and from fixing the over-expanded nozzle.

Three core arrangements are run, because "the forward core must not be wider
than the aft core" admits more than one reading:

* ``free``    -- non-decreasing aft-ward; equal cores allowed (the baseline
                 itself uses three pairs, so ties match current practice)
* ``strict``  -- each core at least 1 mm wider than the one ahead of it
* ``paired``  -- three mandrel sizes used in pairs, exactly how the current
                 motor is built
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rocketopt.design import DesignSpace, SpaceConfig
from rocketopt.optimize import (Objective, direct_search, pareto_indices,
                                scale_constraints, surrogate_pareto)
from rocketopt.sampling import evaluate_batch
from rocketopt.ric import load_ric, save_ric
from rocketopt.simulate import PA_PER_PSI, simulate_motor
from rocketopt.surrogate import Surrogate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
KG_PER_LB_IN2 = 703.0696
SEARCH_DT = 0.02    # matches the surrogate's training data
VERIFY_DT = 0.002   # what every reported number is measured at
FLUX_MARGIN = 0.03  # covers the timestep drift in peak mass flux
ENVELOPE = dict(max_pressure=500 * PA_PER_PSI,
                max_mass_flux=1.05 * KG_PER_LB_IN2,
                max_kn=225.0,
                exit_max=0.076)
ARRANGEMENTS = {
    "free": {},
    "strict": {"min_core_step": 0.001},
    "paired": {"core_groups": (2, 2, 2)},
}


def describe(space, x, label):
    metrics = simulate_motor(space.to_motor(x), timestep=0.002)
    motor = space.to_motor(x)
    return {
        "arrangement": label,
        "designation": metrics.designation,
        "initial_thrust": metrics.initial_thrust,
        "total_impulse": metrics.total_impulse,
        "peak_thrust": metrics.peak_thrust,
        "isp": metrics.isp,
        "burn_time": metrics.burn_time,
        "max_pressure_psi": metrics.max_pressure / PA_PER_PSI,
        "initial_kn": metrics.initial_kn,
        "peak_kn": metrics.peak_kn,
        "mass_flux_lb": metrics.peak_mass_flux / KG_PER_LB_IN2,
        "port_throat": metrics.port_throat,
        "prop_mass": metrics.prop_mass,
        "thrust_variation": metrics.thrust_variation,
        "warnings": "; ".join(metrics.warnings),
        "cores_mm": [round(1000 * c, 2) for c in x[: space.n_grains]],
        "throat_mm": round(1000 * float(x[space.n_grains]), 2),
        "exit_mm": round(1000 * motor["nozzle"]["exit"], 2),
        "x": [float(v) for v in x],
    }


def main() -> None:
    base = load_ric(ROOT / "Data" / "Open Motor Data" / "Current.ric")
    reference_space = DesignSpace(base, SpaceConfig(**ENVELOPE))
    baseline = simulate_motor(base, timestep=0.002)
    objective_base = dict(baseline_thrust=baseline.initial_thrust,
                          baseline_impulse=baseline.total_impulse)
    print("baseline: {:.0f} N initial | {:.0f} N·s | {:.0f} psi | Kn {:.1f}->{:.1f} | "
          "flux {:.3f} lb/in²s".format(
              baseline.initial_thrust, baseline.total_impulse,
              baseline.max_pressure / PA_PER_PSI, baseline.initial_kn,
              baseline.peak_kn, baseline.peak_mass_flux / KG_PER_LB_IN2))

    dataset = pd.read_parquet(OUT / "data" / "designs.parquet")
    pool = dataset[dataset["ok"]].reset_index(drop=True)

    results, fronts = [], []
    for label, overrides in ARRANGEMENTS.items():
        space = DesignSpace(base, SpaceConfig(**ENVELOPE, **overrides))
        surrogate = Surrogate.load(OUT / "models", space)
        search_obj = Objective(0.7, 0.3, flux_margin=FLUX_MARGIN, **objective_base)
        strict_obj = search_obj.strict()

        # The stored dataset was built under a different envelope and a wider
        # nozzle bound, so its design vectors mean different motors here. Take
        # the vectors as starting points only and re-simulate them under this
        # space rather than trusting a single stored metric.
        scored = pool.assign(_s=search_obj.score(
            pool["initial_thrust"], pool["total_impulse"]))
        candidates = space.canonicalize(
            scored.nlargest(400, "_s")[space.names].to_numpy(dtype=float))
        reference = evaluate_batch(space, candidates, timestep=VERIFY_DT, workers=12)
        seeds = np.vstack([
            space.canonical_one(space.from_motor(base))[None, :],
            candidates[:60],
        ])

        ga = direct_search(space, search_obj, pop_size=96, n_gen=45,
                           timestep=SEARCH_DT, workers=12, seed=17)
        for x, method in [(ga["x"], "GA (simulator)")]:
            row = describe(space, x, label)
            row["method"] = method
            results.append(row)

        run = surrogate_pareto(space, surrogate, search_obj, pop_size=200,
                               n_gen=180, timestep=VERIFY_DT, workers=12, seed=17,
                               seed_designs=seeds, reference=reference)
        front = run["front"]
        if len(front):
            front = front.copy()
            front["arrangement"] = label
            fronts.append(front)
            keeps = front[front["total_impulse"] >= baseline.total_impulse]
            pick = (keeps if len(keeps) else front).nlargest(1, "initial_thrust").iloc[0]
            x = space.canonical_one(pick[space.names].to_numpy(dtype=float))
            best = describe(space, x, label)
            best["method"] = "NSGA-II front"
            results.append(best)

        # Nothing is reported or written out unless it clears every limit at the
        # verification timestep with the safety margin removed.
        for row in [r for r in results if r["arrangement"] == label]:
            check = pd.DataFrame([{
                "ok": True, "max_pressure": row["max_pressure_psi"] * PA_PER_PSI,
                "peak_mass_flux": row["mass_flux_lb"] * KG_PER_LB_IN2,
                "port_throat": row["port_throat"], "peak_kn": row["peak_kn"],
                "avg_pressure": space.config.min_chamber_pressure * 2,
            }])
            worst = float(scale_constraints(check, strict_obj, space).max())
            row["feasible"] = worst <= 0.0
            row["worst_violation"] = worst
            if row["feasible"] and row["method"] == "NSGA-II front":
                save_ric(OUT / "motors" / "envelope_{}.ric".format(label),
                         space.to_motor(np.array(row["x"])))
            print("  {:7s} {:14s} {:>5.0f} N ({:+5.1f}%)  {:>6.0f} N·s ({:+5.1f}%)  "
                  "{:>3.0f} psi  Kn {:.0f}->{:.0f}  flux {:.3f}  {:8s} {}".format(
                      label, row["method"], row["initial_thrust"],
                      100 * (row["initial_thrust"] / baseline.initial_thrust - 1),
                      row["total_impulse"],
                      100 * (row["total_impulse"] / baseline.total_impulse - 1),
                      row["max_pressure_psi"], row["initial_kn"], row["peak_kn"],
                      row["mass_flux_lb"], row["designation"],
                      "OK" if row["feasible"] else
                      "VIOLATES (g={:+.3f})".format(worst)))

    if fronts:
        pd.concat(fronts, ignore_index=True).to_parquet(
            OUT / "data" / "envelope_fronts.parquet", index=False)
    (OUT / "data" / "envelope_results.json").write_text(json.dumps(
        {"baseline": describe(reference_space,
                              reference_space.from_motor(base), "baseline"),
         "designs": results}, indent=2))


if __name__ == "__main__":
    main()
