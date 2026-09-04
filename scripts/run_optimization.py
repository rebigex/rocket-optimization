"""Runs every optimiser, verifies the winners, and writes .ric files.

Fidelity discipline: the search and the surrogate both work at the 0.02 s
timestep the training data used, so model error is never confounded with
solver error. Only the handful of designs actually recommended are re-run at
0.002 s, and those are the numbers reported -- including for the baseline, so
the comparison is like for like.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rocketopt.design import DesignSpace
from rocketopt.optimize import (Objective, bayes_optimize, best_design,
                                direct_search, scale_constraints,
                                surrogate_pareto)
from rocketopt.ric import load_ric, save_ric
from rocketopt.sampling import evaluate_batch
from rocketopt.simulate import PA_PER_PSI, simulate_motor
from rocketopt.surrogate import Surrogate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEARCH_DT = 0.02   # matches the training data
FINAL_DT = 0.002   # high-fidelity confirmation of recommended designs
WORKERS = 12
MARGINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def summarise(space, x, label, timestep=FINAL_DT):
    metrics = simulate_motor(space.to_motor(x), timestep=timestep)
    return {
        "label": label,
        "designation": metrics.designation,
        "initial_thrust": metrics.initial_thrust,
        "total_impulse": metrics.total_impulse,
        "peak_thrust": metrics.peak_thrust,
        "avg_thrust": metrics.avg_thrust,
        "isp": metrics.isp,
        "burn_time": metrics.burn_time,
        "max_pressure": metrics.max_pressure,
        "max_pressure_psi": metrics.max_pressure / PA_PER_PSI,
        "peak_mass_flux": metrics.peak_mass_flux,
        "port_throat": metrics.port_throat,
        "prop_mass": metrics.prop_mass,
        "thrust_variation": metrics.thrust_variation,
        "warnings": "; ".join(metrics.warnings),
        "cores_mm": [round(1000 * c, 2) for c in x[: space.n_grains]],
        "throat_mm": round(1000 * float(x[space.n_grains]), 2),
        "exit_mm": round(1000 * space.to_motor(x)["nozzle"]["exit"], 2),
        "x": [float(v) for v in x],
    }


def main() -> None:
    base_motor = load_ric(ROOT / "Data" / "Open Motor Data" / "Current.ric")
    space = DesignSpace(base_motor)
    x_base = space.from_motor(base_motor)

    baseline_search = simulate_motor(space.to_motor(x_base), timestep=SEARCH_DT)
    objective = Objective(
        thrust_weight=0.7, impulse_weight=0.3,
        baseline_thrust=baseline_search.initial_thrust,
        baseline_impulse=baseline_search.total_impulse,
    )
    baseline = summarise(space, x_base, "baseline (Current.ric)")
    print("baseline: {} | initial thrust {:.0f} N | impulse {:.0f} N·s | peak {:.0f} psi".format(
        baseline["designation"], baseline["initial_thrust"],
        baseline["total_impulse"], baseline["max_pressure_psi"]))

    results = {"baseline": baseline}

    # ---- ground truth: genetic search straight against openMotor -----------
    started = time.time()
    direct = direct_search(space, objective, pop_size=96, n_gen=45,
                           timestep=SEARCH_DT, workers=WORKERS, seed=3)
    print("\ndirect GA: {} simulations in {:.0f}s".format(
        direct["n_simulations"], time.time() - started))
    results["direct_ga"] = summarise(space, direct["x"], "direct GA (ground truth)")
    direct["history"].to_parquet(OUT / "data" / "history_direct.parquet", index=False)

    # ---- sample-efficient: Bayesian optimisation ---------------------------
    started = time.time()
    bayes = bayes_optimize(space, objective, n_init=48, n_iter=26, batch=8,
                           timestep=SEARCH_DT, workers=WORKERS, seed=3)
    print("bayesian: {} simulations in {:.0f}s".format(
        bayes["n_simulations"], time.time() - started))
    results["bayesian"] = summarise(space, bayes["x"], "Bayesian optimisation")
    bayes["history"].to_parquet(OUT / "data" / "history_bayes.parquet", index=False)

    # ---- how much pressure are you willing to run? -------------------------
    sweep = []
    for fraction in MARGINS:
        capped = Objective(0.7, 0.3, objective.baseline_thrust,
                           objective.baseline_impulse, pressure_fraction=fraction)
        run = direct_search(space, capped, pop_size=64, n_gen=30,
                            timestep=SEARCH_DT, workers=WORKERS, seed=5)
        if run["x"] is None:
            continue
        row = summarise(space, run["x"], "{:.0f}% of pressure limit".format(100 * fraction))
        row["pressure_fraction"] = fraction
        sweep.append(row)
        print("  {:>4.0f}% limit -> {:>5.0f} N initial, {:>6.0f} N·s, {:>5.0f} psi  {}".format(
            100 * fraction, row["initial_thrust"], row["total_impulse"],
            row["max_pressure_psi"], row["designation"]))
    results["margin_sweep"] = sweep

    # ---- the trade-off, mapped on the surrogate then verified --------------
    surrogate = Surrogate.load(OUT / "models", space)
    started = time.time()
    pareto = surrogate_pareto(space, surrogate, objective, pop_size=200,
                              n_gen=180, timestep=SEARCH_DT, workers=WORKERS, seed=3)
    print("\npareto: {} verified designs, {} on the front, {:.0f}s".format(
        len(pareto.get("verified", [])), len(pareto["front"]), time.time() - started))
    if len(pareto["front"]):
        pareto["front"].to_parquet(OUT / "data" / "pareto_front.parquet", index=False)
        pareto["verified"].to_parquet(OUT / "data" / "pareto_verified.parquet", index=False)

    # ---- write the winners out as motors openMotor can open ---------------
    ric_dir = OUT / "motors"
    ric_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, name in [("direct_ga", "optimized_best"), ("bayesian", "optimized_bayesian")]:
        path = save_ric(ric_dir / "{}.ric".format(name),
                        space.to_motor(np.array(results[key]["x"])))
        written.append(str(path.relative_to(ROOT)))
    for row in sweep:
        path = save_ric(
            ric_dir / "margin_{:03.0f}pct.ric".format(100 * row["pressure_fraction"]),
            space.to_motor(np.array(row["x"])))
        written.append(str(path.relative_to(ROOT)))
    results["ric_files"] = written

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote {} motor files and outputs/results.json".format(len(written)))


if __name__ == "__main__":
    main()
