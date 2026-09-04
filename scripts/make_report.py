"""Turns the optimisation outputs into figures and a written summary."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rocketopt import plotting
from rocketopt.design import DesignSpace
from rocketopt.optimize import Objective, scale_constraints
from rocketopt.ric import load_ric
from rocketopt.simulate import PA_PER_PSI, thrust_curve
from rocketopt.surrogate import TARGETS, Surrogate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIGS = OUT / "figures"


def parity_frame(space, surrogate, dataset):
    """Rebuilds the same held-out split the surrogate was scored on."""
    usable = dataset[dataset["ok"]].reset_index(drop=True)
    _, idx_test = train_test_split(np.arange(len(usable)), test_size=0.2,
                                   random_state=surrogate.seed)
    test = usable.iloc[idx_test]
    predicted = surrogate.predict(test[space.names].to_numpy(dtype=float))
    frame = pd.DataFrame()
    for target in TARGETS:
        frame["actual_" + target] = test[target].to_numpy()
        frame["pred_" + target] = predicted[target].to_numpy()
    return frame


def running_best(history, objective, space):
    violation = scale_constraints(history, objective, space).max(axis=1)
    score = objective.score(history["initial_thrust"], history["total_impulse"])
    score = np.where(history["ok"].to_numpy() & (violation <= 0), score, -np.inf)
    best = np.maximum.accumulate(score)
    return pd.Series(np.where(np.isfinite(best), best, np.nan))


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    base_motor = load_ric(ROOT / "Data" / "Open Motor Data" / "Current.ric")
    space = DesignSpace(base_motor)
    results = json.loads((OUT / "results.json").read_text())
    baseline = results["baseline"]

    objective = Objective(0.7, 0.3, baseline["initial_thrust"],
                          baseline["total_impulse"])

    # thrust curves -------------------------------------------------------
    curves = {"baseline": thrust_curve(space.to_motor(np.array(baseline["x"])))}
    recommended = results.get("recommended") or results["direct_ga"]
    curves["optimised"] = thrust_curve(space.to_motor(np.array(recommended["x"])))
    plotting.thrust_curves(curves, FIGS / "thrust_curves.png",
                           "Thrust curve: baseline vs optimised")

    # Prefer the exit-constrained run when it exists: those nozzles fit inside
    # the case bore, and they cost almost nothing in performance.
    suffix = "_exit76" if (OUT / "data" / "margin_picks_exit76.json").exists() else ""
    picks_path = OUT / "data" / "margin_picks{}.json".format(suffix)
    if picks_path.exists():
        picks = pd.DataFrame(json.loads(picks_path.read_text()))
        picks["max_pressure"] = picks["max_pressure_psi"] * PA_PER_PSI
        plotting.margin_sweep(picks, FIGS / "margin_sweep.png", baseline)
        mid = picks.iloc[(picks["pressure_fraction"] - 0.6).abs().argmin()]
        top = picks.iloc[(picks["pressure_fraction"] - 1.0).abs().argmin()]
        plotting.thrust_curves(
            {"baseline (491 psi)": curves["baseline"],
             "900 psi cap": thrust_curve(space.to_motor(np.array(mid["x"]))),
             "1500 psi cap": thrust_curve(space.to_motor(np.array(top["x"])))},
            FIGS / "thrust_curves.png",
            "Thrust curve: baseline vs optimised at two pressure ceilings")

    fronts_path = OUT / "data" / "margin_fronts{}.parquet".format(suffix)
    if fronts_path.exists():
        plotting.margin_fronts(pd.read_parquet(fronts_path), baseline,
                               FIGS / "margin_fronts.png")

    # pareto --------------------------------------------------------------
    front_path = OUT / "data" / "pareto_front.parquet"
    if front_path.exists():
        front = pd.read_parquet(front_path)
        cloud = pd.read_parquet(OUT / "data" / "designs.parquet")
        cloud = cloud[cloud["feasible"]]
        plotting.pareto_front(front, baseline, FIGS / "pareto_front.png", cloud=cloud)

    # surrogate accuracy --------------------------------------------------
    surrogate = Surrogate.load(OUT / "models", space)
    dataset = pd.read_parquet(OUT / "data" / "designs.parquet")
    plotting.parity(
        parity_frame(space, surrogate, dataset),
        ["initial_thrust", "total_impulse", "max_pressure",
         "peak_mass_flux", "isp", "burn_time"],
        FIGS / "surrogate_parity.png",
        units={"initial_thrust": "N", "total_impulse": "N·s",
               "max_pressure": "Pa", "peak_mass_flux": "kg/m²s",
               "isp": "s", "burn_time": "s"},
    )
    for target, title in [
        ("initial_thrust", "What drives initial thrust"),
        ("total_impulse", "What drives total impulse"),
        ("peak_mass_flux", "What drives peak mass flux"),
    ]:
        table = pd.read_csv(OUT / "data" / "importance_{}.csv".format(target))
        plotting.importance(table, FIGS / "importance_{}.png".format(target), title)

    # convergence ---------------------------------------------------------
    runs = {}
    for label, name in [("direct GA (ground truth)", "history_direct"),
                        ("Bayesian optimisation", "history_bayes")]:
        path = OUT / "data" / "{}.parquet".format(name)
        if path.exists():
            runs[label] = running_best(pd.read_parquet(path), objective, space)
    if runs:
        plotting.convergence(runs, FIGS / "convergence.png", baseline=1.0)

    print("figures written to {}".format(FIGS))
    for figure in sorted(FIGS.glob("*.png")):
        print("  {}".format(figure.name))


if __name__ == "__main__":
    main()
