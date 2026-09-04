"""A Pareto front at each pressure ceiling.

The single scalarised answer per margin hides the trade-off: at a given
pressure cap you can still choose between initial thrust and impulse. Running
the front at each cap is cheap because NSGA-II works against the surrogate --
only the survivors are simulated.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rocketopt.design import DesignSpace, SpaceConfig
from rocketopt.optimize import Objective, surrogate_pareto
from rocketopt.ric import load_ric, save_ric
from rocketopt.simulate import PA_PER_PSI, simulate_motor
from rocketopt.surrogate import Surrogate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MARGINS = [0.4, 0.5, 0.6, 0.7, 0.85, 1.0]


def main() -> None:
    # Optional: cap the nozzle exit diameter. The default 95 mm assumes the exit
    # cone may protrude past the aft closure; pass a smaller value to force the
    # nozzle to fit inside the 82.25 mm case bore.
    exit_max = float(sys.argv[1]) if len(sys.argv) > 1 else SpaceConfig.exit_max
    suffix = "" if exit_max == SpaceConfig.exit_max else "_exit{:.0f}".format(1000 * exit_max)

    base_motor = load_ric(ROOT / "Data" / "Open Motor Data" / "Current.ric")
    space = DesignSpace(base_motor, SpaceConfig(exit_max=exit_max))
    surrogate = Surrogate.load(OUT / "models", space)
    baseline = simulate_motor(space.to_motor(space.from_motor(base_motor)), 0.002)
    objective = Objective(0.7, 0.3, baseline.initial_thrust, baseline.total_impulse)

    # Everything already simulated is fair game as a starting point, so the
    # front can never come back worse than a design we have already seen.
    dataset = pd.read_parquet(OUT / "data" / "designs.parquet")
    baseline_row = pd.DataFrame([{
        **{n: float(v) for n, v in zip(space.names, space.from_motor(base_motor))},
    }])
    reference = dataset[dataset["ok"]].reset_index(drop=True)

    fronts = []
    picks = []
    for fraction in MARGINS:
        capped = Objective(0.7, 0.3, objective.baseline_thrust,
                           objective.baseline_impulse, pressure_fraction=fraction)
        # Seed with the baseline plus the best designs already sampled under
        # this cap, ranked by the objective at hand.
        under_cap = reference[
            reference["max_pressure"] <= fraction * space.max_pressure
        ]
        ranked = under_cap.assign(_score=capped.score(
            under_cap["initial_thrust"], under_cap["total_impulse"]
        )).nlargest(60, "_score")
        seeds = np.vstack([
            space.from_motor(base_motor)[None, :],
            ranked[space.names].to_numpy(dtype=float),
        ]) if len(ranked) else space.from_motor(base_motor)[None, :]

        run = surrogate_pareto(space, surrogate, capped, pop_size=200, n_gen=180,
                               timestep=0.02, workers=12, seed=11,
                               seed_designs=seeds, reference=under_cap)
        front = run["front"]
        if not len(front):
            print("{:.0f}%: no feasible designs".format(100 * fraction))
            continue
        front = front.copy()
        front["pressure_fraction"] = fraction
        fronts.append(front)

        # The most useful single design at this cap: the highest initial thrust
        # among those that also keep at least the baseline's impulse.
        keeps_impulse = front[front["total_impulse"] >= baseline.total_impulse]
        pick = (keeps_impulse if len(keeps_impulse) else front).nlargest(
            1, "initial_thrust").iloc[0]
        x = space.canonical_one(pick[space.names].to_numpy(dtype=float))
        fine = simulate_motor(space.to_motor(x), timestep=0.002)
        picks.append({
            "pressure_fraction": fraction,
            "psi_cap": fraction * space.max_pressure / PA_PER_PSI,
            "designation": fine.designation,
            "initial_thrust": fine.initial_thrust,
            "total_impulse": fine.total_impulse,
            "max_pressure_psi": fine.max_pressure / PA_PER_PSI,
            "isp": fine.isp,
            "burn_time": fine.burn_time,
            "prop_mass": fine.prop_mass,
            "peak_mass_flux": fine.peak_mass_flux,
            "port_throat": fine.port_throat,
            "keeps_baseline_impulse": bool(len(keeps_impulse)),
            "warnings": "; ".join(fine.warnings),
            "cores_mm": [round(1000 * c, 2) for c in x[: space.n_grains]],
            "throat_mm": round(1000 * float(x[space.n_grains]), 2),
            "exit_mm": round(1000 * space.to_motor(x)["nozzle"]["exit"], 2),
            "x": [float(v) for v in x],
        })
        save_ric(OUT / "motors" / "front_{:03.0f}pct{}.ric".format(
            100 * fraction, suffix), space.to_motor(x))
        print("{:>4.0f}% cap ({:>4.0f} psi): {:>3d} designs | best-with-impulse "
              "{:>5.0f} N initial, {:>6.0f} N·s, {:>5.0f} psi actual  {}".format(
                  100 * fraction, fraction * space.max_pressure / PA_PER_PSI,
                  len(front), fine.initial_thrust, fine.total_impulse,
                  fine.max_pressure / PA_PER_PSI, fine.designation))

    if fronts:
        pd.concat(fronts, ignore_index=True).to_parquet(
            OUT / "data" / "margin_fronts{}.parquet".format(suffix), index=False)
    (OUT / "data" / "margin_picks{}.json".format(suffix)).write_text(
        json.dumps(picks, indent=2))
    print("\nbaseline for reference: {:.0f} N initial, {:.0f} N·s, {:.0f} psi".format(
        baseline.initial_thrust, baseline.total_impulse,
        baseline.max_pressure / PA_PER_PSI))


if __name__ == "__main__":
    main()
