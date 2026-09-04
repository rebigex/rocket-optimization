"""Exports the useful designs along the in-envelope Pareto front."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rocketopt.design import DesignSpace, SpaceConfig
from rocketopt.ric import study_motor, load_ric, save_ric
from rocketopt.simulate import PA_PER_PSI, simulate_motor
from run_envelope import ARRANGEMENTS, ENVELOPE, KG_PER_LB_IN2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    base = load_ric(study_motor(ROOT))
    baseline = simulate_motor(base, timestep=0.002)
    front = pd.read_parquet(OUT / "data" / "envelope_fronts.parquet")

    both = front[(front["initial_thrust"] >= baseline.initial_thrust)
                 & (front["total_impulse"] >= baseline.total_impulse)]
    selections = [
        ("thrust_at_equal_impulse", both.nlargest(1, "initial_thrust"),
         "most initial thrust while holding baseline impulse"),
        ("impulse_at_equal_thrust", both.nlargest(1, "total_impulse"),
         "most impulse while holding baseline initial thrust"),
        ("max_thrust", front.nlargest(1, "initial_thrust"),
         "most initial thrust available, impulse traded away"),
        ("max_impulse", front.nlargest(1, "total_impulse"),
         "most impulse available, initial thrust traded away"),
    ]

    rows = []
    for name, chosen, note in selections:
        if not len(chosen):
            continue
        pick = chosen.iloc[0]
        label = pick["arrangement"]
        space = DesignSpace(base, SpaceConfig(**ENVELOPE, **ARRANGEMENTS[label]))
        x = space.canonical_one(pick[space.names].to_numpy(dtype=float))
        metrics = simulate_motor(space.to_motor(x), timestep=0.002)
        motor = space.to_motor(x)
        path = save_ric(OUT / "motors" / "envelope_{}.ric".format(name), motor)
        rows.append({
            "name": name, "note": note, "arrangement": label,
            "designation": metrics.designation,
            "initial_thrust": metrics.initial_thrust,
            "d_thrust": 100 * (metrics.initial_thrust / baseline.initial_thrust - 1),
            "total_impulse": metrics.total_impulse,
            "d_impulse": 100 * (metrics.total_impulse / baseline.total_impulse - 1),
            "psi": metrics.max_pressure / PA_PER_PSI,
            "initial_kn": metrics.initial_kn, "peak_kn": metrics.peak_kn,
            "flux": metrics.peak_mass_flux / KG_PER_LB_IN2,
            "port_throat": metrics.port_throat, "isp": metrics.isp,
            "burn_time": metrics.burn_time, "prop_mass": metrics.prop_mass,
            "cores_mm": [round(1000 * c, 2) for c in x[: space.n_grains]],
            "throat_mm": round(1000 * float(x[space.n_grains]), 2),
            "exit_mm": round(1000 * motor["nozzle"]["exit"], 2),
            "file": str(path.relative_to(ROOT)),
            "x": [float(v) for v in x],
        })

    base_row = {
        "name": "baseline", "note": "Current.ric as built", "arrangement": "paired",
        "designation": baseline.designation,
        "initial_thrust": baseline.initial_thrust, "d_thrust": 0.0,
        "total_impulse": baseline.total_impulse, "d_impulse": 0.0,
        "psi": baseline.max_pressure / PA_PER_PSI,
        "initial_kn": baseline.initial_kn, "peak_kn": baseline.peak_kn,
        "flux": baseline.peak_mass_flux / KG_PER_LB_IN2,
        "port_throat": baseline.port_throat, "isp": baseline.isp,
        "burn_time": baseline.burn_time, "prop_mass": baseline.prop_mass,
        "cores_mm": [40.64, 40.64, 48.26, 48.26, 55.88, 55.88],
        "throat_mm": 33.02, "exit_mm": 57.15, "file": "Data/Open Motor Data/Current.ric",
    }
    (OUT / "data" / "envelope_picks.json").write_text(
        json.dumps({"baseline": base_row, "picks": rows}, indent=2))

    hdr = "{:<26} {:>6} {:>7} {:>7} {:>7} {:>4} {:>4} {:>4} {:>6} {:>5} {:>5} {:>6}"
    print(hdr.format("design", "initF", "d%", "impulse", "d%", "psi", "Kn0", "Knpk",
                     "flux", "ISP", "burn", "throat"))
    for r in [base_row] + rows:
        print(hdr.format(r["name"], "{:.0f}".format(r["initial_thrust"]),
                         "{:+.1f}".format(r["d_thrust"]),
                         "{:.0f}".format(r["total_impulse"]),
                         "{:+.1f}".format(r["d_impulse"]), "{:.0f}".format(r["psi"]),
                         "{:.0f}".format(r["initial_kn"]), "{:.0f}".format(r["peak_kn"]),
                         "{:.3f}".format(r["flux"]), "{:.1f}".format(r["isp"]),
                         "{:.2f}".format(r["burn_time"]),
                         "{:.1f}".format(r["throat_mm"])))
        print("{:<26} cores {}".format("", " · ".join("{:.1f}".format(c) for c in r["cores_mm"])))


if __name__ == "__main__":
    main()
