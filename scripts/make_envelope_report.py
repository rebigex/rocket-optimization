"""Regenerates the envelope figures and the HTML report.

Every number in the report is re-derived here by loading the exported .ric
files and simulating them, so the page cannot drift from the motors shipped
alongside it.
"""
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from rocketopt import plotting
from rocketopt.ric import load_ric
from rocketopt.simulate import PA_PER_PSI, simulate_motor, thrust_curve
from run_envelope import KG_PER_LB_IN2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
VERIFY_DT = 0.002


def measure(path, name, note):
    motor = load_ric(path)
    m = simulate_motor(motor, VERIFY_DT)
    return {
        "name": name, "note": note, "designation": m.designation,
        "initial_thrust": m.initial_thrust, "total_impulse": m.total_impulse,
        "psi": m.max_pressure / PA_PER_PSI, "initial_kn": m.initial_kn,
        "peak_kn": m.peak_kn, "flux": m.peak_mass_flux / KG_PER_LB_IN2,
        "isp": m.isp, "burn_time": m.burn_time, "prop_mass": m.prop_mass,
        "port_throat": m.port_throat,
        "cores_mm": [1000 * g["properties"]["coreDiameter"] for g in motor["grains"]],
        "throat_mm": 1000 * motor["nozzle"]["throat"],
        "exit_mm": 1000 * motor["nozzle"]["exit"],
        "file": str(Path(path).relative_to(ROOT)),
        "motor": motor,
    }


def main() -> None:
    base_path = ROOT / "Data" / "Open Motor Data" / "Current.ric"
    baseline = measure(base_path, "baseline", "Current.ric, as built")

    designs = [baseline]
    for key, name, note in [
        ("free", "free", "cores non-decreasing aft-ward, ties allowed"),
        ("strict", "strict", "each core ≥1 mm wider than the one ahead"),
        ("paired", "paired", "three mandrel sizes used in pairs"),
    ]:
        path = OUT / "motors" / "envelope_best_{}.ric".format(key)
        if path.exists():
            designs.append(measure(path, name, note))
    nozzle_only = OUT / "motors" / "envelope_nozzle_only.ric"
    if nozzle_only.exists():
        designs.append(measure(nozzle_only, "nozzle only",
                               "your grains unchanged, exit cone re-cut"))

    fronts = pd.read_parquet(OUT / "data" / "envelope_fronts.parquet")
    plotting.envelope_fronts(fronts, baseline, FIG / "envelope_fronts.png")
    curves = {"baseline": thrust_curve(baseline["motor"])}
    for d in designs:
        if d["name"] == "free":
            curves["optimised (free)"] = thrust_curve(d["motor"])
        if d["name"] == "strict":
            curves["optimised (strict)"] = thrust_curve(d["motor"])
    plotting.thrust_curves(curves, FIG / "envelope_curves.png",
                           "Thrust curve inside your envelope")

    payload = [{k: v for k, v in d.items() if k != "motor"} for d in designs]
    (OUT / "data" / "envelope_report.json").write_text(json.dumps(payload, indent=2))
    print("figures + envelope_report.json refreshed")
    for d in payload:
        print("  {:12s} {:6.0f} N ({:+5.2f}%)  {:6.0f} N·s ({:+5.2f}%)  {:3.0f} psi  "
              "Kn {:.0f}->{:.0f}  flux {:.3f}  ISP {:.1f}".format(
                  d["name"], d["initial_thrust"],
                  100 * (d["initial_thrust"] / baseline["initial_thrust"] - 1),
                  d["total_impulse"],
                  100 * (d["total_impulse"] / baseline["total_impulse"] - 1),
                  d["psi"], d["initial_kn"], d["peak_kn"], d["flux"], d["isp"]))


if __name__ == "__main__":
    main()
