"""Adds designs concentrated near the tightened operating envelope.

The original sweep covered the whole box. With the ceiling moved down to 500 psi
/ Kn 225 / 1.05 lb-in^-2 s^-1, most of that sweep sits far outside the region
that now matters, so this tops it up with designs screened to land near the new
limits before anything is simulated.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from rocketopt.design import DesignSpace, SpaceConfig
from rocketopt.ric import motor_path, load_ric
from rocketopt.sampling import evaluate_batch, rejection_designs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "data"
PSI = 6894.757293168361
ENVELOPE = dict(max_pressure=500 * PSI, max_mass_flux=1.05 * 703.0696, max_kn=225.0)


def main() -> None:
    n_new = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    base = load_ric(motor_path(ROOT))
    space = DesignSpace(base, SpaceConfig(**ENVELOPE))

    # Peak Kn always exceeds initial Kn, so screen on a band that brackets the
    # 225 ceiling from below rather than filtering at it exactly.
    near_envelope = lambda f: (f["kn_0"] > 120) & (f["kn_0"] < 235)

    started = time.time()
    X = rejection_designs(space, n_new, near_envelope, seed=41)
    fresh = evaluate_batch(space, X, timestep=0.02, workers=12)
    print("{} screened designs simulated in {:.0f}s".format(len(fresh), time.time() - started))

    old = pd.read_parquet(OUT / "designs.parquet")
    if "g_kn" not in old.columns:  # recomputed, not re-simulated
        old["g_kn"] = old["peak_kn"] / space.max_kn - 1.0
    combined = pd.concat([old, fresh], ignore_index=True)
    combined.to_parquet(OUT / "designs.parquet", index=False)

    in_env = combined[
        combined["ok"]
        & (combined["max_pressure"] <= space.max_pressure)
        & (combined["peak_kn"] <= space.max_kn)
        & (combined["peak_mass_flux"] <= space.max_mass_flux)
        & (combined["port_throat"] >= space.min_port_throat)
    ]
    print("dataset now {} designs; {} inside the new envelope ({:.1f}%)".format(
        len(combined), len(in_env), 100 * len(in_env) / len(combined)))


if __name__ == "__main__":
    main()
