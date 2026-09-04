"""Checks the claim that DesignSpace relies on: for a fixed multiset of core
diameters, openMotor's impulse, pressure and burn time do not depend on grain
order, and putting the largest core aft is never worse for mass flux or
port/throat ratio. If this ever fails, sorting the cores stops being free.
"""
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rocketopt.design import DesignSpace
from rocketopt.ric import load_ric
from rocketopt.simulate import simulate_motor

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    space = DesignSpace(load_ric(ROOT / "Data" / "Open Motor Data" / "Current.ric"))
    rng = np.random.default_rng(0)
    order_free = ("total_impulse", "max_pressure", "burn_time", "isp", "prop_mass")
    failures = 0

    for trial in range(6):
        cores = np.sort(rng.uniform(space.lower[0], space.upper[0], size=space.n_grains))
        throat = rng.uniform(space.lower[6], space.upper[6])
        perms = list(permutations(range(space.n_grains)))
        chosen = [perms[0]] + [perms[i] for i in rng.choice(len(perms), 9, replace=False)]

        runs = []
        for perm in chosen:
            x = np.r_[cores[list(perm)], throat, 0.5]
            # to_motor sorts, so build the motor dict by hand to test raw order.
            motor = space.to_motor(x)
            for grain, core in zip(motor["grains"], cores[list(perm)]):
                grain["properties"]["coreDiameter"] = float(core)
            runs.append((perm, simulate_motor(motor, timestep=0.02)))

        reference = runs[0][1]
        for name in order_free:
            values = np.array([getattr(m, name) for _, m in runs])
            spread = values.ptp() / max(abs(values.mean()), 1e-12)
            status = "ok" if spread < 1e-9 else "FAIL"
            if spread >= 1e-9:
                failures += 1
            if trial == 0:
                print("  {:15s} spread across 10 orders: {:.2e}  {}".format(
                    name, spread, status))

        sorted_metrics = runs[0][1]
        flux = np.array([m.peak_mass_flux for _, m in runs])
        port = np.array([m.port_throat for _, m in runs])
        best_flux = sorted_metrics.peak_mass_flux <= flux.min() + 1e-9
        best_port = sorted_metrics.port_throat >= port.max() - 1e-9
        if not (best_flux and best_port):
            failures += 1
        print("trial {}: sorted order gives lowest mass flux {} ({:.1f} vs worst {:.1f}), "
              "highest port/throat {} ({:.2f} vs worst {:.2f})".format(
                  trial, best_flux, flux.min(), flux.max(),
                  best_port, port.max(), port.min()))

    print("\n{} — sorting cores smallest-forward is {}".format(
        "PASS" if failures == 0 else "{} FAILURES".format(failures),
        "lossless" if failures == 0 else "NOT safe"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
