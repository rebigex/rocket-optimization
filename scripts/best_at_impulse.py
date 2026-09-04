"""Maximises initial thrust subject to holding at least the baseline impulse.

The multi-objective front answers this only as well as its sampling density
happens to allow near the baseline point, which made one core arrangement look
worse than the others when it was really just a gap in the front. Optimising
the constrained question directly settles it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from rocketopt.design import DesignSpace, SpaceConfig
from rocketopt.optimize import Objective, direct_search
from rocketopt.ric import motor_path, load_ric, save_ric
from rocketopt.simulate import PA_PER_PSI, simulate_motor
from run_envelope import ARRANGEMENTS, ENVELOPE, KG_PER_LB_IN2, VERIFY_DT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BASE_THRUST, BASE_IMPULSE = 3311.0, 8778.0


class ThrustAtImpulse(Objective):
    """Initial thrust, with a steep penalty for falling under baseline impulse."""

    def score(self, initial_thrust, total_impulse) -> np.ndarray:
        thrust = np.asarray(initial_thrust, dtype=float)
        impulse = np.asarray(total_impulse, dtype=float)
        shortfall = np.maximum(0.0, 1.0 - impulse / self.baseline_impulse)
        return thrust / self.baseline_thrust - 8.0 * shortfall


def main() -> None:
    base = load_ric(motor_path(ROOT))
    print("maximise initial thrust subject to impulse >= {:.0f} N·s\n".format(BASE_IMPULSE))
    for label, overrides in ARRANGEMENTS.items():
        space = DesignSpace(base, SpaceConfig(**ENVELOPE, **overrides))
        objective = ThrustAtImpulse(1.0, 0.0, BASE_THRUST, BASE_IMPULSE, flux_margin=0.03)
        run = direct_search(space, objective, pop_size=96, n_gen=50,
                            timestep=0.02, workers=12, seed=23)
        x = run["x"]
        metrics = simulate_motor(space.to_motor(x), timestep=VERIFY_DT)
        passes = (metrics.max_pressure / PA_PER_PSI <= 500
                  and metrics.peak_kn <= 225
                  and metrics.peak_mass_flux / KG_PER_LB_IN2 <= 1.05
                  and metrics.total_impulse >= BASE_IMPULSE)
        print("  {:7s} initF {:6.0f} ({:+5.2f}%)  I {:6.0f} ({:+5.2f}%)  {:3.0f} psi  "
              "Kn {:.0f}->{:.0f}  flux {:.3f}  {}".format(
                  label, metrics.initial_thrust,
                  100 * (metrics.initial_thrust / BASE_THRUST - 1),
                  metrics.total_impulse,
                  100 * (metrics.total_impulse / BASE_IMPULSE - 1),
                  metrics.max_pressure / PA_PER_PSI, metrics.initial_kn,
                  metrics.peak_kn, metrics.peak_mass_flux / KG_PER_LB_IN2,
                  "OK" if passes else "FAILS CHECK"))
        motor = space.to_motor(x)
        print("          cores {}  throat {:.1f}  exit {:.1f}".format(
            " · ".join("{:.1f}".format(1000 * c) for c in x[: space.n_grains]),
            1000 * float(x[space.n_grains]), 1000 * motor["nozzle"]["exit"]))
        if passes:
            save_ric(OUT / "motors" / "envelope_best_{}.ric".format(label), motor)


if __name__ == "__main__":
    main()
