"""Running openMotor headlessly and reducing a burn to a row of numbers."""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from motorlib.motor import Motor
from motorlib.simResult import SimAlertLevel

# openMotor solves for exit pressure with fsolve, which chatters on
# over-expanded nozzles. The alerts it raises carry the same information, so the
# numpy/scipy noise is not worth propagating.
warnings.filterwarnings("ignore", category=RuntimeWarning)

PA_PER_PSI = 6894.757293168361
#: Thrust is averaged over this window to define "initial thrust". openMotor
#: seeds t=0 with zero force as the ignition point, so that sample is excluded.
INITIAL_THRUST_WINDOW = 0.35


@dataclass
class Metrics:
    """Everything the optimiser or the report might want from one burn."""

    ok: bool = False
    error: str = ""
    initial_thrust: float = 0.0
    ignition_thrust: float = 0.0
    total_impulse: float = 0.0
    peak_thrust: float = 0.0
    avg_thrust: float = 0.0
    thrust_variation: float = 0.0
    isp: float = 0.0
    burn_time: float = 0.0
    max_pressure: float = 0.0
    avg_pressure: float = 0.0
    peak_mass_flux: float = 0.0
    port_throat: float = 0.0
    prop_mass: float = 0.0
    volume_loading: float = 0.0
    initial_kn: float = 0.0
    peak_kn: float = 0.0
    separation_pct: float = 0.0
    designation: str = ""
    n_warnings: int = 0
    warnings: List[str] = field(default_factory=list)
    sim_seconds: float = 0.0

    def as_row(self) -> Dict:
        row = asdict(self)
        row["warnings"] = "; ".join(self.warnings)
        return row


def simulate_motor(motor_dict: Dict, timestep: float | None = None) -> Metrics:
    """Runs one motor and reduces the result to :class:`Metrics`.

    A design that openMotor rejects outright (impossible geometry, and so on)
    comes back with ``ok=False`` rather than raising, so a sampling sweep never
    dies on a bad corner of the space.
    """
    motor_dict = dict(motor_dict)
    if timestep is not None:
        motor_dict["config"] = dict(motor_dict["config"])
        motor_dict["config"]["timestep"] = timestep

    started = time.perf_counter()
    try:
        with np.errstate(all="ignore"):
            result = Motor(motor_dict).runSimulation()
    except Exception as exc:  # geometry the simulator cannot handle at all
        return Metrics(ok=False, error="{}: {}".format(type(exc).__name__, exc),
                       sim_seconds=time.perf_counter() - started)
    elapsed = time.perf_counter() - started

    errors = result.getAlertsByLevel(SimAlertLevel.ERROR)
    if errors or not result.success:
        return Metrics(
            ok=False,
            error="; ".join(alert.description for alert in errors) or "simulation failed",
            sim_seconds=elapsed,
        )

    time_axis = np.asarray(result.channels["time"].getData(), dtype=float)
    force = np.asarray(result.channels["force"].getData(), dtype=float)
    if force.size < 3:
        return Metrics(ok=False, error="burn too short to evaluate", sim_seconds=elapsed)

    # Skip the t=0 ignition seed, which openMotor records as zero thrust.
    live = time_axis > 0.0
    window = live & (time_axis <= INITIAL_THRUST_WINDOW)
    if not window.any():
        window = live & (time_axis <= time_axis[live].min())
    initial_thrust = float(force[window].mean())

    avg_thrust = float(result.getAverageForce())
    peak_thrust = float(force.max())
    warnings_list = [
        alert.description for alert in result.getAlertsByLevel(SimAlertLevel.WARNING)
    ]

    return Metrics(
        ok=True,
        initial_thrust=initial_thrust,
        ignition_thrust=float(force[live][0]) if live.any() else 0.0,
        total_impulse=float(result.getImpulse()),
        peak_thrust=peak_thrust,
        avg_thrust=avg_thrust,
        thrust_variation=peak_thrust / avg_thrust if avg_thrust > 0 else 0.0,
        isp=float(result.getISP()),
        burn_time=float(result.getBurnTime()),
        max_pressure=float(result.getMaxPressure()),
        avg_pressure=float(result.getAveragePressure()),
        peak_mass_flux=float(result.getPeakMassFlux()),
        port_throat=float(result.getPortRatio()),
        prop_mass=float(result.getPropellantMass()),
        volume_loading=float(result.getVolumeLoading()),
        initial_kn=float(result.getInitialKN()),
        peak_kn=float(result.getPeakKN()),
        separation_pct=float(
            result.getPercentBelowThreshold(
                "exitPressure",
                result.motor.config.getProperty("sepPressureRatio")
                * result.motor.config.getProperty("ambPressure"),
            )
        ),
        designation=str(result.getFullDesignation()),
        n_warnings=len(warnings_list),
        warnings=warnings_list,
        sim_seconds=elapsed,
    )


def constraint_violations(metrics: Metrics, space) -> np.ndarray:
    """Constraints in ``g(x) <= 0`` form, normalised so magnitudes compare.

    The limits are the ones already configured in the baseline .ric, so the
    optimiser is held to the same envelope openMotor warns against in the GUI.
    """
    if not metrics.ok:
        return np.ones(5)
    floor = space.config.min_chamber_pressure
    kn_limit = space.max_kn
    return np.array(
        [
            metrics.max_pressure / space.max_pressure - 1.0,
            metrics.peak_mass_flux / space.max_mass_flux - 1.0,
            space.min_port_throat / max(metrics.port_throat, 1e-9) - 1.0,
            floor / max(metrics.avg_pressure, 1e-9) - 1.0,
            (metrics.peak_kn / kn_limit - 1.0) if kn_limit else -1.0,
        ]
    )


def is_feasible(metrics: Metrics, space, tol: float = 0.0) -> bool:
    return bool(metrics.ok and (constraint_violations(metrics, space) <= tol).all())


def curves(motor_dict: Dict, timestep: float = 0.002) -> Dict:
    """Every time series the app plots, from one simulation run.

    Mass flux and mass flow are per-grain in openMotor, which is what makes the
    aft grain's flux the number that matters -- so they come back as a list of
    series rather than one, letting the app show which grain is actually running
    closest to the erosive-burning limit.
    """
    motor_dict = dict(motor_dict)
    motor_dict["config"] = dict(motor_dict["config"])
    motor_dict["config"]["timestep"] = timestep
    with np.errstate(all="ignore"):
        result = Motor(motor_dict).runSimulation()

    def series(name):
        return np.asarray(result.channels[name].getData(), dtype=float)

    def per_grain(name):
        raw = result.channels[name].getData()
        return np.asarray(raw, dtype=float).T if len(raw) else np.zeros((0, 0))

    n_grains = len(motor_dict["grains"])
    return {
        "time": series("time").tolist(),
        "thrust": series("force").tolist(),
        "pressure": series("pressure").tolist(),
        "kn": series("kn").tolist(),
        "exit_pressure": series("exitPressure").tolist(),
        "mass_flux": [row.tolist() for row in per_grain("massFlux")][:n_grains],
        "regression": [row.tolist() for row in per_grain("regression")][:n_grains],
        "ok": bool(result.success),
    }


def thrust_curve(motor_dict: Dict, timestep: float = 0.005):
    """Returns (time, thrust, chamber pressure) arrays for plotting."""
    motor_dict = dict(motor_dict)
    motor_dict["config"] = dict(motor_dict["config"])
    motor_dict["config"]["timestep"] = timestep
    with np.errstate(all="ignore"):
        result = Motor(motor_dict).runSimulation()
    return (
        np.asarray(result.channels["time"].getData(), dtype=float),
        np.asarray(result.channels["force"].getData(), dtype=float),
        np.asarray(result.channels["pressure"].getData(), dtype=float),
    )
