"""Parallel evaluation of design vectors, and Sobol sampling of the space."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .design import DesignSpace, SpaceConfig
from .simulate import Metrics, constraint_violations, simulate_motor

# Worker-process state. Rebuilding the DesignSpace once per worker keeps the
# per-task payload down to an 8-float vector.
_SPACE: Optional[DesignSpace] = None
_TIMESTEP: float = 0.02


def _init_worker(space_class, space_kwargs: Dict, timestep: float) -> None:
    global _SPACE, _TIMESTEP
    _SPACE = space_class(**space_kwargs)
    _TIMESTEP = timestep
    # Each worker runs one simulation at a time; letting BLAS also thread would
    # oversubscribe the machine badly.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _eval_worker(x: np.ndarray) -> Dict:
    assert _SPACE is not None
    return _evaluate(_SPACE, np.asarray(x, dtype=float), _TIMESTEP)


def _evaluate(space: DesignSpace, x: np.ndarray, timestep: float) -> Dict:
    x = space.canonical_one(x)
    metrics = simulate_motor(space.to_motor(x), timestep=timestep)
    row = metrics.as_row()
    row.update({name: float(value) for name, value in zip(space.names, x)})
    row.update(
        {
            name: float(value)
            for name, value in zip(space.feature_names, space.features(x)[0])
        }
    )
    violations = constraint_violations(metrics, space)
    (row["g_pressure"], row["g_massflux"], row["g_portthroat"],
     row["g_minpressure"], row["g_kn"]) = (float(v) for v in violations)
    row["max_violation"] = float(violations.max())
    row["feasible"] = bool(metrics.ok and violations.max() <= 0.0)
    return row


class SimulationPool:
    """A worker pool held open across many batches.

    An evolutionary run calls the simulator once per generation. Building a
    fresh pool each time costs roughly two seconds of process spawn and imports
    -- more than the generation's simulations themselves -- so a search of a few
    hundred generations spends most of its wall clock starting processes.
    Holding one pool open for the whole run removes that entirely.
    """

    def __init__(self, space: DesignSpace, timestep: float = 0.02,
                 workers: Optional[int] = None) -> None:
        self.space = space
        self.timestep = timestep
        self.workers = workers if workers is not None else max(
            1, (os.cpu_count() or 2) - 2
        )
        self._pool: Optional[ProcessPoolExecutor] = None

    def __enter__(self) -> "SimulationPool":
        if self.workers > 1:
            space_class, space_kwargs = self.space.worker_spec()
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_init_worker,
                initargs=(space_class, space_kwargs, self.timestep),
            )
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def evaluate(self, X: np.ndarray) -> pd.DataFrame:
        X = self.space.canonicalize(np.atleast_2d(np.asarray(X, dtype=float)))
        if self._pool is None or len(X) < 2 * self.workers:
            rows: List[Dict] = [
                _evaluate(self.space, x, self.timestep) for x in X
            ]
        else:
            chunk = max(1, len(X) // (self.workers * 4))
            rows = list(self._pool.map(_eval_worker, X, chunksize=chunk))
        return pd.DataFrame(rows)


def evaluate_batch(
    space: DesignSpace,
    X: np.ndarray,
    timestep: float = 0.02,
    workers: Optional[int] = None,
    pool: Optional[SimulationPool] = None,
) -> pd.DataFrame:
    """Simulates every row of ``X``, reusing ``pool`` when one is supplied."""
    if pool is not None:
        return pool.evaluate(X)
    with SimulationPool(space, timestep=timestep, workers=workers) as owned:
        return owned.evaluate(X)


# --- simulating arbitrary motors, not design vectors -----------------------
# Tolerance analysis perturbs a motor directly rather than moving through the
# design space, so it needs a pool that takes motor dicts.

_MOTOR_TIMESTEP: float = 0.01


def _init_motor_worker(timestep: float) -> None:
    global _MOTOR_TIMESTEP
    _MOTOR_TIMESTEP = timestep
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _simulate_motor_worker(motor: Dict):
    return simulate_motor(motor, timestep=_MOTOR_TIMESTEP)


def simulate_many(motors: List[Dict], timestep: float = 0.01,
                  workers: Optional[int] = None) -> List:
    """Simulates a list of complete motors in parallel."""
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    if workers <= 1 or len(motors) < 2 * workers:
        return [simulate_motor(m, timestep=timestep) for m in motors]
    chunk = max(1, len(motors) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_motor_worker,
                             initargs=(timestep,)) as pool:
        return list(pool.map(_simulate_motor_worker, motors, chunksize=chunk))


def sobol_designs(space: DesignSpace, n: int, seed: int = 0) -> np.ndarray:
    """Draws ``n`` canonical designs from a scrambled Sobol sequence.

    Sobol gives far more even coverage of an 8-dimensional box than uniform
    random draws, which matters because the surrogate is only as good as the
    corners of the space it has seen.
    """
    sampler = qmc.Sobol(d=space.n_dim, scramble=True, seed=seed)
    unit = sampler.random(n)
    X = space.lower + unit * (space.upper - space.lower)
    return space.canonicalize(X)


def generate_dataset(
    space: DesignSpace,
    n: int,
    timestep: float = 0.02,
    seed: int = 0,
    workers: Optional[int] = None,
) -> pd.DataFrame:
    X = sobol_designs(space, n, seed=seed)
    return evaluate_batch(space, X, timestep=timestep, workers=workers)


def structured_designs(space: DesignSpace, n: int, seed: int = 0) -> np.ndarray:
    """Designs whose grains repeat a few distinct core diameters.

    Sorting the cores makes "all six grains identical" and "three pairs" -- the
    stacks people actually cast, including the baseline motor -- measure-zero
    under plain Sobol sampling. Drawing them explicitly keeps the surrogate
    honest in the region a builder is most likely to care about.
    """
    rng = np.random.default_rng(seed)
    n_grains = space.n_grains
    lower, upper = space.lower, space.upper
    designs = np.empty((n, space.n_dim), dtype=float)

    # 1 distinct core = uniform stack, 2 or 3 = a stepped stack like the baseline.
    n_distinct = rng.choice([1, 2, 3], size=n, p=[0.4, 0.3, 0.3])
    for i, k in enumerate(n_distinct):
        values = np.sort(rng.uniform(lower[0], upper[0], size=k))
        repeats = _split_evenly(n_grains, int(k))
        designs[i, :n_grains] = np.repeat(values, repeats)
    designs[:, n_grains] = rng.uniform(lower[n_grains], upper[n_grains], size=n)
    designs[:, n_grains + 1] = rng.uniform(0.0, 1.0, size=n)
    return space.canonicalize(designs)


def _split_evenly(total: int, parts: int) -> List[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def mixed_designs(
    space: DesignSpace, n: int, seed: int = 0, structured_fraction: float = 0.35
) -> np.ndarray:
    """Sobol coverage of the full space plus a slice of buildable stacks."""
    n_structured = int(round(n * structured_fraction))
    n_sobol = n - n_structured
    parts = [sobol_designs(space, n_sobol, seed=seed)]
    if n_structured:
        parts.append(structured_designs(space, n_structured, seed=seed + 1))
    return np.vstack(parts)


def generate_mixed_dataset(
    space: DesignSpace,
    n: int,
    timestep: float = 0.02,
    seed: int = 0,
    workers: Optional[int] = None,
    structured_fraction: float = 0.35,
) -> pd.DataFrame:
    X = mixed_designs(space, n, seed=seed, structured_fraction=structured_fraction)
    return evaluate_batch(space, X, timestep=timestep, workers=workers)


def rejection_designs(
    space: DesignSpace,
    n: int,
    predicate,
    seed: int = 0,
    oversample: int = 40,
) -> np.ndarray:
    """Designs kept only if their closed-form features pass ``predicate``.

    Retargeting the envelope moves the interesting region of the space. Because
    the screening features cost nothing to compute, candidates can be filtered
    before any of them is simulated -- so a dataset can be concentrated where
    the constraints now bind without wasting burns elsewhere.
    """
    keep = []
    drawn = 0
    for block in range(oversample):
        batch = mixed_designs(space, max(n, 1024), seed=seed + 100 * block)
        drawn += len(batch)
        mask = predicate(pd.DataFrame(space.features(batch),
                                      columns=space.feature_names))
        keep.append(batch[np.asarray(mask, dtype=bool)])
        if sum(len(k) for k in keep) >= n:
            break
    found = np.vstack(keep)[:n]
    return found
