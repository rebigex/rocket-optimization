"""Optimisers over the motor design space.

Three of them, because they answer different questions:

* :func:`direct_search` runs a genetic algorithm against openMotor itself. It
  needs thousands of simulations but its answer is ground truth, so it is the
  yardstick the learned methods are measured against.
* :func:`bayes_optimize` fits a Gaussian process to the designs it has seen and
  spends each new simulation where expected improvement is highest. This is the
  sample-efficient option -- it is trying to find the same optimum in far fewer
  burns.
* :func:`surrogate_pareto` runs NSGA-II entirely against the trained surrogate,
  which is fast enough to map the whole initial-thrust/impulse trade-off, then
  re-simulates every point on the front so nothing reported is model output.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .design import DesignSpace
from .sampling import SimulationPool, evaluate_batch, sobol_designs

PA_PER_PSI = 6894.757293168361


@dataclass
class Objective:
    """What counts as better, and what may never be exceeded.

    Two modes share one class so the study scripts and the app can use the same
    optimisers. Left alone it behaves as it always has: a weighted blend of
    initial thrust and total impulse, each divided by the baseline motor's value
    so the weights are dimensionless and mean what they say. Give it
    ``objectives``/``constraints`` and it instead follows whatever the user
    picked in the GUI.
    """

    thrust_weight: float = 0.7
    impulse_weight: float = 0.3
    baseline_thrust: float = 1.0
    baseline_impulse: float = 1.0
    #: Fraction of the pressure limit the design is allowed to reach.
    pressure_fraction: float = 1.0
    #: Search-time safety margin on the mass flux limit only. Peak mass flux is
    #: a finite-difference quantity, so it creeps up as the timestep shrinks
    #: (~1.5% from 0.02 s to 0.002 s) while pressure and Kn are timestep
    #: invariant. Searching against a slightly tighter flux limit keeps designs
    #: that sit on the boundary from failing when they are finally verified.
    flux_margin: float = 0.0

    #: User-chosen goals and limits. Empty tuples mean the legacy behaviour.
    objectives: Tuple = ()
    constraints: Tuple = ()
    #: metric -> baseline value, used to normalise objectives against each other.
    baselines: Dict[str, float] = field(default_factory=dict)

    def strict(self) -> "Objective":
        """The same objective with every safety margin removed, for verification."""
        stripped = tuple(replace(c, margin=0.0) for c in self.constraints)
        return replace(self, flux_margin=0.0, constraints=stripped)

    # ------------------------------------------------------------- scoring

    def score(self, initial_thrust, total_impulse) -> np.ndarray:
        """Legacy two-goal blend. Kept because the study scripts call it, and
        one of them subclasses it to add an impulse penalty."""
        return (
            self.thrust_weight * np.asarray(initial_thrust) / self.baseline_thrust
            + self.impulse_weight * np.asarray(total_impulse) / self.baseline_impulse
        )

    @property
    def n_obj(self) -> int:
        return max(1, len(self.objectives)) if self.objectives else 1

    def _normalise(self, frame: pd.DataFrame, spec) -> np.ndarray:
        """One objective as a quantity to maximise, scaled to roughly unity.

        Dividing by the baseline motor's value puts every goal on the same
        footing, so a weight of 1.0 on impulse really does mean the same as a
        weight of 1.0 on thrust despite the units being unrelated.
        """
        values = frame[spec.metric].to_numpy(dtype=float)
        reference = abs(self.baselines.get(spec.metric) or 0.0)
        if reference < 1e-12:
            reference = max(np.nanmax(np.abs(values)), 1e-12)
        if spec.direction == "min":
            return -values / reference
        if spec.direction == "target":
            target = float(spec.target if spec.target is not None else 0.0)
            return -np.abs(values - target) / reference
        return values / reference

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Single number to maximise, for the genetic search."""
        if not self.objectives:
            return self.score(frame["initial_thrust"], frame["total_impulse"])
        total = np.zeros(len(frame), dtype=float)
        for spec in self.objectives:
            total += float(spec.weight) * self._normalise(frame, spec)
        return total

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Objective columns for NSGA-II, already sign-flipped to minimise."""
        if not self.objectives:
            return np.column_stack([
                -frame["initial_thrust"].to_numpy(dtype=float),
                -frame["total_impulse"].to_numpy(dtype=float),
            ])
        return np.column_stack([-self._normalise(frame, s) for s in self.objectives])

    @property
    def objective_labels(self) -> List[str]:
        if not self.objectives:
            return ["initial_thrust", "total_impulse"]
        return [s.metric for s in self.objectives]


def scale_constraints(frame: pd.DataFrame, objective: Objective,
                      space: DesignSpace) -> np.ndarray:
    """Constraint matrix in ``g <= 0`` form, recomputed from raw metrics.

    Deliberately ignores any stored ``g_*`` columns. Those are written at
    sampling time against whatever envelope was in force then, so a dataset
    assembled across two different envelopes carries constraint values that
    disagree with each other -- and merging it would silently admit designs at
    nearly double the current flux limit. Deriving from the measured
    quantities against the live space makes that failure impossible.
    """
    n = len(frame)
    if objective.constraints:
        return _spec_constraints(frame, objective.constraints)
    if n == 0:
        return np.zeros((0, 5))
    pressure_cap = space.max_pressure * objective.pressure_fraction
    flux_cap = space.max_mass_flux * (1.0 - objective.flux_margin)
    columns = [
        frame["max_pressure"].to_numpy(dtype=float) / pressure_cap - 1.0,
        frame["peak_mass_flux"].to_numpy(dtype=float) / flux_cap - 1.0,
        space.min_port_throat
        / np.maximum(frame["port_throat"].to_numpy(dtype=float), 1e-9) - 1.0,
        space.config.min_chamber_pressure
        / np.maximum(frame["avg_pressure"].to_numpy(dtype=float), 1e-9) - 1.0,
        (frame["peak_kn"].to_numpy(dtype=float) / space.max_kn - 1.0)
        if space.max_kn else np.full(n, -1.0),
    ]
    violations = np.column_stack(columns)
    violations[~frame["ok"].to_numpy(dtype=bool)] = 1.0
    return violations


def _spec_constraints(frame: pd.DataFrame, constraints) -> np.ndarray:
    """User-chosen limits, normalised so their magnitudes are comparable.

    Each is expressed as a fractional overshoot of its own limit, which lets
    pymoo weigh a pressure violation against a mass-flux one without either
    drowning the other just because pascals are big numbers.
    """
    active = [c for c in constraints if getattr(c, "enabled", True)]
    n = len(frame)
    if not active:
        return np.full((n, 1), -1.0)
    if n == 0:
        return np.zeros((0, len(active)))
    columns = []
    for spec in active:
        values = frame[spec.metric].to_numpy(dtype=float)
        limit = float(spec.value)
        margin = float(getattr(spec, "margin", 0.0) or 0.0)
        if spec.op == "<=":
            cap = limit * (1.0 - margin)
            columns.append(values / cap - 1.0 if abs(cap) > 1e-12
                           else values - cap)
        else:
            floor = limit * (1.0 + margin)
            columns.append(floor / np.maximum(values, 1e-12) - 1.0 if abs(floor) > 1e-12
                           else floor - values)
    violations = np.column_stack(columns)
    violations[~frame["ok"].to_numpy(dtype=bool)] = 1.0
    return violations


def n_constraints(objective: Objective) -> int:
    """How many constraint columns ``scale_constraints`` will return."""
    if objective.constraints:
        return max(1, len([c for c in objective.constraints
                           if getattr(c, "enabled", True)]))
    return 5


def _callback_kwargs(callback) -> Dict:
    """pymoo replaces a missing callback with a no-op but will happily try to
    call an explicit ``None``, so the argument has to be omitted entirely."""
    return {"callback": callback} if callback is not None else {}


class _SimulatorProblem(Problem):
    """pymoo problem backed by real openMotor runs, evaluated in parallel."""

    def __init__(self, space: DesignSpace, objective: Objective, timestep: float,
                 workers: Optional[int], history: List[pd.DataFrame],
                 pool: Optional[SimulationPool] = None, n_obj: int = 1) -> None:
        super().__init__(
            n_var=space.n_dim, n_obj=n_obj, n_ieq_constr=n_constraints(objective),
            xl=space.lower, xu=space.upper,
        )
        self.space = space
        self.objective = objective
        self.timestep = timestep
        self.workers = workers
        self.history = history
        self.pool = pool

    def _evaluate(self, X, out, *args, **kwargs):
        frame = evaluate_batch(self.space, X, timestep=self.timestep,
                               workers=self.workers, pool=self.pool)
        self.history.append(frame)
        # Kept so a per-generation callback can report what the population
        # actually is, in real units, without re-deriving it from pymoo's
        # normalised objective matrix.
        self.last_frame = frame
        if self.n_obj == 1:
            score = self.objective.score_frame(frame)
            score = np.where(frame["ok"].to_numpy(), score, -1e3)
            out["F"] = (-score).reshape(-1, 1)
        else:
            objectives = self.objective.matrix(frame)
            # A failed simulation must lose on every axis, not just one.
            objectives[~frame["ok"].to_numpy(dtype=bool)] = 1e3
            out["F"] = objectives
        out["G"] = scale_constraints(frame, self.objective, self.space)


def direct_search(
    space: DesignSpace,
    objective: Objective,
    pop_size: int = 96,
    n_gen: int = 40,
    timestep: float = 0.02,
    workers: Optional[int] = None,
    seed: int = 0,
    callback=None,
    seed_designs: Optional[np.ndarray] = None,
) -> Dict:
    """Genetic search straight against the simulator -- the reference answer.

    ``seed_designs`` plants known-good motors in the starting population. With
    several constraints binding at once the feasible region can be thin enough
    that a purely random start spends most of its budget finding its way inside,
    and a run seeded with the user's existing motor cannot come back worse than
    what they already have.
    """
    history: List[pd.DataFrame] = []
    with SimulationPool(space, timestep=timestep, workers=workers) as pool:
        problem = _SimulatorProblem(space, objective, timestep, workers, history, pool)
        if seed_designs is not None and len(seed_designs):
            n_seed = min(len(seed_designs), max(1, pop_size // 4))
            sampling = np.vstack([
                space.canonicalize(np.atleast_2d(seed_designs)[:n_seed]),
                sobol_designs(space, pop_size - n_seed, seed=seed),
            ])
        else:
            sampling = LHS()
        algorithm = GA(pop_size=pop_size, sampling=sampling, eliminate_duplicates=True)
        result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed,
                          verbose=False, **_callback_kwargs(callback))

    evaluated = pd.concat(history, ignore_index=True)
    return {
        "x": space.canonical_one(np.asarray(result.X, dtype=float)),
        "history": evaluated,
        "n_simulations": len(evaluated),
    }


def direct_pareto(
    space: DesignSpace,
    objective: Objective,
    pop_size: int = 96,
    n_gen: int = 45,
    timestep: float = 0.02,
    workers: Optional[int] = None,
    seed: int = 0,
    verify_timestep: float = 0.002,
    seed_designs: Optional[np.ndarray] = None,
    callback=None,
) -> Dict:
    """NSGA-II against openMotor itself, no surrogate in the loop.

    A BATES burn costs about ten milliseconds, so for a handful of objectives
    the simulator can supply a front directly -- no dataset, no training wait.
    The winners are still re-simulated at the fine timestep, because peak mass
    flux drifts with timestep and a design sitting on that limit would otherwise
    be reported as feasible when it is not.
    """
    history: List[pd.DataFrame] = []
    n_obj = objective.n_obj if objective.objectives else 2
    with SimulationPool(space, timestep=timestep, workers=workers) as pool:
        problem = _SimulatorProblem(space, objective, timestep, workers, history,
                                    pool, n_obj=n_obj)
        if seed_designs is not None and len(seed_designs):
            n_seed = min(len(seed_designs), pop_size // 2)
            sampling = np.vstack([
                space.canonicalize(np.atleast_2d(seed_designs)[:n_seed]),
                sobol_designs(space, pop_size - n_seed, seed=seed),
            ])
        else:
            sampling = LHS()
        algorithm = NSGA2(pop_size=pop_size, sampling=sampling,
                          eliminate_duplicates=True)
        result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed,
                          verbose=False, **_callback_kwargs(callback))

    evaluated = pd.concat(history, ignore_index=True) if history else pd.DataFrame()
    if result.X is None:
        return {"front": pd.DataFrame(), "history": evaluated,
                "n_simulations": len(evaluated)}

    X = space.canonicalize(np.atleast_2d(result.X))
    verified = evaluate_batch(space, X, timestep=verify_timestep, workers=workers)
    violation = scale_constraints(verified, objective.strict(), space).max(axis=1)
    keep = verified[verified["ok"] & (violation <= 0)].reset_index(drop=True)
    if not len(keep):
        return {"front": pd.DataFrame(), "verified": verified, "history": evaluated,
                "n_simulations": len(evaluated)}
    front = keep.iloc[pareto_indices(-objective.matrix(keep))]
    front = front.sort_values(objective.objective_labels[0],
                              ascending=False).reset_index(drop=True)
    return {"front": front, "verified": verified, "history": evaluated,
            "n_simulations": len(evaluated)}


# --------------------------------------------------------------------------
# Bayesian optimisation
# --------------------------------------------------------------------------


def _fit_gp(X: np.ndarray, y: np.ndarray, seed: int = 0) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(1e-4, (1e-8, 1e0))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed
    )
    gp.fit(X, y)
    return gp


def bayes_optimize(
    space: DesignSpace,
    objective: Objective,
    n_init: int = 48,
    n_iter: int = 24,
    batch: int = 8,
    n_candidates: int = 8192,
    timestep: float = 0.02,
    workers: Optional[int] = None,
    seed: int = 0,
) -> Dict:
    """Constrained Bayesian optimisation with expected improvement.

    Two Gaussian processes are fitted each round: one on the objective, one on
    the worst constraint violation. The acquisition is expected improvement
    weighted by the probability that a design is actually feasible, so the
    search does not waste burns on motors that would burst the case.
    """
    rng = np.random.default_rng(seed)
    lower, upper = space.lower, space.upper
    span = upper - lower

    def to_unit(X):
        return (np.atleast_2d(X) - lower) / span

    pool = SimulationPool(space, timestep=timestep, workers=workers)
    pool.__enter__()
    X = sobol_designs(space, n_init, seed=seed)
    frames = [pool.evaluate(X)]

    for _ in range(n_iter):
        seen = pd.concat(frames, ignore_index=True)
        X_seen = space.canonicalize(seen[space.names].to_numpy(dtype=float))
        score = objective.score(seen["initial_thrust"], seen["total_impulse"])
        score = np.where(seen["ok"].to_numpy(), score, np.nanmin(score) - 1.0)
        violation = scale_constraints(seen, objective, space).max(axis=1)

        gp_obj = _fit_gp(to_unit(X_seen), score, seed)
        gp_con = _fit_gp(to_unit(X_seen), violation, seed)

        feasible = violation <= 0
        best = score[feasible].max() if feasible.any() else score.max()

        candidates = space.canonicalize(
            lower + qmc.Sobol(space.n_dim, scramble=True,
                              seed=int(rng.integers(1 << 30))).random(n_candidates) * span
        )
        unit = to_unit(candidates)
        mu, sigma = gp_obj.predict(unit, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        improvement = mu - best
        z = improvement / sigma
        expected_improvement = improvement * norm.cdf(z) + sigma * norm.pdf(z)

        mu_c, sigma_c = gp_con.predict(unit, return_std=True)
        prob_feasible = norm.cdf(-mu_c / np.maximum(sigma_c, 1e-12))

        acquisition = expected_improvement * prob_feasible
        # Take the batch from distinct peaks rather than one cluster.
        order = np.argsort(-acquisition)
        picked: List[int] = []
        for idx in order:
            if len(picked) >= batch:
                break
            if all(np.linalg.norm(unit[idx] - unit[j]) > 0.05 for j in picked):
                picked.append(idx)
        if not picked:
            picked = list(order[:batch])

        frames.append(pool.evaluate(candidates[picked]))

    pool.close()
    seen = pd.concat(frames, ignore_index=True)
    return {"history": seen, "n_simulations": len(seen),
            "x": best_design(seen, space, objective)}


def best_design(frame: pd.DataFrame, space: DesignSpace,
                objective: Objective) -> Optional[np.ndarray]:
    """Highest-scoring feasible row of an evaluation history."""
    violation = scale_constraints(frame, objective, space).max(axis=1)
    ok = frame["ok"].to_numpy() & (violation <= 0)
    if not ok.any():
        return None
    score = objective.score(frame["initial_thrust"], frame["total_impulse"])
    winner = np.flatnonzero(ok)[np.argmax(score[ok])]
    return space.canonical_one(frame.iloc[winner][space.names].to_numpy(dtype=float))


# --------------------------------------------------------------------------
# Multi-objective front on the surrogate
# --------------------------------------------------------------------------


class _SurrogateProblem(Problem):
    """Two-objective problem answered by the trained models, not the simulator."""

    def __init__(self, space: DesignSpace, surrogate, objective: Objective) -> None:
        super().__init__(
            n_var=space.n_dim, n_obj=objective.n_obj if objective.objectives else 2,
            n_ieq_constr=n_constraints(objective),
            xl=space.lower, xu=space.upper,
        )
        self.space = space
        self.surrogate = surrogate
        self.objective = objective

    def _evaluate(self, X, out, *args, **kwargs):
        X = self.space.canonicalize(X)
        pred = self.surrogate.predict(X)
        features = pd.DataFrame(
            self.space.features(X), columns=self.space.feature_names
        )
        if self.objective.objectives or self.objective.constraints:
            # The surrogate stands in for the simulator, so give the shared
            # scoring code a frame shaped exactly like a real evaluation.
            predicted = pred.copy()
            predicted["ok"] = True
            if "avg_pressure" not in predicted:
                predicted["avg_pressure"] = features["pressure_0"].to_numpy()
            out["F"] = self.objective.matrix(predicted)
            out["G"] = scale_constraints(predicted, self.objective, self.space)
            return
        out["F"] = np.column_stack(
            [-pred["initial_thrust"].to_numpy(), -pred["total_impulse"].to_numpy()]
        )
        cap = self.objective.pressure_fraction * self.space.max_pressure
        flux_cap = self.space.max_mass_flux * (1.0 - self.objective.flux_margin)
        out["G"] = np.column_stack(
            [
                pred["max_pressure"].to_numpy() / cap - 1.0,
                pred["peak_mass_flux"].to_numpy() / flux_cap - 1.0,
                self.space.min_port_throat
                / np.maximum(pred["port_throat"].to_numpy(), 1e-9) - 1.0,
                self.space.config.min_chamber_pressure
                / np.maximum(features["pressure_0"].to_numpy(), 1e-9) - 1.0,
                (pred["peak_kn"].to_numpy() / self.space.max_kn - 1.0)
                if self.space.max_kn else np.full(len(X), -1.0),
            ]
        )


def surrogate_pareto(
    space: DesignSpace,
    surrogate,
    objective: Objective,
    pop_size: int = 200,
    n_gen: int = 150,
    timestep: float = 0.005,
    workers: Optional[int] = None,
    seed: int = 0,
    seed_designs: Optional[np.ndarray] = None,
    reference: Optional[pd.DataFrame] = None,
) -> Dict:
    """Maps the initial-thrust / impulse trade-off, then verifies it.

    NSGA-II is run against the surrogate because a dense front needs far more
    evaluations than the simulator could supply. Every surviving design is then
    re-simulated at a fine timestep, and the front is recomputed from those real
    numbers -- so model error can cost us a good design, but can never put a
    design on the reported front that does not deserve to be there.

    Two safeguards against an under-converged front. ``seed_designs`` puts
    known-good motors into the initial population, and ``reference`` -- already
    simulated designs, typically the sampled dataset -- is merged in before the
    non-dominated set is taken. Together they guarantee the reported front is
    never worse than something we already had, which a plain NSGA-II run does
    not: at a tight pressure cap the feasible region is thin and the search can
    miss the knee entirely.
    """
    problem = _SurrogateProblem(space, surrogate, objective)
    if seed_designs is not None and len(seed_designs):
        n_seed = min(len(seed_designs), pop_size // 2)
        initial = np.vstack([
            space.canonicalize(np.atleast_2d(seed_designs)[:n_seed]),
            sobol_designs(space, pop_size - n_seed, seed=seed),
        ])
        sampling = initial
    else:
        sampling = LHS()
    algorithm = NSGA2(pop_size=pop_size, sampling=sampling, eliminate_duplicates=True)
    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)

    if result.X is None:
        return {"front": pd.DataFrame(), "predicted": pd.DataFrame()}

    X = space.canonicalize(np.atleast_2d(result.X))
    predicted = surrogate.predict(X)
    verified = evaluate_batch(space, X, timestep=timestep, workers=workers)
    for column in ("initial_thrust", "total_impulse", "max_pressure", "peak_mass_flux"):
        verified["pred_" + column] = predicted[column].to_numpy()

    if reference is not None and len(reference):
        shared = [c for c in verified.columns if c in reference.columns]
        verified = pd.concat(
            [verified, reference[shared]], ignore_index=True, sort=False
        )

    # Verification is the moment of truth, so the safety margin comes off.
    violation = scale_constraints(verified, objective.strict(), space).max(axis=1)
    verified["max_violation_capped"] = violation
    keep = verified[verified["ok"] & (violation <= 0)].reset_index(drop=True)
    if not len(keep):
        return {"front": pd.DataFrame(), "verified": verified, "predicted": predicted}
    # Rank on whatever the user chose, not on a fixed pair. ``matrix`` already
    # points downhill, so negate it to get quantities to maximise.
    axes = -objective.matrix(keep)
    front = keep.iloc[pareto_indices(axes)]
    front = front.sort_values(objective.objective_labels[0],
                              ascending=False).reset_index(drop=True)
    return {"front": front, "verified": verified, "predicted": predicted}


def pareto_indices(objectives: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows, treating every column as maximised."""
    n = len(objectives)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominators = np.all(objectives >= objectives[i], axis=1) & np.any(
            objectives > objectives[i], axis=1
        )
        if dominators.any():
            keep[i] = False
    return np.flatnonzero(keep)
