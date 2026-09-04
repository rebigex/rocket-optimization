"""Turns a :class:`~rocketopt.spec.RunSpec` into results the app can draw.

This is the only place that knows how a GUI configuration becomes a design
space, an objective and a search strategy. Everything it calls -- the sampler,
the surrogate, the two optimisers -- is the same machinery the study scripts
use, so the app cannot drift away from the validated pipeline.

One rule holds throughout: **nothing is reported that was not simulated.** The
surrogate may propose, and the genetic search may run at a coarse timestep, but
every design that reaches the user has been re-run in openMotor at the
verification timestep with all safety margins removed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .design import DesignSpace, SpaceConfig
from .optimize import (Objective, direct_pareto, direct_search, pareto_indices,
                       scale_constraints)
from .sampling import evaluate_batch, generate_mixed_dataset
from .simulate import PA_PER_PSI, curves, simulate_motor
from .spec import OPTIMISABLE_METRICS, RunSpec, VariableSpec
from .surrogate import Surrogate
from .units import KG_M2S_PER_LB_IN2S, M_PER_IN

ProgressFn = Callable[[str, float, str], None]

#: Small extra tightening on every limit during the search, on top of the
#: measured timestep correction, because that correction was calibrated on one
#: motor and the search visits many.
SEARCH_SAFETY_MARGIN = 0.005


def _noop(stage: str, fraction: float, message: str) -> None:
    pass


def jsonable(value):
    """numpy and pandas scalars are not JSON; everything here becomes plain."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


# ---------------------------------------------------------------------------
# Building the space and the objective from a spec
# ---------------------------------------------------------------------------


def build_space(spec: RunSpec, base_motor: Dict) -> DesignSpace:
    """Maps GUI variables onto the internal design space.

    The exit nozzle needs translating. A user sets an exit *diameter* range, but
    internally the exit is a fraction of the span between the smallest useful
    exit and the airframe limit -- that is what keeps exit > throat without a
    coupled constraint. So the exit row moves into the SpaceConfig, and its slot
    in the variable list becomes the fraction.
    """
    by_name = {v.name: v for v in spec.variables}
    n_grains = len(base_motor["grains"])
    cores = [by_name["core_{}".format(i + 1)] for i in range(n_grains)]
    throat = by_name["throat"]
    exit_var = by_name.get("exit")
    tl = by_name.get("throat_length")

    exit_min = exit_var.low if exit_var else 0.0
    exit_max = exit_var.high if exit_var else 0.095
    exit_step = exit_var.step if exit_var else 0.0
    exit_fixed = None
    if exit_var is not None and not exit_var.free:
        exit_fixed = (exit_var.fixed_value if exit_var.fixed_value is not None
                      else 0.5 * (exit_var.low + exit_var.high))

    config = SpaceConfig(
        core_min=min(c.low for c in cores),
        core_max=max(c.high for c in cores),
        throat_min=throat.low,
        throat_max=throat.high,
        exit_min=exit_min,
        exit_max=exit_max,
        exit_step=exit_step,
        exit_fixed=exit_fixed,
        core_step=cores[0].step,
        throat_step=throat.step,
        include_throat_length=tl is not None,
        throat_length_min=tl.low if tl else 0.0,
        throat_length_max=tl.high if tl else 0.0,
        throat_length_step=tl.step if tl else 0.0,
    )
    internal = list(cores) + [throat, VariableSpec(
        name="exit_frac", free=exit_fixed is None, low=0.0, high=1.0,
        step=exit_step, fixed_value=None if exit_fixed is None else 0.5,
        unit="m", label="Nozzle exit")]
    if tl is not None:
        internal.append(tl)
    return DesignSpace(base_motor, config, variables=internal,
                       ordering=spec.ordering)


def apply_hardware(base_motor: Dict, grain_diameter: Optional[float] = None,
                   grain_length: Optional[float] = None,
                   grain_count: Optional[int] = None,
                   inhibited_ends: Optional[str] = None) -> Dict:
    """Restates the case hardware on a motor: grain OD, length, count, ends.

    None of these are ever optimised -- they are the tube and the mould you
    already own. Widening the grain leaves the cores alone, but narrowing it can
    strand a core outside its own grain, so cores are pulled back to leave a
    minimum web rather than producing a motor openMotor would reject.
    """
    from .ric import clone as _clone

    motor = _clone(base_motor)
    grains = motor["grains"]
    if grain_count is not None and grain_count != len(grains):
        template = _clone(grains[0])
        if grain_count < len(grains):
            grains = grains[:grain_count]
        else:
            grains = grains + [_clone(template) for _ in range(grain_count - len(grains))]
        motor["grains"] = grains
    for grain in grains:
        props = grain["properties"]
        if grain_diameter is not None:
            props["diameter"] = float(grain_diameter)
        if grain_length is not None:
            props["length"] = float(grain_length)
        if inhibited_ends is not None:
            props["inhibitedEnds"] = inhibited_ends
        min_web = 0.125 * M_PER_IN
        largest_core = props["diameter"] - 2 * min_web
        if props["coreDiameter"] > largest_core:
            props["coreDiameter"] = max(largest_core, props["diameter"] * 0.2)
    return motor


def default_spec(base_motor: Dict) -> RunSpec:
    """A sensible starting configuration read off the motor itself.

    Bounds bracket what the motor already is rather than spanning the whole
    physically possible range, and the limits come from the .ric's own
    ``maxPressure``/``maxMassFlux``/``minPortThroat``. So the first thing a user
    sees is their own motor, legal and unchanged, with room to move.
    """
    from .spec import ConstraintSpec, ObjectiveSpec, OrderingSpec

    grains = base_motor["grains"]
    bore = grains[0]["properties"]["diameter"]
    cores = [g["properties"]["coreDiameter"] for g in grains]
    throat = base_motor["nozzle"]["throat"]
    exit_d = base_motor["nozzle"]["exit"]
    cfg = base_motor["config"]

    # A hundredth of an inch. Finer than that is not a dimension anyone holds
    # on a mandrel or a reamer, so it is the default rather than an option.
    grid = 0.01 * M_PER_IN
    min_web = 0.25 * M_PER_IN
    throat_length = base_motor["nozzle"].get("throatLength", 0.2 * M_PER_IN)

    variables = [
        VariableSpec(name="core_{}".format(i + 1), free=True,
                     low=0.5 * M_PER_IN, high=bore - 2 * min_web,
                     step=grid, fixed_value=core, unit="m",
                     label="Grain {} core".format(i + 1))
        for i, core in enumerate(cores)
    ]
    # Bounds follow the case, not the nozzle that happened to be on it. Anchor
    # them to the old throat and changing the grain size leaves the optimiser
    # boxed into a nozzle sized for hardware you no longer have.
    variables.append(VariableSpec(
        name="throat", free=True, low=min(throat * 0.5, bore * 0.10),
        high=bore * 0.60, step=grid, fixed_value=throat, unit="m",
        label="Nozzle throat"))
    variables.append(VariableSpec(
        name="exit", free=True, low=throat * 1.15, high=bore * 1.05,
        step=grid, fixed_value=exit_d, unit="m", label="Nozzle exit"))
    variables.append(VariableSpec(
        name="throat_length", free=True, low=0.05 * M_PER_IN, high=1.0 * M_PER_IN,
        step=grid, fixed_value=throat_length, unit="m", label="Throat length"))

    constraints = [
        ConstraintSpec(metric="max_pressure", op="<=", value=cfg["maxPressure"],
                       label="Peak chamber pressure"),
        ConstraintSpec(metric="peak_mass_flux", op="<=", value=cfg["maxMassFlux"],
                       label="Peak mass flux"),
        ConstraintSpec(metric="port_throat", op=">=", value=cfg["minPortThroat"],
                       label="Port/throat ratio"),
        ConstraintSpec(metric="avg_pressure", op=">=", value=200 * PA_PER_PSI,
                       label="Mean chamber pressure"),
        ConstraintSpec(metric="peak_kn", op="<=", value=250.0, enabled=False,
                       label="Peak Kn"),
        ConstraintSpec(metric="total_impulse", op=">=", value=0.0, enabled=False,
                       label="Total impulse"),
    ]
    return RunSpec(
        variables=variables,
        objectives=[
            ObjectiveSpec(metric="initial_thrust", direction="max", weight=1.0),
            ObjectiveSpec(metric="total_impulse", direction="max", weight=1.0,
                          enabled=False),
        ],
        constraints=constraints,
        ordering=OrderingSpec(mode="nondecreasing"),
    )


def timestep_bias(base_motor: Dict, search_dt: float, verify_dt: float,
                  metrics=OPTIMISABLE_METRICS) -> Dict[str, float]:
    """How much each metric shifts between search fidelity and verification.

    Peak mass flux is a finite difference, so it grows as the timestep shrinks;
    total impulse is an integral, so it grows too but for the opposite reason.
    Pressure and Kn barely move at all. Rather than guessing a safety margin,
    measure the ratio on the user's own motor and correct the search-time limits
    by it -- then a design sitting on a limit during the search is still sitting
    on it after verification, whichever way that metric leans.
    """
    coarse = simulate_motor(base_motor, timestep=search_dt)
    fine = simulate_motor(base_motor, timestep=verify_dt)
    bias: Dict[str, float] = {}
    for name in metrics:
        c = float(getattr(coarse, name, 0.0) or 0.0)
        f = float(getattr(fine, name, 0.0) or 0.0)
        bias[name] = c / f if abs(f) > 1e-12 and abs(c) > 1e-12 else 1.0
    return bias


def build_objective(spec: RunSpec, baseline_metrics,
                    bias: Optional[Dict[str, float]] = None) -> Objective:
    """Objective and constraints as the user configured them.

    Baseline metric values come along so that objectives in unrelated units can
    be normalised against each other -- a weight of 1 on impulse then means the
    same as a weight of 1 on thrust. Limits are restated at search fidelity via
    ``bias`` so the search is neither optimistic nor needlessly timid.
    """
    baselines = {name: float(getattr(baseline_metrics, name, 0.0) or 0.0)
                 for name in OPTIMISABLE_METRICS}
    bias = bias or {}
    constraints = []
    for spec_c in spec.enabled_constraints:
        data = spec_c.to_dict()
        ratio = bias.get(spec_c.metric, 1.0)
        if ratio and abs(ratio - 1.0) < 0.25:  # ignore anything implausible
            data["value"] = float(spec_c.value) * ratio
        # A little extra room on top of the measured shift, since the ratio was
        # taken on one motor and the search visits many.
        data["margin"] = spec_c.margin or SEARCH_SAFETY_MARGIN
        constraints.append(type(spec_c)(**data))
    return Objective(
        objectives=tuple(spec.enabled_objectives),
        constraints=tuple(constraints),
        baselines=baselines,
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    baseline: Dict = field(default_factory=dict)
    designs: List[Dict] = field(default_factory=list)
    population: List[Dict] = field(default_factory=list)
    convergence: List[Dict] = field(default_factory=list)
    surrogate: Optional[Dict] = None
    constraint_activity: List[Dict] = field(default_factory=list)
    sensitivity: List[Dict] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    #: The configuration this run actually used. Panels draw limit lines from
    #: it, so editing the form afterwards cannot mislabel a finished result.
    spec: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return jsonable({
            "spec": self.spec,
            "baseline": self.baseline,
            "designs": self.designs,
            "population": self.population,
            "convergence": self.convergence,
            "surrogate": self.surrogate,
            "constraint_activity": self.constraint_activity,
            "sensitivity": self.sensitivity,
            "stats": self.stats,
            "messages": self.messages,
        })


def describe_design(space: DesignSpace, x: np.ndarray, spec: RunSpec,
                    label: str, with_curves: bool = False) -> Dict:
    """One design, simulated at the verification timestep."""
    x = space.canonical_one(np.asarray(x, dtype=float))
    motor = space.to_motor(x)
    metrics = simulate_motor(motor, timestep=spec.verify_timestep)
    row = metrics.as_row()
    row.update({
        "label": label,
        "x": [float(v) for v in x],
        "cores": [float(g["properties"]["coreDiameter"]) for g in motor["grains"]],
        "grain_diameter": float(space.grain_diameter),
        "grain_lengths": [float(v) for v in space.grain_lengths],
        "throat": float(motor["nozzle"]["throat"]),
        "exit": float(motor["nozzle"]["exit"]),
        "expansion_ratio": float((motor["nozzle"]["exit"] / motor["nozzle"]["throat"]) ** 2),
        "throat_length": float(motor["nozzle"].get("throatLength", 0.0)),
        "inhibited_ends": motor["grains"][0]["properties"].get("inhibitedEnds", "Neither"),
        "max_pressure_psi": float(metrics.max_pressure / PA_PER_PSI),
        "avg_pressure_psi": float(metrics.avg_pressure / PA_PER_PSI),
        "mass_flux_lb": float(metrics.peak_mass_flux / KG_M2S_PER_LB_IN2S),
    })
    if with_curves:
        row["curves"] = curves(motor, timestep=spec.verify_timestep)
    return row


def _constraint_report(frame: pd.DataFrame, objective: Objective,
                       space: DesignSpace) -> List[Dict]:
    """Which limits are actually doing the work.

    A constraint nobody comes near is not shaping the answer; one that a large
    share of good designs sit right up against is the thing to relax if the
    result disappoints.
    """
    active = [c for c in objective.constraints if c.enabled]
    if not active or not len(frame):
        return []
    violations = scale_constraints(frame, objective.strict(), space)
    report = []
    for i, spec_c in enumerate(active):
        column = violations[:, i]
        feasible = column <= 0
        near = feasible & (column > -0.02)
        report.append({
            "metric": spec_c.metric,
            "label": OPTIMISABLE_METRICS.get(spec_c.metric, {}).get(
                "label", spec_c.metric),
            "op": spec_c.op,
            "value": float(spec_c.value),
            "binding_fraction": float(near.sum() / max(feasible.sum(), 1)),
            "violated_fraction": float((~feasible).sum() / max(len(column), 1)),
        })
    return report


def _sensitivity(space: DesignSpace, x: np.ndarray, objective: Objective,
                 spec: RunSpec) -> List[Dict]:
    """How much the leading objective moves if each dimension is nudged.

    A step either way per free dimension -- one grid step where a machining grid
    is set, otherwise 2% of the range -- so the tornado reflects choices the
    builder could actually make rather than infinitesimals.
    """
    x = space.canonical_one(np.asarray(x, dtype=float))
    metric = objective.objective_labels[0]
    base_row = evaluate_batch(space, x[None, :], timestep=spec.verify_timestep,
                              workers=1)
    if not len(base_row) or not bool(base_row["ok"].iloc[0]):
        return []
    reference = float(base_row[metric].iloc[0])

    probes, meta = [], []
    for slot, index in enumerate(space.free_index):
        var = space.specs[index]
        delta = var.step if var.step and var.step > 0 else 0.02 * (var.high - var.low)
        for sign in (-1.0, 1.0):
            probe = x.copy()
            probe[slot] = np.clip(probe[slot] + sign * delta, var.low, var.high)
            probes.append(probe)
            meta.append((var, sign))
    frame = evaluate_batch(space, np.array(probes), timestep=spec.verify_timestep)

    gathered: Dict[str, Dict] = {}
    for (var, sign), (_, row) in zip(meta, frame.iterrows()):
        entry = gathered.setdefault(var.name, {
            "variable": var.name, "label": var.label or var.name,
            "down": 0.0, "up": 0.0})
        change = (float(row[metric]) - reference) if bool(row["ok"]) else 0.0
        entry["down" if sign < 0 else "up"] = change
    for entry in gathered.values():
        entry["span"] = abs(entry["up"] - entry["down"])
    return sorted(gathered.values(), key=lambda e: -e["span"])


def _convergence(history: pd.DataFrame, objective: Objective,
                 space: DesignSpace, cap: int = 4000) -> List[Dict]:
    if not len(history):
        return []
    violation = scale_constraints(history, objective, space).max(axis=1)
    score = objective.score_frame(history)
    feasible = history["ok"].to_numpy(dtype=bool) & (violation <= 0)
    best = np.maximum.accumulate(np.where(feasible, score, -np.inf))
    stride = max(1, len(best) // cap)
    return [{"n": int(i + 1), "best": float(v)}
            for i, v in enumerate(best) if np.isfinite(v) and i % stride == 0]


def _population(history: pd.DataFrame, objective: Objective, space: DesignSpace,
                cap: int = 4000) -> List[Dict]:
    """A sample of everything simulated, for the scatter and parallel plots."""
    if not len(history):
        return []
    frame = history.copy()
    violation = scale_constraints(frame, objective.strict(), space).max(axis=1)
    frame["feasible"] = frame["ok"].to_numpy(dtype=bool) & (violation <= 0)
    if len(frame) > cap:
        frame = frame.sample(cap, random_state=0)
    columns = ([m for m in OPTIMISABLE_METRICS if m in frame.columns]
               + [n for n in space.names if n in frame.columns] + ["feasible"])
    return frame[columns].to_dict("records")


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------


def run(spec: RunSpec, base_motor: Dict, on_progress: ProgressFn = _noop,
        workers: Optional[int] = None) -> RunResult:
    problems = spec.validate()
    if problems:
        raise ValueError("; ".join(problems))

    started = time.time()
    result = RunResult(spec=spec.to_dict())
    on_progress("baseline", 0.02, "Simulating your current motor")

    space = build_space(spec, base_motor)
    baseline_metrics = simulate_motor(base_motor, timestep=spec.verify_timestep)
    bias = timestep_bias(base_motor, spec.search_timestep, spec.verify_timestep)
    objective = build_objective(spec, baseline_metrics, bias)
    # Verification uses the limits exactly as the user typed them.
    verify_objective = build_objective(spec, baseline_metrics, bias=None).strict()
    budget = spec.budget

    baseline_x = space.from_motor(base_motor)
    result.baseline = describe_design(space, baseline_x, spec, "Your motor",
                                      with_curves=True)
    # Report the motor as it is, not as the space rounds it -- a baseline that
    # is off the machining grid should still be shown truthfully.
    result.baseline.update({
        "cores": [float(g["properties"]["coreDiameter"]) for g in base_motor["grains"]],
        "throat": float(base_motor["nozzle"]["throat"]),
        "exit": float(base_motor["nozzle"]["exit"]),
        **{k: float(getattr(baseline_metrics, k)) for k in
           ("initial_thrust", "total_impulse", "isp", "burn_time", "peak_kn",
            "initial_kn", "port_throat", "prop_mass", "peak_mass_flux")},
        "max_pressure_psi": float(baseline_metrics.max_pressure / PA_PER_PSI),
        "mass_flux_lb": float(baseline_metrics.peak_mass_flux / KG_M2S_PER_LB_IN2S),
    })

    surrogate = None
    history = pd.DataFrame()
    front = pd.DataFrame()
    n_obj = len(spec.enabled_objectives)

    if spec.mode == "pareto":
        on_progress("sampling", 0.06, "Sampling the design space")
        dataset = generate_mixed_dataset(space, budget["samples"],
                                         timestep=spec.search_timestep,
                                         seed=spec.seed, workers=workers)
        history = dataset
        on_progress("training", 0.42, "Training surrogate models")
        surrogate = Surrogate(space, kind="gbt", seed=spec.seed)
        scores = surrogate.fit(dataset)
        result.surrogate = {
            "kind": surrogate.kind,
            "scores": [s.as_row() for s in scores],
            "importances": surrogate.importances(
                dataset, objective.objective_labels[0]).head(12).to_dict("records"),
            "parity": _parity_sample(surrogate, dataset, space),
        }
        on_progress("search", 0.58, "Mapping the trade-off")
        from .optimize import surrogate_pareto
        seeds = _seed_designs(dataset, space, objective, baseline_x)
        run_out = surrogate_pareto(
            space, surrogate, objective, pop_size=budget["pop"],
            n_gen=budget["gen"], timestep=spec.verify_timestep,
            workers=workers, seed=spec.seed, seed_designs=seeds,
            reference=dataset)
        front = run_out.get("front", pd.DataFrame())
    else:
        searched = _multi_seed_search(space, objective, spec, budget,
                                      baseline_x, n_obj, workers, on_progress)
        front = searched["front"]
        history = searched["history"]
        result.stats["per_seed"] = searched["per_seed"]

    on_progress("verify", 0.90, "Re-simulating the winners at full fidelity")
    result.designs = _rank_designs(space, front, spec, verify_objective, history)
    if not result.designs:
        result.messages.append(
            "No design met every constraint. Try relaxing the tightest limit, "
            "widening a bound, or freeing another dimension.")
    else:
        result.sensitivity = _sensitivity(
            space, np.array(result.designs[0]["x"]), verify_objective, spec)

    result.convergence = _convergence(history, objective, space)
    result.population = _population(history, verify_objective, space)
    result.constraint_activity = _constraint_report(history, verify_objective, space)
    result.stats = {
        **result.stats,
        "simulations": int(len(history)),
        "seeds": int(budget.get("seeds", 1)),
        "budget": int(budget.get("total", 0)),
        "seconds": round(time.time() - started, 1),
        "mode": spec.mode,
        "effort": spec.effort,
        "n_designs": len(result.designs),
        "objective_labels": objective.objective_labels,
        "searched": space.names,
        "frozen": [s.name for s in space.specs if not s.free],
    }
    on_progress("done", 1.0, "Finished")
    return result


def _alternatives(space: DesignSpace, history: pd.DataFrame,
                  objective: Objective, best_x: np.ndarray,
                  count: int = 14) -> pd.DataFrame:
    """The winner plus the best genuinely different designs behind it.

    Deduplicating on the rounded design vector matters: an evolutionary run ends
    with most of its population clustered on the same motor, and a list of
    fourteen copies of one answer helps nobody.
    """
    columns = list(space.names)
    rows = [pd.DataFrame([dict(zip(columns, best_x))])]
    if len(history):
        violation = scale_constraints(history, objective, space).max(axis=1)
        good = history[history["ok"].to_numpy(dtype=bool) & (violation <= 0)]
        if len(good):
            ranked = good.assign(_score=objective.score_frame(good)).sort_values(
                "_score", ascending=False)
            seen = {tuple(np.round(best_x, 5))}
            picked = []
            for _, row in ranked.iterrows():
                key = tuple(np.round(row[columns].to_numpy(dtype=float), 5))
                if key in seen:
                    continue
                seen.add(key)
                picked.append(row[columns])
                if len(picked) >= count - 1:
                    break
            if picked:
                rows.append(pd.DataFrame(picked))
    return pd.concat(rows, ignore_index=True)


def _multi_seed_search(space: DesignSpace, objective: Objective, spec: RunSpec,
                       budget: Dict, baseline_x: np.ndarray, n_obj: int,
                       workers, on_progress) -> Dict:
    """Several independent searches, merged into one front.

    A genetic search converges on whichever basin its starting population
    happened to land in, so the same configuration run twice returns different
    answers -- measured at 6.4% apart on best initial thrust for this motor.
    Splitting the budget across independent seeds and merging beats spending it
    all on one search: three 4,800-simulation runs found a better front than a
    single 14,400-simulation run on the same problem.
    """
    n_seeds = max(1, int(budget.get("seeds", 1)))
    fronts, histories, per_seed = [], [], []

    for index in range(n_seeds):
        seed = int(spec.seed) + 1009 * index      # spread, not consecutive
        label = "search {} of {}".format(index + 1, n_seeds)

        def tick(algorithm, index=index, label=label):
            done = getattr(algorithm, "n_gen", 0) or 0
            within = min(done / max(budget["gen"], 1), 1.0)
            on_progress("search", 0.08 + 0.80 * (index + within) / n_seeds,
                        "{}: generation {} of {}".format(
                            label, min(int(done), budget["gen"]), budget["gen"]))

        if n_obj > 1:
            out = direct_pareto(
                space, objective, pop_size=budget["pop"], n_gen=budget["gen"],
                timestep=spec.search_timestep, workers=workers, seed=seed,
                verify_timestep=spec.verify_timestep,
                seed_designs=baseline_x[None, :], callback=tick)
            found = out.get("front", pd.DataFrame())
        else:
            out = direct_search(
                space, objective, pop_size=budget["pop"], n_gen=budget["gen"],
                timestep=spec.search_timestep, workers=workers, seed=seed,
                callback=tick, seed_designs=baseline_x[None, :])
            found = _alternatives(space, out.get("history", pd.DataFrame()),
                                  objective, out["x"])
        if len(found):
            found = found.copy()
            found["seed"] = seed
            fronts.append(found)
        history = out.get("history", pd.DataFrame())
        histories.append(history)
        per_seed.append({"seed": seed, "designs": int(len(found)),
                         "simulations": int(len(history))})

    combined = pd.concat(fronts, ignore_index=True) if fronts else pd.DataFrame()
    if len(combined) and n_obj > 1:
        # Merge, then take the non-dominated set of everything any seed found.
        combined = combined.iloc[pareto_indices(-objective.matrix(combined))]
        combined = combined.reset_index(drop=True)
    return {"front": combined,
            "history": pd.concat(histories, ignore_index=True) if histories
                       else pd.DataFrame(),
            "per_seed": per_seed}


def _seed_designs(dataset: pd.DataFrame, space: DesignSpace,
                  objective: Objective, baseline_x: np.ndarray) -> np.ndarray:
    """Baseline plus the best already-sampled designs, as a starting population."""
    usable = dataset[dataset["ok"]]
    if not len(usable):
        return baseline_x[None, :]
    ranked = usable.assign(_score=objective.score_frame(usable)).nlargest(
        60, "_score")
    return np.vstack([baseline_x[None, :],
                      space.canonicalize(ranked[space.names].to_numpy(dtype=float))])


def _rank_designs(space: DesignSpace, front: pd.DataFrame, spec: RunSpec,
                  objective: Objective, history: pd.DataFrame) -> List[Dict]:
    """Verifies each candidate and keeps only those that clear every limit."""
    if not len(front):
        return []
    X = space.canonicalize(front[space.names].to_numpy(dtype=float))
    if len(X) > 60:  # keep the front readable, spread across the trade-off
        keep = np.linspace(0, len(X) - 1, 60).round().astype(int)
        X = X[np.unique(keep)]
    verified = evaluate_batch(space, X, timestep=spec.verify_timestep)
    violation = scale_constraints(verified, objective.strict(), space).max(axis=1)
    good = verified[verified["ok"].to_numpy(dtype=bool) & (violation <= 0)]
    if not len(good):
        return []
    axes = -objective.matrix(good)
    kept = good.iloc[pareto_indices(axes)] if axes.shape[1] > 1 else good
    kept = kept.assign(_score=objective.score_frame(kept)).sort_values(
        "_score", ascending=False)
    designs = []
    for rank, (_, row) in enumerate(kept.iterrows()):
        x = space.canonical_one(row[space.names].to_numpy(dtype=float))
        designs.append(describe_design(space, x, spec,
                                       "Option {}".format(rank + 1),
                                       with_curves=rank < 6))
    return designs


def _parity_sample(surrogate: Surrogate, dataset: pd.DataFrame,
                   space: DesignSpace, cap: int = 1500) -> Dict:
    """Predicted-vs-simulated points for the diagnostics profile."""
    usable = dataset[dataset["ok"]]
    sample = usable.sample(min(cap, len(usable)), random_state=0)
    predicted = surrogate.predict(sample[space.names].to_numpy(dtype=float))
    out = {}
    for target in predicted.columns:
        if target in sample.columns:
            out[target] = {
                "actual": sample[target].to_numpy(dtype=float).tolist(),
                "predicted": predicted[target].to_numpy(dtype=float).tolist(),
            }
    return out
