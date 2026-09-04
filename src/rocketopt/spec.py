"""The vocabulary the app and the optimiser share.

Everything a person configures in the GUI -- which dimensions may move, how
finely they may move, what counts as better, and what must never be exceeded --
lands in these dataclasses. They are plain data with ``to_dict``/``from_dict``
on both sides, so a saved run is a JSON file and nothing more.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

#: Metrics a user may optimise or constrain. Anything on Metrics could be
#: offered, but these are the ones that mean something to a motor builder.
OPTIMISABLE_METRICS: Dict[str, Dict] = {
    "initial_thrust": {"label": "Initial thrust", "unit": "N", "kind": None,
                       "help": "Mean thrust over the first 0.25 s. Drives how hard the "
                               "rocket leaves the rail."},
    "total_impulse": {"label": "Total impulse", "unit": "N·s", "kind": None,
                      "help": "Area under the thrust curve. Sets the motor's letter class."},
    "peak_thrust": {"label": "Peak thrust", "unit": "N", "kind": None,
                    "help": "Highest instantaneous thrust."},
    "avg_thrust": {"label": "Average thrust", "unit": "N", "kind": None,
                   "help": "Total impulse divided by burn time."},
    "isp": {"label": "Specific impulse", "unit": "s", "kind": None,
            "help": "Impulse per unit propellant weight. Higher means a more efficient nozzle."},
    "burn_time": {"label": "Burn time", "unit": "s", "kind": None,
                  "help": "How long the motor produces thrust."},
    "thrust_variation": {"label": "Thrust variation", "unit": "peak/avg", "kind": None,
                         "help": "Peak thrust over average. 1.0 is a perfectly flat curve."},
    "max_pressure": {"label": "Peak chamber pressure", "unit": "psi", "kind": "pressure",
                     "help": "The number your case has to survive."},
    "avg_pressure": {"label": "Mean chamber pressure", "unit": "psi", "kind": "pressure",
                     "help": "Too low and a composite propellant chuffs or goes out."},
    "peak_kn": {"label": "Peak Kn", "unit": "", "kind": None,
                "help": "Burning surface area over throat area, at its highest."},
    "initial_kn": {"label": "Initial Kn", "unit": "", "kind": None,
                   "help": "Kn at ignition. Sets initial pressure, so it sets initial thrust."},
    "peak_mass_flux": {"label": "Peak mass flux", "unit": "lb/in\u00b2s", "kind": "mass_flux",
                       "help": "Gas mass flow per unit port area. High values risk erosive burning."},
    "port_throat": {"label": "Port/throat ratio", "unit": "", "kind": None,
                    "help": "Aft port area over throat area. Too low and the port chokes."},
    "prop_mass": {"label": "Propellant mass", "unit": "kg", "kind": None,
                  "help": "How much propellant the motor holds."},
    "volume_loading": {"label": "Volume loading", "unit": "%", "kind": None,
                       "help": "Fraction of the chamber filled with propellant."},
    "separation_pct": {"label": "Flow separation", "unit": "%", "kind": None,
                       "help": "Portion of the burn where the nozzle may be separated."},
}

ORDERING_MODES = {
    "none": "Cores may be in any order",
    "nondecreasing": "Cores never narrow going aft (ties allowed)",
    "strict": "Each core is wider than the one ahead of it",
    "paired": "A few mandrel sizes, each used on several grains",
}


def _clean(value):
    return {k: v for k, v in value.items() if v is not None}


@dataclass
class VariableSpec:
    """One dimension the optimiser may or may not move.

    ``step`` is the machining grid in metres -- 0.00127 for a 0.05 in reamer
    progression, 0 for "any value". ``fixed_value`` is used when ``free`` is
    False, so freezing a dimension does not lose the number it is frozen at.
    """

    name: str
    free: bool = True
    low: float = 0.0
    high: float = 1.0
    step: float = 0.0
    fixed_value: Optional[float] = None
    label: str = ""
    unit: str = "m"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "VariableSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ObjectiveSpec:
    """Something to push up, push down, or land on.

    ``weight`` only matters when several objectives are combined into a single
    score; in a Pareto run each enabled objective becomes its own axis.
    """

    metric: str
    direction: str = "max"          # "max" | "min" | "target"
    weight: float = 1.0
    target: Optional[float] = None  # SI, required when direction == "target"
    enabled: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ObjectiveSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ConstraintSpec:
    """A limit the design must respect, in SI."""

    metric: str
    op: str = "<="                  # "<=" | ">="
    value: float = 0.0
    enabled: bool = True
    #: Search-time tightening, 0-1. Peak mass flux creeps upward as the timestep
    #: shrinks, so searching it a few percent tight keeps boundary designs from
    #: failing the final check. Removed at verification.
    margin: float = 0.0
    label: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ConstraintSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class OrderingSpec:
    """How the grain cores must relate to one another down the stack."""

    mode: str = "nondecreasing"
    min_step: float = 0.0                       # metres, for mode == "strict"
    groups: Optional[Tuple[int, ...]] = None    # for mode == "paired"

    def to_dict(self) -> Dict:
        data = asdict(self)
        if data.get("groups"):
            data["groups"] = list(data["groups"])
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "OrderingSpec":
        groups = data.get("groups")
        return cls(
            mode=data.get("mode", "nondecreasing"),
            min_step=float(data.get("min_step", 0.0) or 0.0),
            groups=tuple(groups) if groups else None,
        )


#: Effort presets, expressed as a total simulation budget. A run splits that
#: budget across several independent searches rather than spending it all on
#: one: three 4,800-simulation searches merged beat a single 14,400-simulation
#: search on the same motor, because a genetic search converges on whatever
#: basin it happened to start in.
EFFORT_LEVELS = {
    "quick":     {"budget": 3600,  "samples": 2048,  "label": "Quick"},
    "standard":  {"budget": 14400, "samples": 8192,  "label": "Standard"},
    "thorough":  {"budget": 43200, "samples": 16384, "label": "Thorough"},
}

#: Generations each search should get. Population is then whatever the budget
#: affords -- too few generations and nothing converges, however wide the
#: population.
TARGET_GENERATIONS = 50
MIN_POPULATION = 40
MAX_POPULATION = 240


@dataclass
class RunSpec:
    """A complete, reproducible description of one optimisation run."""

    variables: List[VariableSpec] = field(default_factory=list)
    objectives: List[ObjectiveSpec] = field(default_factory=list)
    constraints: List[ConstraintSpec] = field(default_factory=list)
    ordering: OrderingSpec = field(default_factory=OrderingSpec)
    effort: str = "standard"
    #: Total simulations to spend, across every seed. None follows the preset.
    budget_simulations: Optional[int] = None
    #: Independent searches to run and merge. Each is seeded differently, and
    #: the reported front is the non-dominated set of everything they found.
    seeds: int = 3
    #: "fast" runs the genetic search straight against openMotor. "pareto" adds
    #: a surrogate and maps the whole trade-off between objectives.
    mode: str = "fast"
    seed: int = 17
    #: Impulse reads about 0.5%% low at 0.02 s, which makes a >= impulse
    #: constraint bite harder during the search than at verification. 0.01 s
    #: halves that bias for twice the simulation cost, which BATES can afford.
    search_timestep: float = 0.01
    verify_timestep: float = 0.002
    display_units: Dict[str, str] = field(
        default_factory=lambda: {"length": "in", "pressure": "psi",
                                 "mass_flux": "lb/(in^2*s)"})

    # ---------------------------------------------------------------- helpers

    @property
    def enabled_objectives(self) -> List[ObjectiveSpec]:
        return [o for o in self.objectives if o.enabled]

    @property
    def enabled_constraints(self) -> List[ConstraintSpec]:
        return [c for c in self.constraints if c.enabled]

    @property
    def budget(self) -> Dict:
        """Total budget split into a population and generation count per seed."""
        preset = EFFORT_LEVELS.get(self.effort, EFFORT_LEVELS["standard"])
        total = int(self.budget_simulations or preset["budget"])
        seeds = max(1, int(self.seeds))
        per_seed = max(total // seeds, MIN_POPULATION * 2)
        pop = int(min(max(per_seed // TARGET_GENERATIONS, MIN_POPULATION),
                      MAX_POPULATION))
        gen = max(int(per_seed // pop), 2)
        return {"pop": pop, "gen": gen, "seeds": seeds,
                "per_seed": pop * gen, "total": pop * gen * seeds,
                "samples": preset["samples"], "label": preset["label"]}

    def validate(self) -> List[str]:
        """Human-readable problems, empty when the spec is runnable."""
        problems: List[str] = []
        if not self.enabled_objectives:
            problems.append("Pick at least one thing to optimise.")
        if not any(v.free for v in self.variables):
            problems.append("At least one dimension has to be free to change.")
        for spec in self.objectives:
            if spec.enabled and spec.metric not in OPTIMISABLE_METRICS:
                problems.append("Unknown objective {!r}.".format(spec.metric))
            if spec.enabled and spec.direction == "target" and spec.target is None:
                problems.append("{} needs a target value.".format(spec.metric))
        for spec in self.constraints:
            if spec.enabled and spec.metric not in OPTIMISABLE_METRICS:
                problems.append("Unknown constraint {!r}.".format(spec.metric))
        for var in self.variables:
            if not var.free:
                continue
            if var.high <= var.low:
                problems.append("{}: upper bound must exceed the lower bound.".format(
                    var.label or var.name))
            elif var.step > 0 and var.step > (var.high - var.low):
                problems.append(
                    "{}: a step of {:.4g} is coarser than its whole range.".format(
                        var.label or var.name, var.step))
        if self.mode == "pareto" and len(self.enabled_objectives) < 2:
            problems.append("A trade-off map needs at least two objectives.")
        return problems

    def to_dict(self) -> Dict:
        return {
            "variables": [v.to_dict() for v in self.variables],
            "objectives": [o.to_dict() for o in self.objectives],
            "constraints": [c.to_dict() for c in self.constraints],
            "ordering": self.ordering.to_dict(),
            "effort": self.effort,
            "budget_simulations": self.budget_simulations,
            "seeds": self.seeds,
            "mode": self.mode,
            "seed": self.seed,
            "search_timestep": self.search_timestep,
            "verify_timestep": self.verify_timestep,
            "display_units": dict(self.display_units),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RunSpec":
        return cls(
            variables=[VariableSpec.from_dict(v) for v in data.get("variables", [])],
            objectives=[ObjectiveSpec.from_dict(o) for o in data.get("objectives", [])],
            constraints=[ConstraintSpec.from_dict(c) for c in data.get("constraints", [])],
            ordering=OrderingSpec.from_dict(data.get("ordering", {})),
            effort=data.get("effort", "standard"),
            budget_simulations=(int(data["budget_simulations"])
                                if data.get("budget_simulations") else None),
            seeds=max(1, int(data.get("seeds", 3) or 1)),
            mode=data.get("mode", "fast"),
            seed=int(data.get("seed", 17)),
            search_timestep=float(data.get("search_timestep", 0.01)),
            verify_timestep=float(data.get("verify_timestep", 0.002)),
            display_units=data.get("display_units") or {
                "length": "in", "pressure": "psi", "mass_flux": "lb/(in^2*s)"},
        )
