"""The design space: what the optimiser is allowed to change, and how a design
vector turns into an openMotor motor.

The case is treated as fixed hardware -- grain outer diameter, grain count,
grain lengths and the propellant all come from the baseline .ric and are never
touched. What is free is the core diameter of each grain plus the nozzle throat
and exit diameters, and any of those may be frozen or restricted to a machining
grid by the caller.

Three simplifications collapse the search space without giving anything up:

* **Cores are stored sorted, smallest forward.** Grain order changes nothing in
  openMotor's model except mass flux and port/throat ratio, and both of those
  are best when the largest port sits at the aft end. So the sorted arrangement
  dominates every permutation of the same six diameters, and searching only
  sorted vectors removes a 6! = 720-fold degeneracy. ``verify_ordering.py``
  checks this claim against the simulator.
* **Frozen dimensions leave the search vector entirely.** A dimension pinned to
  one value is not a variable with a zero-width range -- the evolutionary
  operators divide by range -- so it is removed and reinserted on the way out.
* **Exit diameter is parameterised as a fraction** of the span between a
  slightly-supersonic throat and the largest nozzle that fits the airframe.
  That keeps the space a plain box while guaranteeing exit > throat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from motorlib.propellant import Propellant

from .ric import clone
from .spec import OrderingSpec, VariableSpec
from .units import round_up_to_step, snap


@dataclass(frozen=True)
class Variable:
    name: str
    low: float
    high: float
    unit: str
    description: str


@dataclass(frozen=True)
class SpaceConfig:
    """Bounds for the free variables, plus the operating envelope.

    The ``max_*``/``min_port_throat`` fields override whatever the baseline .ric
    carries, so a tighter envelope than openMotor's own warning thresholds can be
    imposed without editing the motor file.

    This is the convenience path used by the study scripts. The app builds an
    explicit list of :class:`~rocketopt.spec.VariableSpec` instead, which can
    additionally freeze individual dimensions.
    """

    core_min: float = 0.020
    core_max: float = 0.068
    throat_min: float = 0.018
    throat_max: float = 0.050
    exit_max: float = 0.095
    exit_min_ratio: float = 1.15  # exit diameter >= throat * this
    #: Absolute floor on exit diameter, on top of the throat ratio. Zero leaves
    #: the ratio in sole charge.
    exit_min: float = 0.0
    #: Hold the exit at one diameter while the throat still moves. A frozen
    #: *fraction* would not do this -- the fraction spans throat*ratio to the
    #: airframe limit, so it maps to a different diameter for every throat.
    exit_fixed: Optional[float] = None
    #: Below roughly this chamber pressure a composite propellant chuffs or
    #: extinguishes rather than burning steadily. openMotor will happily
    #: simulate a 96 psi motor; a real one would not fly.
    min_chamber_pressure: float = 200 * 6894.757293168361

    # --- machining grids; 0 means any value is acceptable -------------------
    core_step: float = 0.0
    throat_step: float = 0.0
    exit_step: float = 0.0
    throat_length_step: float = 0.0

    #: Nozzle throat length. Off by default so the study scripts keep the
    #: 8-variable space their saved surrogate was trained against; the app
    #: turns it on. Shorter is always better in openMotor's model -- the loss
    #: term is monotonic in throat aspect ratio -- so the useful answer is the
    #: lower bound the builder is willing to machine.
    include_throat_length: bool = False
    throat_length_min: float = 0.0
    throat_length_max: float = 0.0

    # --- operating envelope; None means "take it from the .ric" -------------
    max_pressure: Optional[float] = None
    max_mass_flux: Optional[float] = None
    min_port_throat: Optional[float] = None
    #: Ceiling on peak Kn over the whole burn. openMotor has no such setting,
    #: but Kn is what a builder actually designs to.
    max_kn: Optional[float] = None

    #: Minimum increase in core diameter from one grain to the next going aft.
    #: Zero permits equal cores; a positive value forces a strictly widening
    #: port so mass flow never meets a narrower passage downstream.
    min_core_step: float = 0.0
    #: Group sizes for grains that must share a core diameter, e.g. (2, 2, 2)
    #: for three mandrel sizes used in pairs. None leaves every grain free.
    core_groups: Optional[Tuple[int, ...]] = None


class DesignSpace:
    """Maps a design vector to a motor dict and to physics-derived features."""

    def __init__(self, base_motor: Dict, config: SpaceConfig | None = None,
                 variables: Optional[Sequence[VariableSpec]] = None,
                 ordering: Optional[OrderingSpec] = None) -> None:
        self.base = clone(base_motor)
        self.config = config or SpaceConfig()
        self.n_grains = len(self.base["grains"])
        if self.n_grains == 0:
            raise ValueError("baseline motor has no grains")

        props = [grain["properties"] for grain in self.base["grains"]]
        self.grain_diameter = props[0]["diameter"]
        self.grain_lengths = np.array([p["length"] for p in props], dtype=float)
        if any(abs(p["diameter"] - self.grain_diameter) > 1e-12 for p in props):
            raise ValueError("all grains must share an outer diameter")

        self.propellant_density = self.base["propellant"]["density"]
        self._propellant = Propellant(self.base["propellant"])
        cfg = self.config
        self.max_pressure = (cfg.max_pressure if cfg.max_pressure is not None
                             else self.base["config"]["maxPressure"])
        self.max_mass_flux = (cfg.max_mass_flux if cfg.max_mass_flux is not None
                              else self.base["config"]["maxMassFlux"])
        self.min_port_throat = (cfg.min_port_throat if cfg.min_port_throat is not None
                                else self.base["config"]["minPortThroat"])
        self.max_kn = cfg.max_kn

        self.ordering = ordering or self._ordering_from_config()
        self.specs: List[VariableSpec] = (
            list(variables) if variables is not None else self._specs_from_config())
        # Slots are looked up by name, not by counting -- an optional variable
        # would otherwise silently shift every index after it.
        self.slot = {spec.name: i for i, spec in enumerate(self.specs)}
        required = ["core_{}".format(i + 1) for i in range(self.n_grains)]
        required += ["throat", "exit_frac"]
        missing = [name for name in required if name not in self.slot]
        if missing:
            raise ValueError("missing variable specs: {}".format(", ".join(missing)))
        self.has_throat_length = "throat_length" in self.slot

        self.all_variables: List[Variable] = [
            Variable(spec.name, spec.low, spec.high, spec.unit,
                     spec.label or spec.name)
            for spec in self.specs
        ]
        self.free_mask = np.array([spec.free for spec in self.specs], dtype=bool)
        self.free_index = np.flatnonzero(self.free_mask)
        self.fixed_full = np.array(
            [spec.fixed_value if spec.fixed_value is not None
             else 0.5 * (spec.low + spec.high) for spec in self.specs], dtype=float)

        # A ladder rung that is not a whole number of grid steps would walk the
        # cores off the machining grid, so round it up to one.
        self.core_grid = float(self.specs[0].step or 0.0)
        self.min_core_step = round_up_to_step(
            float(self.ordering.min_step or 0.0), self.core_grid)

    # ------------------------------------------------------- legacy adapters

    def _ordering_from_config(self) -> OrderingSpec:
        cfg = self.config
        if cfg.core_groups:
            return OrderingSpec(mode="paired", min_step=cfg.min_core_step,
                                groups=tuple(cfg.core_groups))
        mode = "strict" if cfg.min_core_step > 0 else "nondecreasing"
        return OrderingSpec(mode=mode, min_step=cfg.min_core_step)

    def _specs_from_config(self) -> List[VariableSpec]:
        cfg = self.config
        core_max = min(cfg.core_max, self.grain_diameter - 0.004)
        specs = [
            VariableSpec(name="core_{}".format(i + 1), free=True, low=cfg.core_min,
                         high=core_max, step=cfg.core_step, unit="m",
                         label="Grain {} core".format(i + 1))
            for i in range(self.n_grains)
        ]
        specs.append(VariableSpec(name="throat", free=True, low=cfg.throat_min,
                                  high=cfg.throat_max, step=cfg.throat_step,
                                  unit="m", label="Nozzle throat"))
        # For the exit slot the bounds are a fraction but the step is a
        # diameter, since a machinist grids the hole, not the fraction.
        specs.append(VariableSpec(name="exit_frac", free=True, low=0.0, high=1.0,
                                  step=cfg.exit_step, unit="m",
                                  label="Nozzle exit"))
        if cfg.include_throat_length:
            low = cfg.throat_length_min or 0.0
            high = cfg.throat_length_max or max(low + 1e-6, 0.02)
            specs.append(VariableSpec(
                name="throat_length", free=True, low=low, high=high,
                step=cfg.throat_length_step, unit="m", label="Throat length"))
        return specs

    def worker_spec(self):
        """(class, kwargs) needed to rebuild this space in a worker process.

        Subclasses and spec-driven spaces take extra constructor arguments, so
        the pool cannot assume the base class with default variables -- doing so
        gives workers a space with the wrong number of variables.
        """
        return type(self), {"base_motor": self.base, "config": self.config,
                            "variables": self.specs, "ordering": self.ordering}

    # ---------------------------------------------------------------- bounds

    @property
    def names(self) -> List[str]:
        """Names of the dimensions actually being searched."""
        return [self.specs[i].name for i in self.free_index]

    @property
    def all_names(self) -> List[str]:
        return [spec.name for spec in self.specs]

    @property
    def lower(self) -> np.ndarray:
        return np.array([self.specs[i].low for i in self.free_index], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.array([self.specs[i].high for i in self.free_index], dtype=float)

    @property
    def n_dim(self) -> int:
        return int(self.free_mask.sum())

    @property
    def variables(self) -> List[Variable]:
        """Metadata for the searched dimensions (kept for reporting)."""
        return [self.all_variables[i] for i in self.free_index]

    # ------------------------------------------------- free <-> full vectors

    def expand(self, X: np.ndarray) -> np.ndarray:
        """Inserts the frozen dimensions back into a searched vector."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        full = np.tile(self.fixed_full, (len(X), 1))
        full[:, self.free_index] = X
        return full

    def contract(self, X_full: np.ndarray) -> np.ndarray:
        X_full = np.atleast_2d(np.asarray(X_full, dtype=float))
        return X_full[:, self.free_index]

    # ------------------------------------------------------------ conversion

    def canonicalize(self, x: np.ndarray) -> np.ndarray:
        """Puts a searched vector into its single canonical form.

        Every design has exactly one canonical form, so the dataset and the
        surrogate never see the same motor under 720 different labels.
        """
        return self.contract(self.canonical_full(self.expand(x)))

    def canonical_full(self, X_full: np.ndarray) -> np.ndarray:
        """Canonicalises a full-length vector: bounds, ordering, machining grid."""
        X = np.atleast_2d(np.asarray(X_full, dtype=float)).copy()
        lows = np.array([s.low for s in self.specs], dtype=float)
        highs = np.array([s.high for s in self.specs], dtype=float)
        np.clip(X, lows, highs, out=X)

        n = self.n_grains
        X[:, :n] = self._arrange_cores(X[:, :n])

        for name in ("throat", "throat_length"):
            if name not in self.slot:
                continue
            i = self.slot[name]
            spec = self.specs[i]
            X[:, i] = snap(X[:, i], spec.step, spec.low, spec.high)
        throat_i, exit_i = self.slot["throat"], self.slot["exit_frac"]
        X[:, exit_i] = self._snap_exit(X[:, throat_i], X[:, exit_i])
        return X

    def canonical_one(self, x: np.ndarray) -> np.ndarray:
        return self.canonicalize(x)[0]

    def _snap_exit(self, throat: np.ndarray, frac: np.ndarray) -> np.ndarray:
        """Snaps the exit *diameter* to its grid and re-solves the fraction.

        The stored variable is a fraction of the span between the smallest
        useful exit and the airframe limit, but nobody machines a fraction. So
        the diameter is snapped and the fraction that reproduces it is stored
        back, which also keeps the canonical form unique -- otherwise many
        fractions would map to one motor and the surrogate would see the same
        design under different labels.
        """
        step = self.specs[self.slot["exit_frac"]].step
        if not step or step <= 0 or self.config.exit_fixed is not None:
            return frac
        low, high = self.exit_span(throat)
        diameter = low + frac * (high - low)
        diameter = snap(diameter, step, low, high)
        return np.clip((diameter - low) / (high - low), 0.0, 1.0)

    def _arrange_cores(self, cores: np.ndarray) -> np.ndarray:
        """Applies ordering, grouping, the machining grid and the ladder.

        Order of operations matters. Grouping averages, which lands off-grid, so
        snapping follows it rather than preceding it. The ladder is applied last
        because its rung is already a whole number of grid steps, so walking it
        keeps every core on the grid.
        """
        cores = cores.copy()
        lo = self.specs[0].low
        hi = self.specs[0].high
        grid = self.core_grid
        mode = self.ordering.mode

        core_free = self.free_mask[: self.n_grains]
        if mode != "none":
            if core_free.all():
                # All cores free: sorting is the provably optimal arrangement.
                cores = np.sort(cores, axis=1)
            else:
                # With some cores pinned, sorting would move a value out of the
                # slot the user pinned it to. Push only the free ones up to
                # their left neighbour instead.
                for i in range(1, cores.shape[1]):
                    if core_free[i]:
                        cores[:, i] = np.maximum(cores[:, i], cores[:, i - 1])

        groups = self.ordering.groups if mode == "paired" else None
        if groups:
            if sum(groups) != self.n_grains:
                raise ValueError("core groups must sum to the grain count")
            start = 0
            for size in groups:
                block = cores[:, start : start + size]
                cores[:, start : start + size] = block.mean(axis=1, keepdims=True)
                start += size

        if grid > 0:
            cores = snap(cores, grid, lo, hi)

        step = self.min_core_step if mode == "strict" else 0.0
        if step > 0:
            if (self.n_grains - 1) * step > hi - lo:
                raise ValueError("minimum core step is too large for the core bounds")
            cores = self._ladder(cores, step, lo, hi, grid)
        return cores

    @staticmethod
    def _ladder(cores: np.ndarray, step: float, lo: float, hi: float,
                grid: float = 0.0) -> np.ndarray:
        """Forces each core to exceed the one ahead of it by at least ``step``.

        Pushing the ladder up can run past the upper bound, so the whole vector
        slides back down by the overshoot and the pass is repeated -- the bounds
        are wide enough for the ladder to fit, so the second pass always lands
        inside them. When a machining grid is in force the slide is rounded up
        to a whole number of steps, so the ladder never falls off the grid.
        """
        cores = cores.copy()
        for _ in range(2):
            for i in range(1, cores.shape[1]):
                cores[:, i] = np.maximum(cores[:, i], cores[:, i - 1] + step)
            overshoot = np.maximum(cores[:, -1] - hi, 0.0)
            if grid > 0:
                overshoot = np.ceil(overshoot / grid - 1e-9) * grid
            cores -= overshoot[:, None]
            cores = snap(cores, grid, lo, hi) if grid > 0 else np.clip(cores, lo, hi)
        return cores

    def exit_span(self, throat):
        """Smallest and largest exit diameter allowed for a given throat."""
        low = np.maximum(throat * self.config.exit_min_ratio, self.config.exit_min)
        high = np.maximum(self.config.exit_max, low + 1e-6)
        return low, high

    def exit_diameter(self, throat: float, exit_frac: float) -> float:
        low, high = self.exit_span(throat)
        if self.config.exit_fixed is not None:
            return float(np.clip(self.config.exit_fixed, low, high))
        return low + exit_frac * (high - low)

    def to_motor(self, x: Sequence[float]) -> Dict:
        """Builds a full openMotor motor dict from one canonical design vector."""
        full = self.canonical_full(self.expand(np.asarray(x, dtype=float)))[0]
        motor = clone(self.base)
        for grain, core in zip(motor["grains"], full[: self.n_grains]):
            grain["properties"]["coreDiameter"] = float(core)
        throat = float(full[self.slot["throat"]])
        motor["nozzle"]["throat"] = throat
        motor["nozzle"]["exit"] = float(
            self.exit_diameter(throat, float(full[self.slot["exit_frac"]]))
        )
        if self.has_throat_length:
            motor["nozzle"]["throatLength"] = float(full[self.slot["throat_length"]])
        return motor

    def from_motor(self, motor: Dict) -> np.ndarray:
        """Encodes an existing motor as a searched vector (used for the baseline)."""
        cores = [g["properties"]["coreDiameter"] for g in motor["grains"]]
        throat = motor["nozzle"]["throat"]
        exit_d = motor["nozzle"]["exit"]
        low, high = self.exit_span(throat)
        frac = float(np.clip((exit_d - low) / (high - low), 0.0, 1.0))
        values = list(cores) + [throat, frac]
        if self.has_throat_length:
            values.append(motor["nozzle"].get("throatLength", 0.0))
        full = np.array(values, dtype=float)
        return self.canonicalize(self.contract(full))[0]

    # ------------------------------------------------------------- features

    @property
    def feature_names(self) -> List[str]:
        return (
            ["core_{}".format(i + 1) for i in range(self.n_grains)]
            + [
                "throat",
                "exit",
                "expansion_ratio",
                "throat_area",
                "burn_area_0",
                "kn_0",
                "port_area_aft",
                "port_throat_0",
                "web_min",
                "web_max",
                "web_spread",
                "prop_volume",
                "prop_mass",
                "pressure_0",
            ]
            + (["throat_length", "throat_aspect"] if self.has_throat_length else [])
        )

    def initial_pressure(self, kn_0) -> np.ndarray:
        """Steady-state chamber pressure at ignition, via openMotor's own
        Saint-Robert solve. Exact, so the surrogate never has to learn it."""
        return np.array(
            [self._propellant.getPressureFromKn(float(k)) for k in np.atleast_1d(kn_0)],
            dtype=float,
        )

    def features(self, x: np.ndarray) -> np.ndarray:
        """Physics-derived features for one or many design vectors.

        These are all closed-form BATES quantities, so they cost nothing next to
        a simulation. Giving the surrogate ``kn_0`` in particular means it does
        not have to rediscover that initial pressure is set by burn area over
        throat area.
        """
        X = self.canonical_full(self.expand(np.atleast_2d(np.asarray(x, dtype=float))))
        cores = X[:, : self.n_grains]
        throat = X[:, self.slot["throat"]]
        frac = X[:, self.slot["exit_frac"]]

        exit_d = np.array(
            [self.exit_diameter(t, f) for t, f in zip(throat, frac)], dtype=float
        )
        throat_area = math.pi * throat**2 / 4.0
        expansion = (exit_d / throat) ** 2

        diameter = self.grain_diameter
        lengths = self.grain_lengths[None, :]
        # BATES, ends uninhibited: core wall + both end faces.
        face_area = math.pi * (diameter**2 - cores**2) / 4.0
        core_area = math.pi * cores * lengths
        burn_area_0 = (core_area + 2.0 * face_area).sum(axis=1)
        kn_0 = burn_area_0 / throat_area

        port_area_aft = math.pi * cores[:, -1] ** 2 / 4.0
        port_throat_0 = port_area_aft / throat_area

        web = (diameter - cores) / 2.0
        prop_volume = (face_area * lengths).sum(axis=1)

        return np.column_stack(
            [
                cores,
                throat,
                exit_d,
                expansion,
                throat_area,
                burn_area_0,
                kn_0,
                port_area_aft,
                port_throat_0,
                web.min(axis=1),
                web.max(axis=1),
                web.max(axis=1) - web.min(axis=1),
                prop_volume,
                prop_volume * self.propellant_density,
                self.initial_pressure(kn_0),
            ]
            + ([X[:, self.slot["throat_length"]],
                X[:, self.slot["throat_length"]] / throat]
               if self.has_throat_length else [])
        )
