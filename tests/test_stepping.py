"""The machining grid, frozen dimensions, and the ordering rules.

These are the parts of the design space that the app exposes and the study
scripts never exercised, so they get their own checks. The guarantee being
tested is a practical one: whatever a user asks for, every design that comes
out is buildable -- on the grid, inside the bounds, and in the right order.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rocketopt.design import DesignSpace, SpaceConfig
from rocketopt.optimize import Objective, scale_constraints
from rocketopt.ric import default_motor_path, load_ric
from rocketopt.spec import ConstraintSpec, OrderingSpec, VariableSpec
from rocketopt.units import parse_number, round_up_to_step, snap

IN = 0.0254
MOTOR = default_motor_path(ROOT)


@pytest.fixture(scope="module")
def base():
    return load_ric(MOTOR)


def random_designs(space, n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(space.lower, space.upper, size=(n, space.n_dim))
    return space.canonicalize(X)


def on_grid(values, step):
    scaled = np.asarray(values) / step
    return np.abs(scaled - np.round(scaled)).max() < 1e-9


# --------------------------------------------------------------- unit parsing


@pytest.mark.parametrize("text,expected", [
    ("0.0625", 0.0625), ("1/16", 0.0625), ("1/32", 0.03125),
    ("1 1/16", 1.0625), ("2-1/2", 2.5), ('0.05"', 0.05), ("1.6", 1.6),
])
def test_parse_number_accepts_shop_fractions(text, expected):
    assert parse_number(text) == pytest.approx(expected)


def test_snap_anchors_at_zero_not_at_the_lower_bound():
    # A 1/16 in grid must offer whole sixteenths, regardless of where the
    # allowed range happens to start.
    step = IN / 16
    assert snap(0.4 * IN, step) == pytest.approx(step * round(0.4 * IN / step))
    assert on_grid([snap(0.4 * IN, step, low=0.3 * IN, high=0.9 * IN)], step)


def test_snap_never_leaves_the_bounds():
    step = 0.05 * IN
    assert snap(10.0, step, low=0.0, high=2.0 * IN) <= 2.0 * IN + 1e-12
    assert snap(-10.0, step, low=0.5 * IN, high=2.0 * IN) >= 0.5 * IN - 1e-12


def test_round_up_to_step_is_a_whole_number_of_steps():
    assert round_up_to_step(0.004, 0.00127) == pytest.approx(0.00381 + 0.00127)
    assert round_up_to_step(0.0, 0.00127) == 0.0
    assert round_up_to_step(0.5, 0.0) == 0.5


# ------------------------------------------------------------ machining grid


def test_every_diameter_lands_on_its_own_grid(base):
    cfg = SpaceConfig(core_step=0.05 * IN, throat_step=IN / 32, exit_step=IN / 16)
    space = DesignSpace(base, cfg)
    X = random_designs(space)
    assert on_grid(X[:, :6], 0.05 * IN)
    assert on_grid(X[:, 6], IN / 32)
    exits = np.array([space.to_motor(x)["nozzle"]["exit"] for x in X[:120]])
    assert on_grid(exits, IN / 16)


def test_grid_keeps_cores_sorted_and_inside_bounds(base):
    space = DesignSpace(base, SpaceConfig(core_step=0.05 * IN))
    X = random_designs(space)
    assert (np.diff(X[:, :6], axis=1) >= -1e-12).all()
    assert (X >= space.lower - 1e-12).all() and (X <= space.upper + 1e-12).all()


def test_exit_fraction_reproduces_the_snapped_diameter(base):
    """Canonical form must be unique: one motor, one design vector."""
    cfg = SpaceConfig(exit_step=IN / 16)
    space = DesignSpace(base, cfg)
    X = random_designs(space, n=200, seed=3)
    once = np.array([space.to_motor(x)["nozzle"]["exit"] for x in X])
    twice = np.array([space.to_motor(space.canonical_one(x))["nozzle"]["exit"] for x in X])
    assert np.abs(once - twice).max() < 1e-12


# --------------------------------------------------------------- ordering


def test_strict_ladder_and_grid_coexist(base):
    step = 0.05 * IN
    space = DesignSpace(base, SpaceConfig(core_step=step),
                        ordering=OrderingSpec(mode="strict", min_step=step))
    X = random_designs(space, seed=1)
    cores = X[:, :6]
    assert on_grid(cores, step)
    assert (np.diff(cores, axis=1) >= step - 1e-9).all()
    assert (cores >= space.lower[0] - 1e-12).all()
    assert (cores <= space.upper[0] + 1e-12).all()


def test_min_step_is_rounded_up_to_the_grid(base):
    """A rung that is not a whole number of steps would walk cores off-grid."""
    grid = 0.05 * IN
    space = DesignSpace(base, SpaceConfig(core_step=grid),
                        ordering=OrderingSpec(mode="strict", min_step=0.06 * IN))
    assert space.min_core_step == pytest.approx(2 * grid)
    assert on_grid(random_designs(space, seed=2)[:, :6], grid)


def test_paired_grouping_with_a_grid(base):
    grid = 0.05 * IN
    space = DesignSpace(base, SpaceConfig(core_step=grid),
                        ordering=OrderingSpec(mode="paired", groups=(2, 2, 2)))
    cores = random_designs(space, seed=4)[:, :6]
    for i in (0, 2, 4):
        assert np.abs(cores[:, i] - cores[:, i + 1]).max() < 1e-12
    assert on_grid(cores, grid)
    assert (np.diff(cores, axis=1) >= -1e-12).all()


def test_ladder_fits_even_from_a_degenerate_vector(base):
    space = DesignSpace(base, SpaceConfig(core_step=0.05 * IN),
                        ordering=OrderingSpec(mode="strict", min_step=0.3 * IN))
    x = np.r_[[space.upper[0]] * 6, space.upper[6], 1.0]
    cores = space.canonical_one(x)[:6]
    assert (cores >= space.lower[0] - 1e-12).all()
    assert (cores <= space.upper[0] + 1e-12).all()
    assert (np.diff(cores) >= 0.3 * IN - 1e-9).all()


# ------------------------------------------------------------- frozen slots


def test_freezing_shrinks_the_search_and_holds_the_value(base):
    specs = DesignSpace(base).specs
    frozen = list(specs)
    frozen[0] = VariableSpec(**{**specs[0].to_dict(), "free": False,
                                "fixed_value": 1.6 * IN})
    frozen[6] = VariableSpec(**{**specs[6].to_dict(), "free": False,
                                "fixed_value": 1.3 * IN})
    space = DesignSpace(base, SpaceConfig(), variables=frozen)
    assert space.n_dim == 6
    assert "core_1" not in space.names and "throat" not in space.names
    for x in random_designs(space, n=50, seed=5):
        motor = space.to_motor(x)
        assert motor["nozzle"]["throat"] == pytest.approx(1.3 * IN)
        assert motor["grains"][0]["properties"]["coreDiameter"] == pytest.approx(1.6 * IN)


def test_a_fixed_exit_diameter_survives_a_moving_throat(base):
    """A frozen *fraction* would not hold the diameter -- the span moves."""
    space = DesignSpace(base, SpaceConfig(exit_fixed=2.25 * IN))
    for throat_in in (1.0, 1.3, 1.6):
        x = space.from_motor(base).copy()
        x[space.names.index("throat")] = throat_in * IN
        assert space.to_motor(x)["nozzle"]["exit"] == pytest.approx(2.25 * IN)


def test_worker_spec_rebuilds_an_identical_space(base):
    """Pools rebuild the space in child processes from this alone."""
    space = DesignSpace(base, SpaceConfig(core_step=0.05 * IN),
                        ordering=OrderingSpec(mode="strict", min_step=0.05 * IN))
    cls, kwargs = space.worker_spec()
    rebuilt = cls(**kwargs)
    assert rebuilt.n_dim == space.n_dim
    assert rebuilt.names == space.names
    x = random_designs(space, n=1, seed=9)[0]
    assert np.allclose(rebuilt.canonical_one(x), space.canonical_one(x))


# ----------------------------------------------------- legacy compatibility


def space_containing(motor):
    """A space whose box actually holds the motor it was built from.

    The default :class:`SpaceConfig` carries fixed bounds that were tuned for
    one particular motor. A baseline outside them is clamped to the bound --
    correct behaviour, but it makes "does this round-trip" untestable. Derive
    the box from the motor instead, so these tests check the mechanics rather
    than one file's dimensions.
    """
    cores = [g["properties"]["coreDiameter"] for g in motor["grains"]]
    outer = motor["grains"][0]["properties"]["diameter"]
    throat = motor["nozzle"]["throat"]
    exit_d = motor["nozzle"]["exit"]
    return SpaceConfig(
        core_min=min(cores) * 0.5,
        core_max=min(max(cores) * 1.25, outer * 0.95),
        throat_min=throat * 0.5,
        throat_max=throat * 1.5,
        exit_max=max(exit_d * 1.25, throat * 2.0),
    )


def test_the_baseline_still_round_trips_exactly(base):
    """The study scripts depend on this being lossless."""
    space = DesignSpace(base, space_containing(base))
    motor = space.to_motor(space.from_motor(base))
    assert motor["nozzle"]["throat"] == pytest.approx(base["nozzle"]["throat"])
    assert motor["nozzle"]["exit"] == pytest.approx(base["nozzle"]["exit"], abs=1e-9)
    for built, original in zip(motor["grains"], base["grains"]):
        assert (built["properties"]["coreDiameter"]
                == pytest.approx(original["properties"]["coreDiameter"]))


def test_analytic_features_match_the_simulator(base):
    from rocketopt.simulate import simulate_motor
    space = DesignSpace(base, space_containing(base))
    features = dict(zip(space.feature_names, space.features(space.from_motor(base))[0]))
    metrics = simulate_motor(base, timestep=0.002)
    assert features["kn_0"] == pytest.approx(metrics.initial_kn, rel=1e-4)
    assert features["prop_mass"] == pytest.approx(metrics.prop_mass, rel=1e-3)
    assert features["port_throat_0"] == pytest.approx(metrics.port_throat, rel=1e-4)


# -------------------------------------------------------------- constraints


def test_spec_constraints_are_normalised_overshoots():
    import pandas as pd
    frame = pd.DataFrame({"ok": [True, True, True],
                          "max_pressure": [400.0, 500.0, 600.0]})
    objective = Objective(constraints=(
        ConstraintSpec(metric="max_pressure", op="<=", value=500.0),))
    g = scale_constraints(frame, objective, space=None)
    assert g[:, 0] == pytest.approx([-0.2, 0.0, 0.2])


def test_a_failed_simulation_violates_everything():
    import pandas as pd
    frame = pd.DataFrame({"ok": [False], "max_pressure": [0.0]})
    objective = Objective(constraints=(
        ConstraintSpec(metric="max_pressure", op="<=", value=500.0),))
    assert (scale_constraints(frame, objective, space=None) > 0).all()


def test_search_margin_tightens_but_verification_does_not():
    import pandas as pd
    frame = pd.DataFrame({"ok": [True], "peak_mass_flux": [738.0]})
    tight = Objective(constraints=(ConstraintSpec(
        metric="peak_mass_flux", op="<=", value=738.4, margin=0.03),))
    assert scale_constraints(frame, tight, space=None)[0, 0] > 0     # rejected
    assert scale_constraints(frame, tight.strict(), space=None)[0, 0] < 0  # allowed
