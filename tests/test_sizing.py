"""The configuration count, checked against brute-force enumeration.

The formulas replace an enumeration that would be impossible at real grid
sizes, so they are verified on small cases where enumeration is still cheap.
"""
import sys
from itertools import combinations_with_replacement, product
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rocketopt.ric import default_motor_path, load_ric
from rocketopt.runner import default_spec
from rocketopt.sizing import _multiset, _steps, size_space
from rocketopt.spec import OrderingSpec
from rocketopt.units import M_PER_IN as IN

MOTOR = default_motor_path(ROOT)


@pytest.fixture(scope="module")
def spec():
    """The configuration these features exist for: a real operating envelope on
    a 0.01 in grid. The stock defaults leave Kn unconstrained, which is exactly
    the case where there is nothing to derive."""
    from rocketopt.spec import ConstraintSpec
    from rocketopt.units import KG_M2S_PER_LB_IN2S as LB
    from rocketopt.units import PA_PER_PSI as PSI

    base = load_ric(MOTOR)
    s = default_spec(base)
    for var in s.variables:
        var.step = 0.01 * IN
        var.free = True
    s.constraints = [
        ConstraintSpec("peak_kn", "<=", 225.0, label="Peak Kn"),
        ConstraintSpec("max_pressure", "<=", 500 * PSI, label="Peak chamber pressure"),
        ConstraintSpec("peak_mass_flux", "<=", 1.05 * LB, label="Peak mass flux"),
        ConstraintSpec("port_throat", ">=", 1.4, label="Port/throat ratio"),
    ]
    return s


@pytest.mark.parametrize("values,slots", [(6, 3), (8, 4), (10, 3), (5, 5)])
def test_multiset_matches_enumeration(values, slots):
    brute = sum(1 for _ in combinations_with_replacement(range(values), slots))
    assert _multiset(values, slots) == brute


@pytest.mark.parametrize("values,slots,gap", [(10, 3, 2), (12, 4, 2), (9, 3, 3)])
def test_strict_ladder_matches_enumeration(values, slots, gap):
    brute = sum(1 for x in product(range(values), repeat=slots)
                if all(x[i + 1] - x[i] >= gap for i in range(slots - 1)))
    assert _multiset(values - (slots - 1) * gap, slots) == brute


def test_steps_counts_both_endpoints():
    assert _steps(0.0, 1.0, 0.25) == 5           # 0, .25, .5, .75, 1
    assert _steps(0.5 * IN, 4.5 * IN, 0.01 * IN) == 401
    assert _steps(0.0, 1.0, 0.0) == 0            # continuous


def test_sorting_collapses_the_space_by_the_permutation_count(spec):
    """Sorted cores should be smaller than free order by close to n!."""
    import copy
    free = copy.deepcopy(spec)
    free.ordering = OrderingSpec(mode="none")
    sortd = copy.deepcopy(spec)
    sortd.ordering = OrderingSpec(mode="nondecreasing")
    a = size_space(free, 6)["cores"]["count"]
    b = size_space(sortd, 6)["cores"]["count"]
    assert 600 < a / b < 720          # approaches 6! = 720 as the grid grows


def test_ordering_rules_only_shrink_the_space(spec):
    import copy
    counts = {}
    for mode, kw in (("none", {}), ("nondecreasing", {}),
                     ("strict", {"min_step": 0.05 * IN}),
                     ("paired", {"groups": (2, 2, 2)})):
        s = copy.deepcopy(spec)
        s.ordering = OrderingSpec(mode=mode, **kw)
        counts[mode] = size_space(s, 6)["cores"]["count"]
    assert counts["none"] > counts["nondecreasing"] > counts["strict"]
    assert counts["paired"] < counts["nondecreasing"]


def test_a_continuous_dimension_makes_the_count_unbounded(spec):
    import copy
    s = copy.deepcopy(spec)
    s.variables[0].step = 0.0
    result = size_space(s, 6)
    assert result["total"] is None and result["continuous"]
    assert "step" in result["total_text"]


def test_everything_held_is_exactly_one_motor(spec):
    import copy
    s = copy.deepcopy(spec)
    for var in s.variables:
        var.free = False
    assert size_space(s, 6)["total"] == 1


def test_impossible_ladder_reports_zero(spec):
    """Six cores needing 1 in of separation cannot fit a 4 in range."""
    import copy
    s = copy.deepcopy(spec)
    s.ordering = OrderingSpec(mode="strict", min_step=1.0 * IN)
    assert size_space(s, 6)["total"] == 0


def test_exit_is_counted_jointly_with_the_throat(spec):
    """A wider throat leaves fewer legal exits, so the pair is not a product."""
    import copy
    s = copy.deepcopy(spec)
    nozzle = size_space(s, 6)["nozzle"]
    throat = next(v for v in s.variables if v.name == "throat")
    exit_var = next(v for v in s.variables if v.name == "exit")
    naive = _steps(throat.low, throat.high, throat.step) * \
        _steps(exit_var.low, exit_var.high, exit_var.step)
    assert nozzle["count"] < naive


# --------------------------------------------------- using the limits to prune


def test_closed_form_peak_kn_tracks_the_simulator():
    """The screen replaces a simulation, so it has to agree with one."""
    import numpy as np

    from rocketopt.ric import clone
    from rocketopt.simulate import simulate_motor
    from rocketopt.sizing import peak_burn_area

    base = load_ric(MOTOR)
    diameter = base["grains"][0]["properties"]["diameter"]
    length = base["grains"][0]["properties"]["length"]
    rng = np.random.default_rng(3)
    ratios = []
    for _ in range(6):
        cores = np.sort(rng.uniform(0.8 * IN, 3.8 * IN, 6))
        throat = rng.uniform(1.4 * IN, 2.4 * IN)
        motor = clone(base)
        for grain, core in zip(motor["grains"], cores):
            grain["properties"]["coreDiameter"] = float(core)
        motor["nozzle"]["throat"] = float(throat)
        motor["nozzle"]["exit"] = float(throat * 2.2)
        sim = simulate_motor(motor, 0.002)
        if not sim.ok:
            continue
        closed = peak_burn_area(cores[None, :], diameter, length)[0]
        ratios.append(closed / (3.14159265358979 * throat ** 2 / 4) / sim.peak_kn)
    assert ratios, "no usable samples"
    assert max(ratios) < 1.02, "closed form runs hotter than the screen margin allows"
    assert min(ratios) > 0.90


def test_the_kn_screen_never_rejects_a_feasible_design():
    """With the margin applied, anything the screen throws away is truly illegal."""
    import numpy as np

    from rocketopt.ric import clone
    from rocketopt.simulate import simulate_motor
    from rocketopt.sizing import KN_SCREEN_MARGIN, peak_burn_area

    base = load_ric(MOTOR)
    diameter = base["grains"][0]["properties"]["diameter"]
    length = base["grains"][0]["properties"]["length"]
    limit = 225.0
    rng = np.random.default_rng(11)
    for _ in range(10):
        cores = np.sort(rng.uniform(0.8 * IN, 4.0 * IN, 6))
        throat = rng.uniform(1.3 * IN, 2.6 * IN)
        closed_kn = peak_burn_area(cores[None, :], diameter, length)[0] / \
            (3.14159265358979 * throat ** 2 / 4)
        if closed_kn <= limit * (1 + KN_SCREEN_MARGIN):
            continue                        # kept, nothing to prove
        motor = clone(base)
        for grain, core in zip(motor["grains"], cores):
            grain["properties"]["coreDiameter"] = float(core)
        motor["nozzle"]["throat"] = float(throat)
        motor["nozzle"]["exit"] = float(throat * 2.2)
        sim = simulate_motor(motor, 0.002)
        if sim.ok:
            assert sim.peak_kn > limit, "screen rejected a design the simulator allows"


def test_uniform_multiset_sampling_is_actually_uniform():
    import numpy as np
    from collections import Counter

    from rocketopt.sizing import _multiset, _uniform_multisets

    rng = np.random.default_rng(0)
    grid = np.arange(5.0)
    drawn = _uniform_multisets(grid, 3, 40000, rng)
    counts = Counter(map(tuple, drawn.astype(int)))
    assert len(counts) == _multiset(5, 3)
    values = np.array(sorted(counts.values()))
    assert values.max() / values.min() < 1.3      # sampling noise only
    assert (np.diff(drawn, axis=1) >= 0).all()


def test_throat_floor_is_a_real_lower_bound(spec):
    """No throat below the derived floor can satisfy Kn, whatever the cores."""
    import numpy as np

    from rocketopt.sizing import peak_burn_area, tighten_bounds

    base = load_ric(MOTOR)
    tight = tighten_bounds(spec, base)
    floor = tight.get("throat_low")
    assert floor, "expected a throat floor for this configuration"
    diameter = base["grains"][0]["properties"]["diameter"]
    length = base["grains"][0]["properties"]["length"]
    core = next(v for v in spec.variables if v.name.startswith("core"))
    smallest = np.full((1, 6), core.low)
    area = peak_burn_area(smallest, diameter, length)[0]
    just_under = floor * 0.98
    assert area / (3.14159265358979 * just_under ** 2 / 4) > 225


def test_tightening_never_removes_the_whole_space(spec):
    from rocketopt.sizing import reduction_chain
    chain = reduction_chain(spec, load_ric(MOTOR), samples=400)
    assert chain["after_bounds"] > 0
    assert chain["after_bounds"] <= chain["total"]
    assert chain["legal"] <= chain["after_bounds"]


def test_kn_and_pressure_are_reported_as_one_constraint(spec):
    from rocketopt.sizing import equivalent_limits
    found = equivalent_limits(spec, load_ric(MOTOR))
    assert found and found[0]["binding"] in ("Kn", "pressure")
    assert 150 < found[0]["kn_at_pressure"] < 400


# ------------------------------------------------------- tolerance analysis


def test_per_grain_tolerances_vary_independently():
    """Six cores from six passes must not all move together."""
    import numpy as np

    from rocketopt.tolerance import ToleranceSpec, perturb

    base = load_ric(MOTOR)
    rng = np.random.default_rng(0)
    tol = [ToleranceSpec("core_diameter", 0.01 * IN)]
    built = perturb(base, tol, rng)
    deltas = [b["properties"]["coreDiameter"] - a["properties"]["coreDiameter"]
              for a, b in zip(base["grains"], built["grains"])]
    assert len(set(round(d, 12) for d in deltas)) > 1, "cores moved identically"


def test_batch_tolerances_move_every_grain_together():
    """Burn rate is a property of the mix, not of each grain."""
    import numpy as np

    from rocketopt.tolerance import ToleranceSpec, perturb

    base = load_ric(MOTOR)
    built = perturb(base, [ToleranceSpec("burn_rate_a", 0.05)],
                    np.random.default_rng(0))
    ratios = [t2["a"] / t1["a"] for t1, t2
              in zip(base["propellant"]["tabs"], built["propellant"]["tabs"])]
    assert len(set(round(r, 12) for r in ratios)) == 1
    assert abs(ratios[0] - 1.0) > 1e-9, "nothing was perturbed"


def test_perturbation_never_builds_an_impossible_motor():
    """However unlucky the draw, the result must still simulate."""
    import numpy as np

    from rocketopt.simulate import simulate_motor
    from rocketopt.tolerance import ToleranceSpec, perturb

    base = load_ric(MOTOR)
    rng = np.random.default_rng(4)
    wild = [ToleranceSpec("core_diameter", 0.5 * IN),
            ToleranceSpec("throat", 0.5 * IN),
            ToleranceSpec("burn_rate_a", 0.5)]
    for _ in range(25):
        built = perturb(base, wild, rng)
        for grain in built["grains"]:
            props = grain["properties"]
            assert 0 < props["coreDiameter"] < props["diameter"]
        assert built["nozzle"]["exit"] >= built["nozzle"]["throat"]
        assert simulate_motor(built, timestep=0.05) is not None


def test_zero_tolerance_reproduces_the_design_exactly():
    import numpy as np

    from rocketopt.tolerance import ToleranceSpec, perturb

    base = load_ric(MOTOR)
    built = perturb(base, [ToleranceSpec("throat", 0.0)], np.random.default_rng(0))
    assert built["nozzle"]["throat"] == base["nozzle"]["throat"]


def test_propagate_reports_a_pass_rate_and_names_the_risk():
    from rocketopt.spec import ConstraintSpec
    from rocketopt.tolerance import ToleranceSpec, propagate, summarise
    from rocketopt.units import PA_PER_PSI as PSI

    base = load_ric(MOTOR)
    limits = [ConstraintSpec("max_pressure", "<=", 900 * PSI,
                             label="Peak chamber pressure")]
    report = propagate(base, [ToleranceSpec("throat", 0.01 * IN)], limits,
                       samples=40, timestep=0.05, workers=2)
    assert report["available"]
    assert 0.0 <= report["pass_rate"] <= 1.0
    assert report["pass_low"] <= report["pass_rate"] <= report["pass_high"]
    assert report["per_limit"][0]["metric"] == "max_pressure"
    assert isinstance(summarise(report), str)


def test_propagate_says_so_when_nothing_varies():
    from rocketopt.spec import ConstraintSpec
    from rocketopt.tolerance import ToleranceSpec, propagate

    report = propagate(load_ric(MOTOR),
                       [ToleranceSpec("throat", 0.0, enabled=False)],
                       [ConstraintSpec("peak_kn", "<=", 225.0)],
                       samples=10, timestep=0.05)
    assert not report["available"]
