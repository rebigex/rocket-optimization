# Rocket Optimization

Machine-learning-assisted design search for solid rocket motors, driven by
[openMotor](https://github.com/reilleya/openMotor)'s internal ballistics engine running
headlessly. Point it at a `.ric`, say which dimensions may move and what you want more
of, and it returns motors you can open in openMotor and machine to the numbers.

```bash
scripts/setup_env.sh              # venv, deps, openMotor, tests
.venv/bin/python app.py           # opens http://localhost:8420
```

![The optimizer](docs/screenshot.png)

**[Field guide](docs/guide.html)** ([PDF](docs/Optimizer-Field-Guide.pdf)) — how to drive it.
**[Worked study](docs/five-inch-study.html)** — two optimisations of a 5-inch motor, one of
which provably has no answer.

Everything reported is simulated, never predicted: a surrogate may propose and the search
may run at a coarse timestep, but every design that reaches you has been re-run in
openMotor at the verification timestep with all safety margins removed.

---

The motor in `Data/Open Motor Data/Current.ric` is the working example throughout:
six 5.00 × 6.00 in BATES grains, uninhibited ends, Mad River Blue propellant.

## What is optimised

Grain outer diameter, length, count and the propellant are hardware — read from the
`.ric` and never changed, by the optimiser or by the app. Nine dimensions are free:

| Variable | Note |
|---|---|
| `core_1 … core_6` | core diameter of each grain, grain 1 forward |
| `throat` | nozzle throat diameter |
| `exit` | exit diameter, floored at `1.15 × throat` |
| `throat_length` | nozzle throat length |

Each takes a **machining step** — 0.01 in, 1/16 in, whatever you actually hold — and the
optimiser only ever returns values on that grid. Bounds, objectives and limits are all
set in the app; the defaults come from the `.ric`'s own `maxPressure`, `maxMassFlux` and
`minPortThroat`.

Objectives are any of sixteen metrics, each maximised, minimised, or driven to a target.
Pick two and the result is a trade-off curve rather than a single motor.

## Two things that shrink the problem

**Grain order is free.** In openMotor's model, impulse, pressure and burn time
depend only on the *multiset* of core diameters, not their order — verified to
1e-16 across 60 permutations by `scripts/verify_ordering.py`. Order changes only
mass flux and port/throat ratio, and both are best with the largest core aft. So
cores are always stored sorted smallest-forward, removing a 720-fold degeneracy.

**Some outputs are closed-form.** Port/throat ratio, initial Kn, propellant mass
and ignition chamber pressure are exact BATES results, computed in
`design.py` rather than learned. Only quantities that require integrating the
whole burn get a model.

## The app — Lior's Really Good™ Rocket Optimizer

```bash
.venv/bin/python app.py          # opens http://localhost:8420 in your browser
```

A local web app for driving all of this without editing Python. You pick which
dimensions may move, what counts as better, and what must never be exceeded;
it searches, verifies, and hands back `.ric` files.

**Machining precision.** Every diameter takes an optional *step* — your grid.
Type `1/16`, `0.05`, or leave it blank for any size. The baseline motor is
imperial to its last digit (cores of exactly 1.600/1.900/2.200 in, a 0.100 in
grid), so the grid is anchored at zero and offers whole fractions of an inch,
which is how tooling actually comes. Snapping composes with the ordering rules:
grouping averages, so snapping follows it, and the minimum core increment is
rounded up to a whole number of grid steps so walking the ladder never leaves
the grid.

**Two search modes.** *Fast* runs a genetic search straight against openMotor —
tens of seconds, and it needs no model. *Map the full trade-off* samples the
space, trains surrogates, runs NSGA-II against them, then re-simulates the
survivors; slower, but you get a curve of options instead of one answer.

**The size of the space, before you run anything.** As soon as bounds and steps are set,
the app shows how many distinct motors the configuration admits. It is not a plain
product of the per-variable counts: cores are stored sorted, so a set of six diameters is
one motor rather than 720, and the count is a multiset coefficient. Ordering rules cut
further, and the exit is counted jointly with the throat because its range depends on it.
The panel also says how many of them the search will actually simulate, and how long
brute force would take — on a 0.01 in grid with all nine dimensions free that is
4.5 × 10¹⁹ motors and 4.4 × 10¹⁰ years, which is the argument for a genetic search in one
line. `sizing.py` carries the combinatorics; `tests/test_sizing.py` checks the formulas
against brute-force enumeration.

**Watch it search.** After Optimize, the workspace shows the live population rather than
a progress bar: every dot is a motor that has actually been simulated, grey if it broke a
limit and blue if it did not, with the current best trade-off drawn through them and a
fading wake of the last dozen generations so you can see which way the search is
travelling. It reads the real evaluated population in real units, not a stand-in
animation, and the whole payload is under 3 KB a second. A run can be reopened while it
is still going with `?job=<id>`.

**Tolerance analysis.** The optimiser works from nominal dimensions, so every design it
returns sits exactly on whatever limits you set. Compare & Safety will build that design
a few hundred times with your shop's tolerances applied — core diameters varying
independently because each is a separate reamer pass, the propellant batch varying as one
draw because every grain comes from the same mix — and report how often it still stays
legal. On the balanced design from the 5-inch study, with ±0.005 in on the throat and
cores and 3% on the burn-rate coefficient, that is **51%**. Nominal is not a guarantee.

Uncertainty is declared against the hardware and the propellant, once, in the rail. It
never needs to know what is being optimised, which is why it needs no per-run setup.

**Several searches, merged.** NSGA-II is stochastic and offers no guarantee it found the
global front. Running the identical configuration three times with different seeds put
best initial thrust 6.4% apart, and 17–62% of each run's front was strictly beaten by
another run's. So a run now splits its budget across independent searches and reports the
non-dominated set of everything they found. That is not just insurance, it is cheaper:
three 4,800-simulation searches merged reached 6,984 N where a single 14,400-simulation
search reached 6,775 N — **+3.1% for the same money**. The simulation budget and the number
of searches are both set in the app; population and generations are derived from them.

**An analytic ceiling.** For initial thrust there is a bound no motor in the space can
beat, and it needs no search: bound one grain's burning area over the whole box, multiply
by the grain count (max of a sum ≤ sum of maxes — peak area is *not* monotone in each
core, so the obvious argument fails), divide by the binding Kn to get the largest useful
throat area, and multiply by the best achievable thrust coefficient and the pressure
limit. On the 5-inch motor that ceiling is **7,517 N**, and the merged search reaches
within 7.1% of it. That is the closest thing to assurance available here — the algorithm
itself provides none.

**Using the limits to prune.** Where the limits are closed-form, the app says what they
rule out before anything is simulated. Burning area for uninhibited BATES is
`π(d+2r)(L−2r) + (π/2)(D²−(d+2r)²)` through the whole burn, so peak Kn can be computed
exactly rather than simulated — it tracks openMotor to within about 1%, and with that
margin applied it becomes a screen that can never reject a design the simulator would
accept. From it come two provable bounds: a **throat floor** (burning area is smallest
with every core at its minimum, so if even that needs a wider throat, nothing narrower can
ever be legal) and a **core ceiling**. Apply-tighter-bounds narrows the spec to them.

The app also detects limits that are secretly the same limit. Chamber pressure is a
monotone function of Kn, so a pressure ceiling and a Kn ceiling are one constraint: on the
5-inch motor, 500 psi is reached at **Kn 224.5** against a Kn ceiling of 225, so pressure
binds by a hair and raising Kn alone would change nothing.

Worth being honest about the size of the win: on that configuration the chain runs
4.5 × 10¹⁹ → 2.5 × 10¹⁹ → 2.1 × 10¹⁹, about 2×. The limits shrink the box far less than
sorting the cores already did (694×). Pruning is worth doing because it costs
microseconds instead of a simulation, not because it makes the space small.

**Technical reports on demand.** Tick any finished runs in the report panel and press
Generate. The document is written from the runs themselves — hardware and limits read off
the motor and the spec, and the trade-off curve and option tables built from the
verified designs. A run that found nothing gets the most attention: which limit could never be met, how close anything got, and — where
burning area is closed-form — a proof that no core diameter would have worked, with the
throat diameter that would. Two runs in one report are compared side by side. Output
lands in `outputs/reports/` and opens in a new tab.

**Four visualization profiles**, switchable from the tab strip:

| Profile | Shows |
|---|---|
| Design Review | thrust curve · pressure + Kn · scaled cross-section · spec sheet · margin bars |
| Trade-off Explorer | clickable option front · every design tried · parallel coordinates · result spread · exportable options table |
| Optimizer Diagnostics | search progress · which limit binds · model accuracy · which dimension matters |
| Compare & Safety | before/after thrust · spec delta · per-grain mass flux · sensitivity tornado |

Nothing shown was merely predicted: the surrogate may propose and the search may
run coarse, but every reported design is re-simulated at 0.002 s with all safety
margins removed. A finished run keeps its own configuration, so editing the form
afterwards cannot mislabel a result, and `?job=<id>` reopens it.

## Pipeline

```bash
scripts/setup_env.sh                          # venv, deps, openMotor, tests
.venv/bin/python app.py                       # the app — everything below, driven

.venv/bin/python scripts/verify_ordering.py   # prove the core-sorting assumption
.venv/bin/python scripts/run_envelope.py      # optimise inside a fixed envelope
.venv/bin/python scripts/big_run.py           # the 100,000-simulation, 8-seed study
.venv/bin/python -m pytest tests/ -q
```

Outputs land in `outputs/`: `results.json`, `figures/*.png`, and `motors/*.ric`
files you can open directly in openMotor.

## Where the machine learning actually earns its keep

A BATES simulation takes ~10 ms, so a surrogate is *not* needed to make search
possible. It is used where it pays:

- **Mapping the trade-off.** NSGA-II needs tens of thousands of evaluations to
  produce a dense initial-thrust/impulse front. The surrogate supplies them in
  seconds; every design on the reported front is then **re-simulated**, so no
  model output is ever reported as a result.
- **Sample efficiency.** Bayesian optimisation with a Gaussian process reaches a
  comparable design in a few hundred simulations instead of thousands.
- **Sensitivity.** Permutation importance says which levers matter, and by how
  much.

Initial thrust is very nearly analytic — it is set by initial Kn and throat area,
which is why the surrogate predicts it to R² 0.9997. The genuinely learned
quantities are total impulse, peak pressure and peak mass flux, which depend on
how the burn evolves.

## Layout

```
app.py           entry point -- starts the server, opens the browser
app/
  server.py      HTTP surface: load motor, defaults, run, poll, export
  jobs.py        background runs and their progress
  static/        the interface; charts.js declares the panels and profiles
src/rocketopt/
  ric.py         read/write .ric without PyQt
  units.py       inch/mm conversion, shop-fraction parsing, grid snapping
  spec.py        what the GUI configures: variables, objectives, constraints
  design.py      design space, canonical form, closed-form features
  simulate.py    headless openMotor runs reduced to metrics
  sampling.py    Sobol + structured sampling, persistent worker pool
  surrogate.py   per-target models, scoring, permutation importance
  optimize.py    direct GA, Bayesian optimisation, NSGA-II + verification
  runner.py      one configuration -> one verified set of results
  sizing.py      how many distinct motors a configuration admits
  tolerance.py   what a design does when it is built, not drawn
  report.py      the technical report, derived entirely from the runs
  report_style.py  the report stylesheet, kept as data
  plotting.py    figures for the static report
tests/           machining grid, frozen dimensions, ordering rules
docs/            the field guide, the worked study, recommended motors
vendor/openMotor cloned at setup, GPLv3, never committed
```

## Search fidelity

The search runs at a 0.01 s timestep and verification at 0.002 s. Those disagree
slightly, and in different directions per metric: peak mass flux is a finite
difference so it grows as the timestep shrinks (-0.68% at 0.01 s), total impulse
is an integral (-0.18%), and pressure and Kn are exactly invariant. Rather than
guess a safety margin, `runner.timestep_bias` measures the ratio on your own
motor and restates the search-time limits by it, so a design sitting on a limit
during the search is still sitting on it after verification.

## Licence

This project is MIT (see `LICENSE`). openMotor is GPLv3 and is **not** redistributed
here — `setup_env.sh` clones it at a pinned commit. Read [`NOTICE.md`](NOTICE.md) before
making this repository public: the code imports `motorlib` directly, and whether that
makes it a derivative work is the usual unsettled question about linking to GPL code.
