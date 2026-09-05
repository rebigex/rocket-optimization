"""A motor built in code, for tests to run against.

The repository ships no motor. ``motor/`` is yours and is not committed, and the
app is expected not to work until you put something there.

Testing a motor optimiser still needs a motor, so this constructs one: round
textbook dimensions that are nobody's design, and Nakka KNSB exactly as
published in openMotor's own default propellant library. Built rather than
committed, so there is no .ric file in this repository at all.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict

IN = 0.0254
PA_PER_PSI = 6894.757293168361
#: 1 lb/(in^2*s) in SI, for the mass-flux limit.
LB_IN2S = 703.06957829636

#: Six grains because the ordering and grouping rules are written for a stack;
#: 4 in bore and a 1.3 in throat because the result is an unremarkable, legal
#: motor with no warnings, which is what a test wants to start from.
GRAIN_COUNT = 6
CORES_IN = (1.80, 1.80, 2.00, 2.00, 2.20, 2.20)
OUTER_IN, LENGTH_IN = 4.00, 5.00
THROAT_IN, EXIT_IN, THROAT_LENGTH_IN = 1.30, 2.20, 0.25


def _knsb(root: Path) -> Dict:
    """Nakka KNSB, read from openMotor's own defaults rather than retyped."""
    source = root / "vendor" / "openMotor" / "uilib" / "defaults.py"
    text = source.read_text()
    opening = text.index("{", text.index("KNSB_PROPS = {"))
    depth = 0
    for index in range(opening, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return ast.literal_eval(text[opening:index + 1])
    raise RuntimeError("could not read KNSB_PROPS from {}".format(source))


def build(root: Path) -> Dict:
    """The sample motor, as the dict a .ric holds."""
    return {
        "config": {
            "ambPressure": 101325.0,
            "burnoutThrustThres": 0.1,
            "burnoutWebThres": 0.001 * IN,
            "flowSeparationWarnPercent": 0.05,
            "mapDim": 750,
            "maxMachNumber": 0.7,
            "maxMassFlux": 1.5 * LB_IN2S,
            "maxPressure": 1000.0 * PA_PER_PSI,
            "minPortThroat": 1.4,
            "sepPressureRatio": 0.4,
            "timestep": 0.03,
        },
        "grains": [
            {"type": "BATES",
             "properties": {"diameter": OUTER_IN * IN, "length": LENGTH_IN * IN,
                            "coreDiameter": core * IN, "inhibitedEnds": "Neither"}}
            for core in CORES_IN
        ],
        "nozzle": {
            "throat": THROAT_IN * IN, "exit": EXIT_IN * IN,
            "throatLength": THROAT_LENGTH_IN * IN,
            "convAngle": 30.0, "divAngle": 15.0, "efficiency": 0.95,
            "erosionCoeff": 0.0, "slagCoeff": 0.0,
        },
        "propellant": _knsb(root),
    }


def write(root: Path, path: Path) -> Path:
    """Writes the sample motor to ``path`` as a .ric."""
    import sys
    sys.path.insert(0, str(root / "src"))
    from rocketopt.ric import save_ric

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ric(path, build(root))
    return path


#: Written once per process, into a temporary directory that lives as long as
#: the process does. Tests take a path rather than a dict because that is what
#: load_ric and the study scripts take.
_CACHE: Dict[str, Any] = {}


def path(root: Path) -> Path:
    """The sample motor as a file, written on first use."""
    if "path" not in _CACHE:
        import tempfile
        holder = tempfile.TemporaryDirectory(prefix="rocketopt-sample-")
        _CACHE["holder"] = holder            # keep it alive for the process
        _CACHE["path"] = write(root, Path(holder.name) / "sample-motor.ric")
    return _CACHE["path"]


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[1]
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "motor" / "sample.ric"
    print(write(root, target))
