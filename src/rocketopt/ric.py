"""Read and write openMotor ``.ric`` motor files.

openMotor's own loader lives in ``uilib.fileIO``, which imports PyQt. We only
need the YAML payload, so this module reproduces the file's two Python-specific
tags directly and stays headless.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

# uilib.fileIO.fileTypes.MOTOR -- the enum value openMotor stamps on .ric files
MOTOR_FILE_TYPE = 3
RIC_VERSION = (0, 6, 2)


class _RicLoader(yaml.SafeLoader):
    """SafeLoader that understands the two python-specific tags in a .ric."""


_RicLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)
_RicLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/apply:",
    lambda loader, suffix, node: loader.construct_sequence(node, deep=True),
)


class _FileType:
    """Serialises back to the ``fileTypes`` enum tag openMotor expects."""

    def __init__(self, value: int) -> None:
        self.value = value


class _VersionTuple:
    def __init__(self, parts) -> None:
        self.parts = tuple(parts)


class _RicDumper(yaml.SafeDumper):
    """SafeDumper that emits the tags openMotor's loader requires."""


_RicDumper.add_representer(
    _FileType,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:python/object/apply:uilib.fileIO.fileTypes", [data.value]
    ),
)
_RicDumper.add_representer(
    _VersionTuple,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:python/tuple", list(data.parts)
    ),
)
# Design vectors arrive as numpy scalars; openMotor's loader wants plain numbers.
_RicDumper.add_representer(
    np.float64, lambda dumper, data: dumper.represent_float(float(data))
)
_RicDumper.add_representer(
    np.int64, lambda dumper, data: dumper.represent_int(int(data))
)
for _np_type in (np.float32, np.int32, np.int16, np.bool_):
    _RicDumper.add_representer(
        _np_type,
        lambda dumper, data: dumper.represent_data(data.item()),
    )


#: Where the motor lives. Whatever .ric sits here is the motor being optimised --
#: no file name is special, and nothing in this project names one. Singular
#: because one motor is optimised at a time, which is what motor_path assumes.
MOTOR_DIR = ("motor",)


def motor_path(root: Path) -> Path:
    """The motor to work on: whatever .ric is in the motor folder.

    Drop a file in and it is the one that gets optimised. If several are there
    the newest wins, since that is the one you just put in, and the app names
    the file it opened so there is never a question which.
    """
    found = find_motors(root)
    if not found:
        raise FileNotFoundError(
            "No .ric file in {}. Put the motor you want to optimise there, or "
            "load one from the app.".format(root.joinpath(*MOTOR_DIR)))
    return found[0]


def find_motors(root: Path) -> list:
    """Every motor in the folder, newest first."""
    folder = root.joinpath(*MOTOR_DIR)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.ric"), key=lambda p: -p.stat().st_mtime)


def load_ric(path: str | Path) -> Dict[str, Any]:
    """Returns the ``data`` dict of a .ric file, ready for ``Motor(...)``."""
    with open(path, "r") as handle:
        raw = yaml.load(handle, Loader=_RicLoader)
    if "data" not in raw:
        raise ValueError("{} is not an openMotor motor file".format(path))
    return raw["data"]


def save_ric(path: str | Path, motor: Dict[str, Any]) -> Path:
    """Writes a motor dict out as a .ric that openMotor can open directly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "data": _plain(motor),
        "type": _FileType(MOTOR_FILE_TYPE),
        "version": _VersionTuple(RIC_VERSION),
    }
    with open(path, "w") as handle:
        yaml.dump(document, handle, Dumper=_RicDumper, default_flow_style=False)
    return path


def _plain(value):
    """Recursively converts numpy scalars to Python scalars."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def clone(motor: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(motor)
