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


#: The motor the project was built around. Only a preference -- see
#: :func:`default_motor_path`.
DEFAULT_MOTOR_NAME = "Current.ric"
DEFAULT_MOTOR_DIR = ("Data", "Open Motor Data")


def default_motor_path(root: Path) -> Optional[Path]:
    """The motor to open when nobody named one.

    Everything used to hard-code ``Current.ric``. Rename that file and the app
    boots with nothing loaded and the whole test suite errors out, which is a
    silly way for a tool to break. Prefer the named file, fall back to any .ric
    sitting beside it, and let the caller handle "there are none".
    """
    folder = root.joinpath(*DEFAULT_MOTOR_DIR)
    preferred = folder / DEFAULT_MOTOR_NAME
    if preferred.exists():
        return preferred
    if folder.is_dir():
        for found in sorted(folder.glob("*.ric")):
            return found
    return None


def study_motor(root: Path) -> Path:
    """The exact motor the study scripts were written against.

    Deliberately no fallback. These scripts print figures that are quoted in
    the report and the README; quietly running them against whatever .ric
    happens to be in the folder would make those numbers describe a different
    motor than their captions claim.
    """
    path = root.joinpath(*DEFAULT_MOTOR_DIR) / DEFAULT_MOTOR_NAME
    if not path.exists():
        alternatives = sorted(p.name for p in path.parent.glob("*.ric")) \
            if path.parent.is_dir() else []
        raise FileNotFoundError(
            "{} is missing, and the study scripts are pinned to it so their "
            "published figures stay reproducible.{}".format(
                path,
                "  Found instead: {}.  Either restore the file or pass the "
                "motor you want explicitly.".format(", ".join(alternatives))
                if alternatives else ""))
    return path


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
