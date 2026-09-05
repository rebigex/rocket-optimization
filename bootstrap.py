"""Builds the environment this app needs, on a machine that has none of it.

Two rules shape everything here.

**It never installs into the interpreter you happened to run.** Everything goes
into ``.venv`` beside this file. Silently adding a dozen packages to somebody's
system Python is not a convenience, it is a mess someone else has to clean up.

**It asks first.** Installing software is a change to your machine, so it says
what it is about to do and waits, unless you pass ``--yes`` or run
``scripts/setup_env.sh``, which is the same thing with the answer already given.

Standard library only: it has to run before anything is installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent

#: openMotor is GPLv3 and is not redistributed here. It is cloned, pinned, so
#: the two stay separable. See NOTICE.md.
OPENMOTOR_REPO = "https://github.com/reilleya/openMotor.git"
OPENMOTOR_COMMIT = "0dfb3f1dd4f843499c7f71dc85a3dfde5dd15c6a"

#: Enough of the dependency set to tell a built environment from a bare one.
#: motorlib is last because it is the one that needs a compiler.
REQUIRED = ("fastapi", "uvicorn", "numpy", "pandas", "pymoo", "sklearn",
            "matplotlib", "plotly", "motorlib.motor")

MIN_PYTHON = (3, 9)


def venv_python(root: Path = ROOT) -> Path:
    """Where the environment's interpreter lives, on this platform."""
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def can_import(python: Path, modules=REQUIRED) -> bool:
    """True when that interpreter can import everything the app needs."""
    if not Path(python).exists():
        return False
    code = "import " + ", ".join(modules)
    result = subprocess.run([str(python), "-c", code],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def missing_prerequisites() -> List[str]:
    """Things that have to be installed by hand, with how to do it."""
    problems = []
    if sys.version_info < MIN_PYTHON:
        problems.append(
            "Python {}.{} or newer (this is {}.{}).".format(
                MIN_PYTHON[0], MIN_PYTHON[1], *sys.version_info[:2]))
    if shutil.which("git") is None:
        problems.append(
            "git, to fetch openMotor. macOS: xcode-select --install. "
            "Debian/Ubuntu: sudo apt install git. Windows: https://git-scm.com")
    return problems


def _run(command, cwd: Optional[Path] = None, what: str = "",
         quiet: bool = True) -> None:
    """Runs a build step, showing its output only when it fails.

    Compiling openMotor's extension prints several hundred lines of clang
    invocation. That is noise when it works and the only useful thing when it
    does not, so it is held back and printed on failure.
    """
    print("  {}".format(what or " ".join(str(c) for c in command)))
    result = subprocess.run(
        command, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None, text=True)
    if result.returncode != 0:
        if quiet and result.stdout:
            print(result.stdout)
        raise subprocess.CalledProcessError(result.returncode, command)


def build(root: Path = ROOT) -> Path:
    """Creates the environment. Returns the interpreter to run the app with."""
    python = venv_python(root)

    if not python.exists():
        _run([sys.executable, "-m", "venv", str(root / ".venv")],
             what="creating .venv")

    _run([str(python), "-m", "pip", "install", "--upgrade", "--quiet", "pip"],
         what="upgrading pip")
    _run([str(python), "-m", "pip", "install", "--quiet", "-r",
          str(root / "requirements.txt")], what="installing dependencies")

    vendor = root / "vendor" / "openMotor"
    if not (vendor / ".git").exists():
        vendor.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--quiet", OPENMOTOR_REPO, str(vendor)],
             what="cloning openMotor (GPLv3, not redistributed with this project)")
    subprocess.run(["git", "-C", str(vendor), "fetch", "--quiet", "origin",
                    OPENMOTOR_COMMIT], check=False)
    _run(["git", "-C", str(vendor), "checkout", "--quiet", OPENMOTOR_COMMIT],
         what="pinning openMotor to {}".format(OPENMOTOR_COMMIT[:7]))

    # motorlib ships a Cython extension whose setup.py is too old for pip to
    # install editable, so build it in place and put it on the path with a .pth.
    try:
        _run([str(python), "setup.py", "build_ext", "--inplace"], cwd=vendor,
             what="building openMotor's native extension")
    except subprocess.CalledProcessError:
        raise SystemExit(
            "\nopenMotor's extension did not build. That step needs a C compiler:\n"
            "  macOS          xcode-select --install\n"
            "  Debian/Ubuntu  sudo apt install build-essential python3-dev\n"
            "  Windows        Microsoft C++ Build Tools\n"
            "Install one and run this again.")

    site = subprocess.run(
        [str(python), "-c",
         "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True).stdout.strip()
    (Path(site) / "openmotor_vendor.pth").write_text(str(vendor) + "\n")

    # The app serves Plotly from disk so it works offline. The JS ships inside
    # the plotly package; copy it rather than pulling from a CDN.
    plotly_js = subprocess.run(
        [str(python), "-c",
         "import plotly, pathlib; print(pathlib.Path(plotly.__file__).parent "
         "/ 'package_data' / 'plotly.min.js')"],
        capture_output=True, text=True, check=True).stdout.strip()
    target = root / "app" / "static" / "vendor"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plotly_js, target / "plotly.min.js")
    print("  copied plotly.min.js for offline use")

    if not can_import(python):
        raise SystemExit("\nThe environment built but still cannot import "
                         "everything. Run scripts/setup_env.sh to see the errors.")
    return python


def ensure(root: Path = ROOT, assume_yes: bool = False,
           force: bool = False) -> Path:
    """Returns a ready interpreter, building the environment if needed.

    ``force`` rebuilds even when the environment imports cleanly. The readiness
    check only proves the modules it knows about are present, so it cannot see a
    new line in requirements.txt -- which is exactly what someone re-running the
    setup script is usually trying to pick up.
    """
    python = venv_python(root)
    if can_import(python) and not force:
        return python

    problems = missing_prerequisites()
    if problems:
        raise SystemExit("\nThis needs a couple of things first:\n\n"
                         + "\n".join("  - " + p for p in problems) + "\n")

    print("\n  This is the first run, so the environment has to be built.\n"
          "  It creates .venv here, installs the packages in requirements.txt\n"
          "  into it, and clones openMotor. Nothing outside this folder is\n"
          "  touched, and your system Python is left alone.\n")
    if not assume_yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                "  Run scripts/setup_env.sh, or python bootstrap.py --yes,\n"
                "  to build it without being asked.\n")
        if input("  Build it now? [Y/n] ").strip().lower() in ("n", "no"):
            raise SystemExit("  Nothing was changed.")
    print()
    return build(root)


if __name__ == "__main__":
    ready = ensure(assume_yes="--yes" in sys.argv or "-y" in sys.argv,
                   force="--force" in sys.argv)
    print("\n  Ready. Start the app with:  {} app.py\n".format(
        os.path.relpath(ready, ROOT)))
