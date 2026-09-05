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

#: The pinned dependency set publishes wheels for these versions and no others
#: -- numpy 1.26, scipy 1.13 and scikit-image 0.24 all stop at cp312. PyPI
#: metadata says only ">=3.9", with no upper bound, so on a newer Python pip
#: does not refuse: it tries to compile numpy from source, which fails on a
#: machine without a full C/Fortran toolchain and buries the reason in a meson
#: log. Refusing up front, with the version to install, is the whole point of
#: this check.
MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 12)

#: Tried in order when the interpreter running this is out of range. The
#: Windows launcher is asked for specific versions; elsewhere the names are.
CANDIDATE_PYTHONS = ["python3.12", "python3.11", "python3.10", "python3.9"]
WINDOWS_LAUNCHER_VERSIONS = ["3.12", "3.11", "3.10", "3.9"]


def version_of(python) -> Optional[tuple]:
    """The (major, minor) that interpreter reports, or None if it will not run."""
    try:
        out = subprocess.run(
            [str(python), "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return tuple(int(x) for x in out.stdout.split())
    except ValueError:
        return None


def supported(version: Optional[tuple]) -> bool:
    return bool(version) and MIN_PYTHON <= version <= MAX_PYTHON


def find_supported_python() -> Optional[List[str]]:
    """A command on this machine whose Python the pinned wheels cover.

    Returns the command as a list, because the Windows launcher takes the
    version as an argument rather than being a differently named executable.
    """
    if supported(sys.version_info[:2]):
        return [sys.executable]
    for name in CANDIDATE_PYTHONS:
        found = shutil.which(name)
        if found and supported(version_of(found)):
            return [found]
    launcher = shutil.which("py")
    if launcher:
        for wanted in WINDOWS_LAUNCHER_VERSIONS:
            probe = subprocess.run(
                [launcher, "-" + wanted, "-c", "import sys; print(sys.version_info[1])"],
                capture_output=True, text=True)
            if probe.returncode == 0:
                return [launcher, "-" + wanted]
    return None


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
    if find_supported_python() is None:
        problems.append(
            "Python between {}.{} and {}.{} -- this is {}.{}, and no other was "
            "found on this machine.\n"
            "    The pinned versions of numpy, scipy and scikit-image only "
            "publish wheels up to {}.{};\n"
            "    on anything newer pip tries to compile them from source, which "
            "needs a full C and\n"
            "    Fortran toolchain and is what the numpy/meson error you may have "
            "just seen was.\n"
            "    Install {}.{} from https://www.python.org/downloads/ and run "
            "this again.".format(
                MIN_PYTHON[0], MIN_PYTHON[1], MAX_PYTHON[0], MAX_PYTHON[1],
                sys.version_info[0], sys.version_info[1],
                MAX_PYTHON[0], MAX_PYTHON[1], MAX_PYTHON[0], MAX_PYTHON[1]))
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

    # A .venv left over from an attempt on an unsupported Python is worse than
    # none: its interpreter is the wrong version, and installing into it fails
    # exactly the way it failed the first time. Replace it rather than reuse it.
    existing = version_of(python) if python.exists() else None
    if existing and not supported(existing):
        print("  replacing .venv, which was built with Python {}.{}".format(*existing))
        shutil.rmtree(root / ".venv", ignore_errors=True)

    if not venv_python(root).exists():
        base = find_supported_python()
        if base is None:
            raise SystemExit("No Python this dependency set supports.")
        if base != [sys.executable]:
            print("  using {} for the environment ({}.{} is out of range)".format(
                " ".join(base), *sys.version_info[:2]))
        _run(base + ["-m", "venv", str(root / ".venv")], what="creating .venv")
        python = venv_python(root)

    _run([str(python), "-m", "pip", "install", "--upgrade", "--quiet", "pip"],
         what="upgrading pip")
    try:
        _run([str(python), "-m", "pip", "install", "--quiet", "-r",
              str(root / "requirements.txt")], what="installing dependencies")
    except subprocess.CalledProcessError:
        built = version_of(python)
        raise SystemExit(
            "\nInstalling the dependencies failed. The output above says why.\n"
            "If it mentions building numpy, scipy or scikit-image from source, "
            "the\nenvironment is on Python {}, and the pinned versions only ship "
            "wheels up to\n{}.{}. Delete .venv, install Python {}.{}, and run this "
            "again.".format(
                "{}.{}".format(*built) if built else "an unsupported version",
                MAX_PYTHON[0], MAX_PYTHON[1], MAX_PYTHON[0], MAX_PYTHON[1]))

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
        try:
            answer = input("  Build it now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D or Ctrl-C at the prompt is an answer, not a crash.
            raise SystemExit("\n  Nothing was changed.")
        if answer in ("n", "no"):
            raise SystemExit("  Nothing was changed.")
    print()
    return build(root)


if __name__ == "__main__":
    ready = ensure(assume_yes="--yes" in sys.argv or "-y" in sys.argv,
                   force="--force" in sys.argv)
    print("\n  Ready. Start the app with:  {} app.py\n".format(
        os.path.relpath(ready, ROOT)))
