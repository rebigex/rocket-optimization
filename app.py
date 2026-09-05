#!/usr/bin/env python
"""Starts Lior's Really Good™ Rocket Optimizer and opens it in a browser.

Run it with `python app.py`. On a machine that has never run it before it offers
to build the environment first, then starts itself inside it -- so this is the
only command anybody needs.
"""
import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

HOST = "127.0.0.1"
PORT = 8420


def ready() -> bool:
    """Can this interpreter run the app as it stands?"""
    try:
        import fastapi        # noqa: F401
        import uvicorn        # noqa: F401
        from motorlib.motor import Motor        # noqa: F401
    except Exception:
        return False
    return True


def relaunch_in_environment() -> None:
    """Builds the environment if needed, then re-runs this file inside it.

    Handing over with execv rather than importing across interpreters: the app
    then runs under the environment's own Python, with nothing of the outer
    one's import state carried in.
    """
    import bootstrap

    python = bootstrap.ensure(ROOT, assume_yes="--yes" in sys.argv or "-y" in sys.argv)
    # Compared by prefix, not by resolving the paths: a venv's bin/python is a
    # symlink to the interpreter it was built from, so resolve() makes the two
    # look like the same file and the handover never happens.
    if sys.prefix == str(ROOT / ".venv"):
        raise SystemExit("The environment is built but still incomplete.")
    print("\n  Starting under {}\n".format(os.path.relpath(python, ROOT)))
    # execv replaces the process without flushing Python's buffers, so anything
    # printed above is lost unless it is pushed out first.
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(str(python), [str(python), str(ROOT / "app.py")]
             + [a for a in sys.argv[1:] if a not in ("--yes", "-y")])


def main() -> None:
    import uvicorn

    url = "http://{}:{}".format(HOST, PORT)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print("\n  Lior's Really Good™ Rocket Optimizer  ->  {}\n".format(url))
    uvicorn.run("app.server:app", host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    if not ready():
        relaunch_in_environment()
    main()
