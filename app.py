#!/usr/bin/env python
"""Starts Lior's Really Good™ Rocket Optimizer and opens it in a browser.

Run it with `python app.py`. Nothing else to configure.
"""
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

HOST = "127.0.0.1"
PORT = 8420


def main() -> None:
    import uvicorn

    url = "http://{}:{}".format(HOST, PORT)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print("\n  Lior's Really Good™ Rocket Optimizer  ->  {}\n".format(url))
    uvicorn.run("app.server:app", host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
