"""Turning a finished report into a PDF.

A report is written as HTML because that is what the figures, tables and type
are laid out in, but a report is a document people file, email and print, so
the artefact they get handed is a PDF.

Rendering is done by whatever Chromium-family browser is already installed.
That is a real dependency, and a deliberate one: the report leans on grid,
custom properties and web fonts, and the pure-Python HTML-to-PDF libraries
render none of those faithfully -- a report that silently comes out looking
wrong is worse than one that says it could not be made.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

#: Where a Chromium-family browser usually lives. Checked in order, first hit
#: wins. Anything that speaks --headless --print-to-pdf will do.
BROWSERS: List[str] = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/microsoft-edge",
]

#: Names to try on PATH, for installs that are not where we looked.
BROWSER_NAMES = ["google-chrome", "chromium", "chromium-browser", "chrome",
                 "microsoft-edge", "brave-browser"]


class NoBrowser(RuntimeError):
    """No Chromium-family browser to render with."""


def find_browser() -> Optional[str]:
    for candidate in BROWSERS:
        if Path(candidate).exists():
            return candidate
    for name in BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def html_to_pdf(html_path: Path, pdf_path: Path, timeout: int = 120) -> Path:
    """Renders one local HTML file to PDF. Raises :class:`NoBrowser` if it cannot.

    The report is self-contained -- figures are embedded as data URIs -- so this
    needs no network beyond the web fonts, which fall back cleanly when offline.
    """
    browser = find_browser()
    if browser is None:
        raise NoBrowser(
            "No Chrome, Chromium, Edge or Brave found to render the PDF with. "
            "The report is still written as HTML next to it.")

    html_path = Path(html_path).resolve()
    pdf_path = Path(pdf_path).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Deliberately no --user-data-dir. Pointing one at a fresh directory makes
    # headless Chrome hang indefinitely on first-run profile setup; without it
    # the render takes about two seconds. --no-sandbox is for CI and containers,
    # where the sandbox cannot start.
    try:
        subprocess.run(
            [browser, "--headless", "--disable-gpu", "--no-sandbox",
             "--no-pdf-header-footer",
             "--print-to-pdf=" + str(pdf_path), html_path.as_uri()],
            check=True, timeout=timeout,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise NoBrowser("{} could not render the report: {}".format(
            Path(browser).name, exc)) from exc

    if not pdf_path.exists():
        raise NoBrowser("The browser ran but produced no PDF.")
    return pdf_path
