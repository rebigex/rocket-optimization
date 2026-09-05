"""The look of every figure in the report.

Colours come from a validated categorical palette; the first three slots are
the only ones used, because scatter plots put every pair on screen at once and
past three slots the pairs stop being reliably distinguishable to colourblind
readers. Every chart names its single y-quantity -- no dual axes.

This used to hold the charts as well. They drew the figures for a set of study
scripts that no longer exist; :mod:`rocketopt.report` and :mod:`rocketopt.bundle`
draw their own against the palette below.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e2e1dc"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
LIMIT = "#e34948"  # reserved status colour, only ever used for a limit line



def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "text.color": INK,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.titlepad": 12,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "figure.dpi": 140,
        }
    )
