"""Figures for the optimisation report.

Colours come from a validated categorical palette; the first three slots are
the only ones used, because scatter plots put every pair on screen at once and
past three slots the pairs stop being reliably distinguishable to colourblind
readers. Every chart names its single y-quantity -- no dual axes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e2e1dc"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
LIMIT = "#e34948"  # reserved status colour, only ever used for a limit line

PA_PER_PSI = 6894.757293168361


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


def _finish(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def thrust_curves(curves: Dict[str, tuple], path: Path, title: str) -> Path:
    """Overlays thrust-vs-time for a handful of motors."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, (time_axis, force, _)) in enumerate(curves.items()):
        colour = SERIES[i % len(SERIES)]
        ax.plot(time_axis, force, color=colour, label=label)
        # Label at each curve's peak. Labelling the end would stack every label
        # on top of the others, since all the traces converge on zero thrust.
        peak = int(np.argmax(force))
        ax.annotate(
            label, xy=(time_axis[peak], force[peak]), xytext=(0, 9),
            textcoords="offset points", color=colour, fontsize=9,
            ha="center", fontweight="600",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thrust (N)")
    ax.set_title(title)
    ax.legend(loc="lower center", ncol=len(curves), bbox_to_anchor=(0.5, -0.32))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def pareto_front(front: pd.DataFrame, baseline: Dict, path: Path,
                 cloud: Optional[pd.DataFrame] = None,
                 baseline_label: str = "baseline motor") -> Path:
    """Initial thrust against total impulse, with the baseline motor marked."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if cloud is not None and len(cloud):
        ax.scatter(cloud["total_impulse"], cloud["initial_thrust"], s=6,
                   color=GRID, edgecolors="none", zorder=1,
                   label="sampled designs")
    ax.plot(front["total_impulse"], front["initial_thrust"], "-o",
            color=SERIES[0], markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3, label="Pareto front (simulated)")
    ax.scatter([baseline["total_impulse"]], [baseline["initial_thrust"]],
               s=110, color=SERIES[1], edgecolors=SURFACE, linewidths=1.5,
               zorder=4, label=baseline_label, marker="D")
    ax.annotate("baseline", xy=(baseline["total_impulse"], baseline["initial_thrust"]),
                xytext=(10, -14), textcoords="offset points",
                color=SERIES[1], fontsize=9, fontweight="600")
    ax.set_xlabel("Total impulse (N·s)")
    ax.set_ylabel("Initial thrust, mean over first 0.25 s (N)")
    ax.set_title("Initial thrust vs total impulse: what you can actually have")
    ax.legend(loc="lower left")
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def parity(frame: pd.DataFrame, targets: Sequence[str], path: Path,
           units: Optional[Dict[str, str]] = None) -> Path:
    """Predicted vs simulated, one panel per target -- small multiples rather
    than one scatter with many colours."""
    apply_style()
    units = units or {}
    n = len(targets)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, target in zip(axes, targets):
        actual = frame["actual_" + target].to_numpy()
        predicted = frame["pred_" + target].to_numpy()
        ax.scatter(actual, predicted, s=7, alpha=0.35, color=SERIES[0],
                   edgecolors="none")
        lo = float(min(actual.min(), predicted.min()))
        hi = float(max(actual.max(), predicted.max()))
        ax.plot([lo, hi], [lo, hi], color=INK_SOFT, linewidth=1.0, linestyle="--")
        r2 = 1 - np.sum((predicted - actual) ** 2) / np.sum((actual - actual.mean()) ** 2)
        ax.set_title("{}\nR² = {:.4f}".format(target.replace("_", " "), r2),
                     fontsize=10)
        unit = units.get(target, "")
        ax.set_xlabel("simulated" + (" ({})".format(unit) if unit else ""))
        ax.set_ylabel("predicted")
        _finish(ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Surrogate accuracy on held-out designs", fontsize=13,
                 fontweight="600", y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def importance(frame: pd.DataFrame, path: Path, title: str, top: int = 12) -> Path:
    """Horizontal bars, single hue -- rank is the message, not identity."""
    apply_style()
    data = frame.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(data) + 1.6))
    ax.barh(data["feature"], data["importance"], color=SERIES[0], height=0.62)
    ax.set_xlabel("Permutation importance (drop in R²)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def convergence(runs: Dict[str, pd.Series], path: Path, baseline: float) -> Path:
    """Best feasible score so far against simulations spent."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (label, series) in enumerate(runs.items()):
        ax.plot(np.arange(1, len(series) + 1), series.to_numpy(),
                color=SERIES[i % len(SERIES)], label=label)
    ax.axhline(baseline, color=INK_SOFT, linewidth=1.2, linestyle="--")
    ax.annotate("baseline motor", xy=(1, baseline), xytext=(6, 6),
                textcoords="offset points", color=INK_SOFT, fontsize=9)
    ax.set_xlabel("openMotor simulations run")
    ax.set_ylabel("Best feasible score found")
    ax.set_title("How many burns each method needed")
    ax.set_xscale("log")
    ax.legend(loc="center right")
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def margin_sweep(frame: pd.DataFrame, path: Path, baseline: Dict) -> Path:
    """Best achievable initial thrust at each pressure-limit fraction."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    psi = frame["max_pressure"] / PA_PER_PSI
    ax.plot(psi, frame["initial_thrust"], "-o", color=SERIES[0],
            markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
    for _, row in frame.iterrows():
        ax.annotate("{:.0f}%".format(100 * row["pressure_fraction"]),
                    xy=(row["max_pressure"] / PA_PER_PSI, row["initial_thrust"]),
                    xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8, color=INK_SOFT)
    ax.scatter([baseline["max_pressure"] / PA_PER_PSI], [baseline["initial_thrust"]],
               s=110, marker="D", color=SERIES[1], edgecolors=SURFACE,
               linewidths=1.5, zorder=4)
    ax.annotate("baseline", xy=(baseline["max_pressure"] / PA_PER_PSI,
                                baseline["initial_thrust"]),
                xytext=(10, -4), textcoords="offset points",
                color=SERIES[1], fontsize=9, fontweight="600")
    ax.set_xlabel("Peak chamber pressure (psi)")
    ax.set_ylabel("Initial thrust, mean over first 0.25 s (N)")
    ax.set_title("Initial thrust bought per psi of peak pressure")
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


#: Single-hue sequential ramp. Pressure cap is a magnitude, not an identity, so
#: the fronts get one hue light-to-dark rather than six categorical colours.
BLUE_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def margin_fronts(fronts: pd.DataFrame, baseline: Dict, path: Path) -> Path:
    """One Pareto front per pressure ceiling, on shared axes."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fractions = sorted(fronts["pressure_fraction"].unique())
    ramp = [BLUE_RAMP[int(round(i * (len(BLUE_RAMP) - 1) / max(len(fractions) - 1, 1)))]
            for i in range(len(fractions))]
    for colour, fraction in zip(ramp, fractions):
        part = fronts[fronts["pressure_fraction"] == fraction].sort_values("total_impulse")
        label = "{:.0f} psi cap".format(fraction * 1500)
        ax.plot(part["total_impulse"], part["initial_thrust"], "-",
                color=colour, linewidth=2.0, label=label)
        if len(fractions) <= 4:  # past four, the legend alone is less cluttered
            tip = part.nlargest(1, "initial_thrust").iloc[0]
            ax.annotate(label, xy=(tip["total_impulse"], tip["initial_thrust"]),
                        xytext=(6, 2), textcoords="offset points",
                        color=colour, fontsize=8.5, fontweight="600")
    ax.scatter([baseline["total_impulse"]], [baseline["initial_thrust"]],
               s=120, marker="D", color=LIMIT, edgecolors=SURFACE,
               linewidths=1.5, zorder=5)
    ax.annotate("baseline\n{:.0f} N / {:.0f} N·s".format(
                    baseline["initial_thrust"], baseline["total_impulse"]),
                xy=(baseline["total_impulse"], baseline["initial_thrust"]),
                xytext=(12, -26), textcoords="offset points",
                color=LIMIT, fontsize=9, fontweight="600")
    ax.set_xlabel("Total impulse (N·s)")
    ax.set_ylabel("Initial thrust, mean over first 0.25 s (N)")
    ax.set_title("What each pressure ceiling buys you")
    ax.legend(loc="upper right", fontsize=8.5)
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def envelope_fronts(fronts: pd.DataFrame, baseline: Dict, path: Path) -> Path:
    """The trade-off inside a fixed operating envelope, by core arrangement.

    Three series, which is the cap for a plot where every pair of colours is on
    screen at once -- past three, adjacent-pair separation stops being reliable
    for colourblind readers.
    """
    apply_style()
    names = {"free": "free (ties allowed)",
             "paired": "paired (3 mandrel sizes)",
             "strict": "strictly increasing (≥1 mm)"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for colour, key in zip(SERIES, ["free", "paired", "strict"]):
        part = fronts[fronts["arrangement"] == key].sort_values("total_impulse")
        if not len(part):
            continue
        ax.plot(part["total_impulse"], part["initial_thrust"], "-",
                color=colour, linewidth=2.0, label=names[key])
    ax.axvline(baseline["total_impulse"], color=INK_SOFT, linewidth=1,
               linestyle=":", zorder=1)
    ax.axhline(baseline["initial_thrust"], color=INK_SOFT, linewidth=1,
               linestyle=":", zorder=1)
    ax.scatter([baseline["total_impulse"]], [baseline["initial_thrust"]],
               s=130, marker="D", color=LIMIT, edgecolors=SURFACE,
               linewidths=1.5, zorder=5)
    ax.annotate("your motor\n{:.0f} N / {:.0f} N·s".format(
                    baseline["initial_thrust"], baseline["total_impulse"]),
                xy=(baseline["total_impulse"], baseline["initial_thrust"]),
                xytext=(-12, -34), textcoords="offset points", ha="right",
                color=LIMIT, fontsize=9, fontweight="600")
    ax.set_xlabel("Total impulse (N·s)")
    ax.set_ylabel("Initial thrust, mean over first 0.25 s (N)")
    ax.set_title("Inside 500 psi / Kn 225 / 1.05 lb·in⁻²s⁻¹")
    ax.legend(loc="upper right", fontsize=9)
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
