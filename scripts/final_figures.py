"""Figures for the 5.00 in motor study."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rocketopt import plotting
from rocketopt.plotting import GRID, INK, INK_SOFT, LIMIT, SERIES, SURFACE
from rocketopt.ric import load_ric
from rocketopt.simulate import curves
from rocketopt.units import M_PER_IN as IN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final"
FIG = OUT / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plotting.apply_style()
    B = json.loads((OUT / "B_cores_and_nozzle.json").read_text())
    picks = json.loads((OUT / "selection.json").read_text())
    front = sorted(B["designs"], key=lambda d: d["total_impulse"])
    baseline = B["baseline"]

    # --- the trade-off curve -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.plot([d["total_impulse"] for d in front], [d["initial_thrust"] for d in front],
            "-", color=SERIES[0], linewidth=2, label="legal designs (60)")
    ax.scatter([p["total_impulse"] for p in picks], [p["initial_thrust"] for p in picks],
               s=64, color=SERIES[0], edgecolors=SURFACE, linewidths=1.6, zorder=4,
               label="selected options")
    for i, p in enumerate(picks, 1):
        ax.annotate(str(i), xy=(p["total_impulse"], p["initial_thrust"]),
                    xytext=(0, 11), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="700", color=SERIES[0])
    ax.scatter([baseline["total_impulse"]], [baseline["initial_thrust"]], s=130,
               marker="D", color=LIMIT, edgecolors=SURFACE, linewidths=1.5, zorder=5)
    ax.annotate("your Current.ric as saved\n860 psi · Kn 323 — outside the limits",
                xy=(baseline["total_impulse"], baseline["initial_thrust"]),
                xytext=(-16, -46), textcoords="offset points", ha="right",
                fontsize=9, color=LIMIT, fontweight="600",
                arrowprops=dict(arrowstyle="-", color=LIMIT, linewidth=0.9,
                                shrinkA=0, shrinkB=6))
    ax.set_xlabel("Total impulse (N·s)")
    ax.set_ylabel("Initial thrust, mean of first 0.35 s (N)")
    ax.set_title("What a 5.00 in × 6.00 in six-grain motor can do inside your limits")
    ax.legend(loc="lower left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "tradeoff.png", bbox_inches="tight"); plt.close(fig)

    # --- thrust curves of the selected options ------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    for i, p in enumerate(picks, 1):
        motor = load_ric(OUT / "motors" / p["file"])
        c = curves(motor, timestep=0.002)
        ax.plot(c["time"], c["thrust"], color=ramp[i - 1], linewidth=1.9,
                label="{} · {:,.0f} N·s".format(i, p["total_impulse"]))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Thrust (N)")
    ax.set_title("Thrust curves, option 1 (shortest) through 7 (longest)")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "curves.png", bbox_inches="tight"); plt.close(fig)

    # --- why the fixed nozzle cannot work -----------------------------------
    import math
    base = load_ric(ROOT / "Data" / "Open Motor Data" / "Current.ric")
    At = math.pi * base["nozzle"]["throat"] ** 2 / 4
    d_in = np.linspace(0.3, 4.6, 200)
    d = d_in * IN
    area = 6 * (math.pi * d * 6 * IN + (math.pi / 2) * ((5 * IN) ** 2 - d ** 2))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(d_in, area / At, color=SERIES[0], linewidth=2.2, label="Kn at ignition")
    ax.axhline(225, color=LIMIT, linewidth=1.4, linestyle="--")
    ax.annotate("Kn limit 225", xy=(0.35, 225), xytext=(0, 8),
                textcoords="offset points", color=LIMIT, fontsize=9, fontweight="600")
    ax.fill_between(d_in, 225, area / At, where=(area / At >= 225),
                    color=LIMIT, alpha=0.10)
    ax.set_xlabel("Core diameter, all six grains (in)")
    ax.set_ylabel("Kn with the 1.2953 in throat")
    ax.set_title("Optimisation A: no core diameter keeps Kn legal, even before the burn starts")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "infeasible.png", bbox_inches="tight"); plt.close(fig)

    print("figures written to", FIG)
    for f in sorted(FIG.glob("*.png")):
        print("  {} ({:.0f} KB)".format(f.name, f.stat().st_size / 1000))


if __name__ == "__main__":
    main()
