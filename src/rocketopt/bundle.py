"""One report per design on the trade-off curve, zipped for sending.

A run's report argues about the curve as a whole: what the trade is, which
limit binds, why one end is unreachable. That is the right document for
deciding. It is the wrong document for handing someone a motor, because the
motor they are being handed is one row of one table in it.

So this makes the other document: a one-page sheet per design, with its
dimensions, what it does, how much room it has against each limit, and where it
sits on the curve. Sixty of those in a zip is a set of options someone can
actually circulate.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .pdf import NoBrowser, html_to_pdf
from .report import (ReportRun, data_uri, display, esc, inches, inches_exact,
                     metric_label, metric_unit)
from .report_style import CSS, FONT_LINK
from .simulate import PA_PER_PSI
from .units import KG_M2S_PER_LB_IN2S as LB

#: Never render more than this many. The front is normally a few dozen; a
#: pathological one should not turn a button into an hour of rendering.
MAX_SHEETS = 60

ProgressFn = Callable[[int, int, str], None]


def _noop(done: int, total: int, message: str) -> None:
    return None


def sheet_name(design: Dict, index: int) -> str:
    """A file name that sorts in curve order and says what it is."""
    designation = "".join(c for c in str(design.get("designation", ""))
                          if c.isalnum()) or "design"
    return "{:02d}-{}".format(index + 1, designation)


def _curve_figure(design: Dict, path: Path) -> Optional[Path]:
    """Thrust and chamber pressure against time, for this one motor."""
    curves = design.get("curves") or {}
    time = curves.get("time") or []
    if len(time) < 2:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import plotting
    from .plotting import SERIES

    plotting.apply_style()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.4, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    top.plot(time, curves["thrust"], color=SERIES[0], linewidth=1.8)
    top.fill_between(time, curves["thrust"], color=SERIES[0], alpha=0.10)
    top.set_ylabel("Thrust (N)")
    bottom.plot(time, [p / PA_PER_PSI for p in curves["pressure"]],
                color=SERIES[1], linewidth=1.6)
    bottom.set_ylabel("Chamber (psi)")
    bottom.set_xlabel("Time (s)")
    for axis in (top, bottom):
        axis.margins(x=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _position_figure(designs: Sequence[Dict], index: int, metrics: Sequence[str],
                     path: Path) -> Optional[Path]:
    """The whole curve, with this design marked. Context in one glance."""
    if len(designs) < 2 or len(metrics) < 2:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import plotting
    from .plotting import LIMIT, SERIES, SURFACE

    plotting.apply_style()
    ax_x, ax_y = metrics[1], metrics[0]
    order = sorted(designs, key=lambda d: d[ax_x])
    xs = [display(ax_x, d[ax_x]) for d in order]
    ys = [display(ax_y, d[ax_y]) for d in order]

    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.plot(xs, ys, "-o", color=SERIES[0], markersize=3.2, linewidth=1.5,
            markeredgecolor=SURFACE, markeredgewidth=0.7)
    here = designs[index]
    ax.scatter([display(ax_x, here[ax_x])], [display(ax_y, here[ax_y])],
               s=120, marker="D", color=LIMIT, edgecolors=SURFACE,
               linewidths=1.4, zorder=5)
    ax.set_xlabel("{} ({})".format(metric_label(ax_x), metric_unit(ax_x)))
    ax.set_ylabel("{} ({})".format(metric_label(ax_y), metric_unit(ax_y)))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _margins(design: Dict, constraints) -> List[Dict]:
    """How much room this design has left against each limit."""
    rows = []
    for spec in constraints:
        value = design.get(spec.metric)
        if value is None:
            continue
        limit = spec.value
        if spec.op == "<=":
            used = value / limit if limit else 0.0
            room = 1.0 - used
        else:
            used = limit / value if value else 0.0
            room = 1.0 - used
        rows.append({
            "label": metric_label(spec.metric),
            "unit": metric_unit(spec.metric),
            "value": display(spec.metric, value),
            "limit": display(spec.metric, limit),
            "op": "≤" if spec.op == "<=" else "≥",
            "room": max(min(room, 1.0), 0.0),
            "tight": room < 0.05,
        })
    return rows


def design_html(design: Dict, index: int, run: ReportRun, base_motor: Dict,
                figures: Dict[str, Path]) -> str:
    designs = run.designs
    metrics = run.result.get("stats", {}).get("objective_labels", [])
    grain = base_motor["grains"][0]["properties"]

    cores = "".join(
        "<tr><td>Grain {}</td><td class=\"n\">{}</td></tr>".format(i + 1, inches_exact(c))
        for i, c in enumerate(design.get("cores", [])))

    performance = [
        ("Initial thrust", "{:,.0f} N".format(design.get("initial_thrust", 0))),
        ("Total impulse", "{:,.0f} N·s".format(design.get("total_impulse", 0))),
        ("Peak thrust", "{:,.0f} N".format(design.get("peak_thrust", 0))),
        ("Burn time", "{:.2f} s".format(design.get("burn_time", 0))),
        ("Specific impulse", "{:.0f} s".format(design.get("isp", 0))),
        ("Propellant mass", "{:.2f} kg".format(design.get("prop_mass", 0))),
        ("Designation", esc(design.get("designation", "—"))),
    ]
    perf = "".join("<dt>{}</dt><dd>{}</dd>".format(esc(k), v) for k, v in performance)

    geometry = [
        ("Grain outer diameter", inches(grain["diameter"]) + " in"),
        ("Grain length", inches(grain["length"]) + " in each"),
        ("Throat diameter", inches_exact(design.get("throat", 0))),
        ("Exit diameter", inches_exact(design.get("exit", 0))),
        ("Throat length", inches_exact(design.get("throat_length", 0))),
        ("Expansion ratio", "{:.2f}".format(design.get("expansion_ratio", 0))),
    ]
    geom = "".join("<dt>{}</dt><dd>{}</dd>".format(esc(k), v) for k, v in geometry)

    bars = ""
    for row in _margins(design, run.spec.enabled_constraints):
        bars += (
            "<tr><td>{label}</td>"
            "<td class=\"n\">{value:,.3g}</td>"
            "<td class=\"n\">{op} {limit:,.3g}</td>"
            "<td class=\"n\">{room:.0f}%</td>"
            "<td class=\"bar\"><span style=\"display:block;width:{room:.0f}%\" "
            "class=\"{cls}\"></span></td></tr>"
        ).format(label=esc(row["label"]), value=row["value"], op=row["op"],
                 limit=row["limit"], room=100 * row["room"],
                 cls="tight" if row["tight"] else "")

    figure_blocks = ""
    if figures.get("curve"):
        figure_blocks += (
            "<figure><img src=\"{}\" alt=\"Thrust and pressure against time\">"
            "<figcaption>Thrust and chamber pressure through the burn, simulated "
            "at the verification timestep.</figcaption></figure>".format(
                data_uri(figures["curve"])))
    if figures.get("position"):
        figure_blocks += (
            "<figure><img src=\"{}\" alt=\"Where this design sits on the curve\">"
            "<figcaption>Where this design sits among the {} legal designs from "
            "the same run.</figcaption></figure>".format(
                data_uri(figures["position"]), len(designs)))

    title = "{} — design {} of {}".format(
        design.get("designation", "Motor"), index + 1, len(designs))
    trade = ""
    if len(metrics) >= 2:
        trade = " Ranked on {} against {}.".format(
            metric_label(metrics[0]).lower(), metric_label(metrics[1]).lower())

    body = """<header>
  <p class="eyebrow">{n} × BATES {d} × {l} in · {prop}</p>
  <h1>{title}</h1>
  <p class="byline">Created by Lior Benshoshan · from {label}</p>
  <p class="lede">One design off the trade-off curve, with the numbers you would
  machine to.{trade}</p>
</header>
<hr class="rule">
<section>
  <h2>Cut these</h2>
  <div class="scroll"><table>
    <thead><tr><th>Core diameter</th><th class="n">in</th></tr></thead>
    <tbody>{cores}</tbody>
  </table></div>
  <dl class="spec">{geom}</dl>
</section>
<section>
  <h2>What it does</h2>
  <dl class="spec">{perf}</dl>
  {figures}
</section>
<section>
  <h2>Room against each limit</h2>
  <div class="scroll"><table>
    <thead><tr><th>Limit</th><th class="n">This design</th><th class="n">Limit</th>
      <th class="n">Room</th><th>&nbsp;</th></tr></thead>
    <tbody>{bars}</tbody>
  </table></div>
  <p class="prose" style="margin-top:14px;font-size:.92rem">Room is how far this
  design sits from the limit, as a share of it. A bar in orange has under 5%
  left, which is inside what machining tolerance alone can move.</p>
</section>
<footer>
  Simulated in openMotor at the verification timestep, every search-time margin
  removed. Lior&#8217;s Really Good&#8482; Rocket Optimizer &middot; created by
  Lior Benshoshan
</footer>""".format(
        n=len(base_motor["grains"]), d=inches(grain["diameter"]),
        l=inches(grain["length"]), prop=esc(base_motor["propellant"]["name"]),
        title=esc(title), label=esc(run.label), trade=esc(trade),
        cores=cores, geom=geom, perf=perf, bars=bars, figures=figure_blocks)

    extra = """
.bar{width:130px}
.bar span{height:8px;border-radius:2px;background:var(--good);display:block}
.bar span.tight{background:var(--ember)}
"""
    return "<title>{}</title>\n{}\n<style>{}{}</style>\n<div class=\"wrap\">{}</div>".format(
        esc(title), FONT_LINK, CSS, extra, body)


def build_bundle(run: ReportRun, base_motor: Dict, out_path: Path,
                 on_progress: ProgressFn = _noop) -> Path:
    """Writes one report per design and zips them.

    Rendering is the slow part -- a browser launch per sheet -- so progress is
    reported per design rather than left as a silent wait.
    """
    designs = run.designs[:MAX_SHEETS]
    if not designs:
        raise ValueError("That run found no legal designs, so there is nothing "
                         "to write sheets for.")
    metrics = run.result.get("stats", {}).get("objective_labels", [])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(designs)
    with tempfile.TemporaryDirectory(prefix="rocketopt-bundle-") as staging:
        stage = Path(staging)
        made: List[Path] = []
        failures = 0
        for index, design in enumerate(designs):
            name = sheet_name(design, index)
            on_progress(index, total, "Rendering {} ({} of {})".format(
                name, index + 1, total))
            figures = {
                "curve": _curve_figure(design, stage / (name + "-curve.png")),
                "position": _position_figure(designs, index, metrics,
                                             stage / (name + "-pos.png")),
            }
            source = stage / (name + ".html")
            source.write_text(design_html(design, index, run, base_motor, figures))
            try:
                made.append(html_to_pdf(source, stage / (name + ".pdf")))
            except (NoBrowser, OSError, ValueError):
                # One sheet failing must not lose the other fifty-nine.
                failures += 1
                made.append(source)

        on_progress(total, total, "Packing the zip")
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in made:
                archive.write(path, arcname=path.name)
        if failures:
            archive_note = ("{} of {} sheets could not be rendered to PDF and are "
                            "included as HTML.".format(failures, total))
            with zipfile.ZipFile(out_path, "a", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("README.txt", archive_note)
    return out_path
