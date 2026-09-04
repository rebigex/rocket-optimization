"""HTTP surface for the motor optimizer.

Deliberately small: load a motor, hand back sensible defaults, start a run, poll
it, fetch results, export a design. All the thinking lives in
``rocketopt.runner``; this only translates it to and from JSON.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rocketopt.ric import load_ric, save_ric
from rocketopt.runner import (apply_hardware, build_space, default_spec,
                              describe_design, jsonable)
from rocketopt.simulate import PA_PER_PSI, curves, simulate_motor
from rocketopt.sizing import reduction_chain, size_space
from rocketopt.tolerance import (TOLERANCE_FIELDS, ToleranceSpec,
                                 default_tolerances, propagate, summarise)
from rocketopt.spec import (EFFORT_LEVELS, OPTIMISABLE_METRICS, ORDERING_MODES,
                            RunSpec)
from rocketopt.units import KG_M2S_PER_LB_IN2S, M_PER_IN

from rocketopt.report import ReportRun, build_report

from .jobs import JobRegistry

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_MOTOR = ROOT / "Data" / "Open Motor Data" / "Current.ric"

app = FastAPI(title="Lior's Really Good™ Rocket Optimizer")
jobs = JobRegistry()

#: The motor currently loaded. One app, one motor at a time -- this is a local
#: single-user tool, and pretending otherwise would only add ceremony.
STATE: Dict = {"motor": None, "name": ""}

#: Hardware overrides the user has typed into the Hardware panel this session.
#: Empty by default, and deliberately so: a loaded .ric must simulate as the
#: file says it does, or the app is lying about the motor in front of you.
#: Overrides only ever come from an explicit edit.
HARDWARE: Dict = {}


def _apply(motor: Dict) -> Dict:
    return apply_hardware(motor, **HARDWARE) if HARDWARE else motor


def _load_default() -> None:
    if DEFAULT_MOTOR.exists():
        STATE["motor"] = _apply(load_ric(DEFAULT_MOTOR))
        STATE["name"] = DEFAULT_MOTOR.name


_load_default()


def _require_motor() -> Dict:
    if STATE["motor"] is None:
        raise HTTPException(400, "Load a motor file first.")
    return STATE["motor"]


def motor_summary(motor: Dict) -> Dict:
    """What the header shows about the loaded motor."""
    metrics = simulate_motor(motor, timestep=0.002)
    grains = motor["grains"]
    return {
        "name": STATE["name"],
        "grain_count": len(grains),
        "grain_diameter": grains[0]["properties"]["diameter"],
        "grain_lengths": [g["properties"]["length"] for g in grains],
        "cores": [g["properties"]["coreDiameter"] for g in grains],
        "throat": motor["nozzle"]["throat"],
        "exit": motor["nozzle"]["exit"],
        "propellant": motor["propellant"]["name"],
        "throat_length": motor["nozzle"].get("throatLength", 0.0),
        "inhibited_ends": grains[0]["properties"].get("inhibitedEnds", "Neither"),
        "designation": metrics.designation,
        "ok": metrics.ok,
        "initial_thrust": metrics.initial_thrust,
        "total_impulse": metrics.total_impulse,
        "isp": metrics.isp,
        "burn_time": metrics.burn_time,
        "max_pressure_psi": metrics.max_pressure / PA_PER_PSI,
        "peak_kn": metrics.peak_kn,
        "initial_kn": metrics.initial_kn,
        "mass_flux_lb": metrics.peak_mass_flux / KG_M2S_PER_LB_IN2S,
        "port_throat": metrics.port_throat,
        "prop_mass": metrics.prop_mass,
        "warnings": metrics.warnings,
    }


# --------------------------------------------------------------------- routes


@app.get("/api/defaults")
def get_defaults() -> JSONResponse:
    motor = _require_motor()
    return JSONResponse(jsonable({
        "motor": motor_summary(motor),
        "spec": default_spec(motor).to_dict(),
        "metrics": OPTIMISABLE_METRICS,
        "ordering_modes": ORDERING_MODES,
        "effort_levels": EFFORT_LEVELS,
        "tolerance_fields": TOLERANCE_FIELDS,
        "tolerances": [t.to_dict() for t in default_tolerances()],
        "hardware": {
            "grain_diameter": motor["grains"][0]["properties"]["diameter"],
            "grain_length": motor["grains"][0]["properties"]["length"],
            "grain_count": len(motor["grains"]),
            "inhibited_ends": motor["grains"][0]["properties"].get(
                "inhibitedEnds", "Neither"),
            "overridden": bool(HARDWARE),
        },
        "curves": curves(motor, timestep=0.002),
    }))


class Hardware(BaseModel):
    """Only the ends are adjustable. Grain outer diameter, length and count are
    read from the .ric and never overridden -- they are the tube and the mould,
    and an app that quietly changes them is describing a motor you do not own."""

    inhibited_ends: Optional[str] = None


@app.post("/api/hardware")
def set_hardware(payload: Hardware) -> JSONResponse:
    """Restates the case the motor is built in. Never optimised, only set."""
    motor = _require_motor()
    for key, value in payload.dict(exclude_none=True).items():
        HARDWARE[key] = value
    STATE["motor"] = apply_hardware(motor, **HARDWARE)
    return get_defaults()


@app.post("/api/hardware/reset")
def reset_hardware() -> JSONResponse:
    """Drops every override and reloads the motor exactly as its file has it."""
    HARDWARE.clear()
    if DEFAULT_MOTOR.exists():
        STATE["motor"] = load_ric(DEFAULT_MOTOR)
        STATE["name"] = DEFAULT_MOTOR.name
    return get_defaults()


class MotorPath(BaseModel):
    path: str


@app.post("/api/motor/path")
def load_motor_path(payload: MotorPath) -> JSONResponse:
    path = Path(payload.path).expanduser()
    if not path.exists():
        raise HTTPException(404, "No file at {}".format(path))
    try:
        STATE["motor"] = _apply(load_ric(path))
        STATE["name"] = path.name
    except Exception as exc:
        raise HTTPException(400, "Could not read that .ric file: {}".format(exc))
    return get_defaults()


class MotorUpload(BaseModel):
    name: str
    content: str


@app.post("/api/motor/upload")
def upload_motor(payload: MotorUpload) -> JSONResponse:
    """Takes the file's text straight from the browser, no temp file dance."""
    with tempfile.NamedTemporaryFile("w", suffix=".ric", delete=False) as handle:
        handle.write(payload.content)
        temp_path = Path(handle.name)
    try:
        STATE["motor"] = _apply(load_ric(temp_path))
        STATE["name"] = payload.name
    except Exception as exc:
        raise HTTPException(400, "Could not read that .ric file: {}".format(exc))
    finally:
        temp_path.unlink(missing_ok=True)
    return get_defaults()


class SpecPayload(BaseModel):
    spec: Dict


@app.post("/api/validate")
def validate(payload: SpecPayload) -> JSONResponse:
    """Problems with a configuration, before anyone waits on a run."""
    spec = RunSpec.from_dict(payload.spec)
    motor = _require_motor()
    problems = spec.validate()
    notes = []
    try:
        space = build_space(spec, motor)
        baseline = space.from_motor(motor)
        motor_now = space.to_motor(baseline)
        for i, grain in enumerate(motor["grains"]):
            actual = grain["properties"]["coreDiameter"]
            landed = motor_now["grains"][i]["properties"]["coreDiameter"]
            if abs(actual - landed) > 1e-6:
                notes.append(
                    "Grain {} core {:.4f} in sits outside your bounds or off the "
                    "machining grid; the optimiser would start it at {:.4f} in."
                    .format(i + 1, actual / 0.0254, landed / 0.0254))
                break
    except Exception as exc:
        problems.append(str(exc))
    estimate = _estimate(spec)
    sizing = size_space(spec, len(motor["grains"]),
                        evaluated=estimate["simulations"])
    try:
        sizing["reduction"] = _plain_counts(reduction_chain(spec, motor))
    except Exception:
        sizing["reduction"] = None      # never let an extra insight break validate
    # A count of zero means the rules contradict the bounds -- worth saying so
    # here rather than letting the run fail a minute later.
    if sizing.get("total") == 0:
        problems.append(
            "These rules leave no possible motor. The minimum increase between "
            "cores needs {} in of range across {} grains, but the core bounds "
            "only span {:.2f} in.".format(
                round((spec.ordering.min_step or 0) / 0.0254, 3) if spec.ordering.min_step else 0,
                len(motor["grains"]),
                max((v.high - v.low) / 0.0254 for v in spec.variables
                    if v.name.startswith("core"))))
    # JavaScript loses integer precision past 2^53, so the exact count travels
    # as a string and the pre-formatted text is what actually gets displayed.
    if sizing.get("total") is not None:
        sizing["total_exact"] = str(sizing["total"])
    sizing["total"] = None if sizing.get("total") is None else float(sizing["total"])
    for key in ("cores", "nozzle"):
        block = sizing.get(key) or {}
        if block.get("count") is not None:
            block["count_exact"] = str(block["count"])
            block["count"] = float(block["count"])
    return JSONResponse({"problems": problems, "notes": notes,
                         "estimate": estimate, "sizing": sizing})


def _plain_counts(chain: Dict) -> Dict:
    """Big integers become strings; JavaScript cannot hold them exactly."""
    for key in ("total", "after_bounds", "legal"):
        if chain.get(key) is not None:
            chain[key + "_exact"] = str(chain[key])
            chain[key] = float(chain[key])
    return chain


class ApplyBounds(BaseModel):
    spec: Dict


@app.post("/api/tighten")
def apply_tighter_bounds(payload: ApplyBounds) -> JSONResponse:
    """Narrows the spec to the bounds the limits provably rule out.

    Returns the edited spec rather than storing it, so the browser stays the
    only place a configuration lives.
    """
    motor = _require_motor()
    spec = RunSpec.from_dict(payload.spec)
    chain = reduction_chain(spec, motor, samples=1)
    tight = chain["tightened"]
    for var in spec.variables:
        if var.name == "throat" and tight.get("throat_low"):
            var.low = max(var.low, tight["throat_low"])
        if var.name.startswith("core") and tight.get("core_high"):
            var.high = min(var.high, tight["core_high"])
    return JSONResponse({"spec": spec.to_dict(),
                         "changes": tight.get("changes", [])})


def _estimate(spec: RunSpec) -> Dict:
    """Rough wall-clock, so nobody starts a five-minute run by accident."""
    budget = spec.budget
    sims = budget["total"]
    if spec.mode == "pareto":
        sims += budget["samples"]
    # Verification re-runs the survivors at the fine timestep, and the
    # sensitivity sweep costs two more per free dimension.
    free = max(1, sum(1 for v in spec.variables if v.free))
    sims += 60 + 2 * free
    # Measured: ~32 designs/second on a 12-worker pool at a 0.01 s timestep.
    # Cost scales with the number of steps in a burn, so a finer timestep is
    # slower, not faster.
    rate = 32.0 * (max(spec.search_timestep, 0.002) / 0.01) ** 0.75
    seconds = sims / max(rate, 1.0)
    if spec.mode == "pareto":
        seconds += 90  # surrogate training and permutation importance
    return {"simulations": int(sims), "seconds": int(seconds),
            "seeds": budget["seeds"], "pop": budget["pop"], "gen": budget["gen"],
            "per_seed": budget["per_seed"]}


@app.post("/api/run")
def start_run(payload: SpecPayload) -> JSONResponse:
    motor = _require_motor()
    spec = RunSpec.from_dict(payload.spec)
    problems = spec.validate()
    if problems:
        raise HTTPException(400, "; ".join(problems))
    job = jobs.start(spec, motor)
    return JSONResponse(job.status_dict())


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    """Runs this session still remembers, for the report picker."""
    return JSONResponse(jobs.listing())


class ReportRequest(BaseModel):
    job_ids: list
    title: Optional[str] = None


@app.post("/api/report")
def make_report(payload: ReportRequest) -> Response:
    """Builds the technical report for one or more finished runs."""
    motor = _require_motor()
    chosen = []
    for job_id in payload.job_ids:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Run {} is no longer held.".format(job_id))
        if job.status != "done" or job.result is None:
            raise HTTPException(409, "Run {} is {}.".format(job_id, job.status))
        chosen.append(job)
    # Sections read in the order the runs were made, not the order the picker
    # happened to list them.
    chosen.sort(key=lambda j: j.started_at)
    selected = [ReportRun(label=j.label, result=j.result.to_dict(), spec=j.spec)
                for j in chosen]
    if not selected:
        raise HTTPException(400, "Pick at least one finished run.")

    out = ROOT / "outputs" / "reports"
    path = build_report(selected, motor, out, title=payload.title)
    return Response(
        content=path.read_text(), media_type="text/html",
        headers={"Content-Disposition": 'attachment; filename="motor-report.html"'})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such run.")
    return JSONResponse(job.status_dict())


@app.get("/api/jobs/{job_id}/live")
def job_live(job_id: str) -> JSONResponse:
    """The search as it stands right now, for the waiting screen."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such run.")
    status = job.status_dict()
    status["telemetry"] = job.telemetry
    return JSONResponse(jsonable(status))


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> JSONResponse:
    if not jobs.cancel(job_id):
        raise HTTPException(400, "That run has already finished.")
    return JSONResponse({"ok": True})


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such run.")
    if job.status != "done" or job.result is None:
        raise HTTPException(409, "That run is {}.".format(job.status))
    return JSONResponse(job.result.to_dict())


class ExportPayload(BaseModel):
    spec: Dict
    x: list
    name: Optional[str] = None


@app.post("/api/export")
def export_design(payload: ExportPayload) -> Response:
    """Hands back a .ric the user can open straight in openMotor."""
    motor = _require_motor()
    spec = RunSpec.from_dict(payload.spec)
    space = build_space(spec, motor)
    import numpy as np

    built = space.to_motor(np.asarray(payload.x, dtype=float))
    with tempfile.NamedTemporaryFile("w", suffix=".ric", delete=False) as handle:
        temp_path = Path(handle.name)
    save_ric(temp_path, built)
    text = temp_path.read_text()
    temp_path.unlink(missing_ok=True)
    filename = (payload.name or "optimized") + ".ric"
    return Response(content=text, media_type="application/x-yaml",
                    headers={"Content-Disposition":
                             'attachment; filename="{}"'.format(filename)})


class CurvePayload(BaseModel):
    spec: Dict
    x: list


class RobustnessRequest(BaseModel):
    spec: Dict
    x: list
    tolerances: list
    samples: int = 400


@app.post("/api/robustness")
def robustness(payload: RobustnessRequest) -> JSONResponse:
    """How often this design stays legal once it is actually built."""
    motor = _require_motor()
    spec = RunSpec.from_dict(payload.spec)
    space = build_space(spec, motor)
    import numpy as np

    built = space.to_motor(np.asarray(payload.x, dtype=float))
    tolerances = [ToleranceSpec.from_dict(t) for t in payload.tolerances]
    report = propagate(built, tolerances, spec.enabled_constraints,
                       samples=max(50, min(int(payload.samples), 2000)),
                       timestep=spec.search_timestep)
    report["summary"] = summarise(report)
    return JSONResponse(jsonable(report))


@app.post("/api/curves")
def design_curves(payload: CurvePayload) -> JSONResponse:
    """Time series for a design the user clicked on in the trade-off plot."""
    motor = _require_motor()
    spec = RunSpec.from_dict(payload.spec)
    space = build_space(spec, motor)
    import numpy as np

    x = np.asarray(payload.x, dtype=float)
    design = describe_design(space, x, spec, "Selected", with_curves=True)
    return JSONResponse(jsonable(design))


app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
