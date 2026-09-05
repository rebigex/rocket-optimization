"""Background optimisation runs and their progress.

An optimisation takes tens of seconds to several minutes, which is far too long
to hold an HTTP request open. Each run therefore becomes a job with an id: the
browser starts one, polls it about once a second, and collects the result when
it finishes. The heavy work already happens in child processes via
``SimulationPool``, so a plain thread here is enough to keep the server
responsive.


by the way if you are trying to understand this code, good luck.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from rocketopt.bundle import build_bundle
from rocketopt.outputs import write_run
from rocketopt.report import ReportRun, build_report
from rocketopt.runner import RunResult, run
from rocketopt.spec import RunSpec


def describe_spec(spec: RunSpec) -> str:
    """A short name for what a run was allowed to move.

    Reports name their own sections, so the label has to come from the
    configuration rather than from whoever started the run.
    """
    free = [v.name for v in spec.variables if v.free]
    cores = sum(1 for n in free if n.startswith("core"))
    nozzle = [n for n in free if not n.startswith("core")]
    if cores and not nozzle:
        return "Cores only, nozzle fixed"
    if cores and nozzle:
        return "Cores plus {} nozzle dimension{}".format(
            len(nozzle), "" if len(nozzle) == 1 else "s")
    if nozzle:
        return "Nozzle only"
    return "No free dimensions"


@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | running | done | failed | cancelled
    stage: str = "queued"
    fraction: float = 0.0
    message: str = "Waiting to start"
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Optional[RunResult] = None
    spec: Optional[RunSpec] = None
    label: str = ""
    #: Latest generation snapshot, plus a short scalar history. Replaced rather
    #: than appended so the poll payload stays a fixed size however long the
    #: search runs.
    telemetry: Optional[Dict] = None
    trace: List[Dict] = field(default_factory=list)
    #: What the estimate promised, and which kind of run this is, so the
    #: registry can learn how far off the estimate runs for this shape.
    predicted: float = 0.0
    shape: str = ""
    #: The report for this run, written as soon as it finishes. Every run gets
    #: one -- a run you have to remember to write up is a run you will not.
    report: Optional[Path] = None
    report_error: str = ""
    #: A zip holding one sheet per design on the trade-off curve. Built only on
    #: request -- it is a browser launch per design, so it is not something to
    #: do behind every run.
    bundle: Optional[Path] = None
    bundle_status: str = "idle"      # idle | building | ready | failed
    bundle_done: int = 0
    bundle_total: int = 0
    bundle_message: str = ""
    bundle_error: str = ""
    _cancel: threading.Event = field(default_factory=threading.Event)

    def status_dict(self) -> Dict:
        elapsed = (self.finished_at or time.time()) - self.started_at
        return {
            "report": self.report.name if self.report else "",
            "report_error": self.report_error,
            "bundle_status": self.bundle_status,
            "bundle_done": self.bundle_done,
            "bundle_total": self.bundle_total,
            "bundle_message": self.bundle_message,
            "bundle_error": self.bundle_error,
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "fraction": round(self.fraction, 4),
            "message": self.message,
            "error": self.error,
            "elapsed": round(elapsed, 1),
            "label": self.label,
            "n_designs": len(self.result.designs) if self.result else 0,
        }


class JobRegistry:
    """Holds running and recently finished jobs, newest kept."""

    def __init__(self, keep: int = 12) -> None:
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self.keep = keep
        #: How wrong the estimate turned out to be, per kind of run. A run has
        #: costs that are not simulations -- ranking, curve building, fitting --
        #: and they differ by mode. Rather than bury another machine-specific
        #: constant in the estimate, learn the correction from what happened.
        self._factors: Dict[str, float] = {}

    def factor(self, shape: str) -> float:
        return self._factors.get(shape, 1.0)

    def has_seen(self, shape: str) -> bool:
        """Whether a run of this kind has finished and taught us its cost."""
        return shape in self._factors

    def record_outcome(self, shape: str, predicted: float, actual: float) -> None:
        """Folds one run's accuracy into the correction for its kind of run.

        Smoothed rather than replaced, so one unlucky run under load does not
        throw the next estimate off in the other direction.
        """
        if not predicted or predicted <= 0 or actual <= 0:
            return
        observed = actual / predicted
        previous = self._factors.get(shape)
        blended = observed if previous is None else 0.5 * previous + 0.5 * observed
        self._factors[shape] = float(min(max(blended, 0.2), 5.0))

    def start_bundle(self, job: Job, base_motor: Dict, out_dir: Path) -> bool:
        """Renders one sheet per design, in the background.

        Sixty designs is sixty browser launches, so this cannot happen inside
        the request that asks for it.
        """
        if job.result is None or job.status != "done":
            return False
        if job.bundle_status == "building":
            return True

        job.bundle_status = "building"
        job.bundle_done, job.bundle_total = 0, len(job.result.designs)
        job.bundle_message = "Starting"
        job.bundle_error = ""

        def progress(done: int, total: int, message: str) -> None:
            job.bundle_done, job.bundle_total = done, total
            job.bundle_message = message

        def target() -> None:
            try:
                out = out_dir / "design-sheets-{}.zip".format(job.id)
                job.bundle = build_bundle(
                    ReportRun(label=job.label, result=job.result.to_dict(),
                              spec=job.spec),
                    base_motor, out, on_progress=progress)
                job.bundle_status = "ready"
                job.bundle_message = "Ready"
            except Exception as exc:
                job.bundle_status = "failed"
                job.bundle_error = str(exc)
                job.bundle_message = "Could not build the sheets"
                traceback.print_exc()

        threading.Thread(target=target, name="bundle-" + job.id,
                         daemon=True).start()
        return True

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def listing(self) -> List[Dict]:
        """Newest first, so the report picker shows the latest run at the top."""
        with self._lock:
            return [self._jobs[i].status_dict()
                    for i in reversed(self._order) if i in self._jobs]

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in ("done", "failed", "cancelled"):
            return False
        job._cancel.set()
        job.status = "cancelled"
        job.message = "Cancelled"
        job.finished_at = time.time()
        return True

    def start(self, spec: RunSpec, base_motor: Dict, workers: Optional[int] = None,
              predicted: float = 0.0, shape: str = "",
              reports_dir: Optional[Path] = None,
              outputs_dir: Optional[Path] = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], spec=spec,
                  label=describe_spec(spec))
        job.predicted = float(predicted)
        job.shape = shape
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self.keep:
                self._jobs.pop(self._order.pop(0), None)

        def telemetry(snapshot: Dict) -> None:
            history = job.trace
            best = snapshot.get("best")
            if best is not None:
                history.append({"seed": snapshot["seed_index"],
                                "gen": snapshot["generation"],
                                "a": best[0], "b": best[1]})
                # A long run would otherwise accumulate thousands of points that
                # no sparkline can show; thin the oldest half when it gets big.
                if len(history) > 600:
                    del history[: len(history) // 2]
            snapshot["trace"] = history[-240:]
            job.telemetry = snapshot

        def progress(stage: str, fraction: float, message: str) -> None:
            if job._cancel.is_set():
                # The optimiser has no cancel hook of its own, so raising out of
                # the progress callback is how a run gets stopped mid-flight.
                raise RuntimeError("__cancelled__")
            job.stage, job.fraction, job.message = stage, fraction, message

        def target() -> None:
            job.status = "running"
            try:
                job.result = run(spec, base_motor, on_progress=progress,
                                 workers=workers, on_telemetry=telemetry)
                # Every run produces its report, without being asked. Failing to
                # write it must not lose the run that has already been done.
                if reports_dir is not None or outputs_dir is not None:
                    job.stage, job.fraction = "report", 0.97
                    job.message = "Writing the report"
                    try:
                        # outputs/ mirrors the last run only, so it is emptied
                        # first and the figures below land in the fresh folder.
                        figures_dir = None
                        if outputs_dir is not None:
                            from rocketopt.runner import build_space
                            write_run(job.result, base_motor,
                                      build_space(spec, base_motor), outputs_dir)
                            figures_dir = Path(outputs_dir) / "figures"
                        if reports_dir is not None:
                            files = build_report(
                                [ReportRun(label=job.label,
                                           result=job.result.to_dict(), spec=spec)],
                                base_motor, reports_dir, figures_dir=figures_dir)
                            job.report = files.path
                            job.report_error = files.pdf_error
                    except Exception as exc:
                        job.report_error = str(exc)
                        traceback.print_exc()
                job.status, job.stage, job.fraction = "done", "done", 1.0
                job.message = "Finished"
                # How long this kind of run really takes, for the next
                # estimate. Only this one loop corrects -- see _rate_at in
                # server.py for what happened when two of them did.
                self.record_outcome(job.shape, job.predicted,
                                    time.time() - job.started_at)
            except Exception as exc:  # surfaced to the user, not swallowed
                if "__cancelled__" in str(exc):
                    job.status, job.message = "cancelled", "Cancelled"
                else:
                    job.status = "failed"
                    job.error = str(exc)
                    job.message = "Run failed"
                    traceback.print_exc()
            finally:
                job.finished_at = time.time()

        threading.Thread(target=target, name="optimise-" + job.id,
                         daemon=True).start()
        return job
