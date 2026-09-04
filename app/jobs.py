"""Background optimisation runs and their progress.

An optimisation takes tens of seconds to several minutes, which is far too long
to hold an HTTP request open. Each run therefore becomes a job with an id: the
browser starts one, polls it about once a second, and collects the result when
it finishes. The heavy work already happens in child processes via
``SimulationPool``, so a plain thread here is enough to keep the server
responsive.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    _cancel: threading.Event = field(default_factory=threading.Event)

    def status_dict(self) -> Dict:
        elapsed = (self.finished_at or time.time()) - self.started_at
        return {
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

    def start(self, spec: RunSpec, base_motor: Dict, workers: Optional[int] = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], spec=spec,
                  label=describe_spec(spec))
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self.keep:
                self._jobs.pop(self._order.pop(0), None)

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
                                 workers=workers)
                job.status, job.stage, job.fraction = "done", "done", 1.0
                job.message = "Finished"
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
