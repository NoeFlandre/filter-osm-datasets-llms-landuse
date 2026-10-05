"""Observable jobs (ADR-0032): one flushed line per phase, mirrored to a small bucket marker.

A publish job spends an hour before its first commit; this prints what it is doing (OAR stdout)
and writes ``published/<dataset>.progress.json`` at most once per ``PUT_SECONDS`` so the operator
reads the live phase with one bucket download. Everything effectful is injected."""

import json
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from landuse_filter.adapters.remote import Remote

PUT_SECONDS = 60.0  # at most one marker upload per minute
TICK_SECONDS = 60.0  # per-item progress lines (parts read, files built) at most this often


def progress_path(dataset: str) -> str:
    return f"published/{dataset}.progress.json"


def _fmt(counters: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in counters.items())


def _stdout(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class Progress:
    def __init__(
        self,
        *,
        dataset: str = "",
        job_id: str = "",
        out: Callable[[str], None] = _stdout,
        put: Callable[[dict], None] = lambda _state: None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.dataset, self.job_id, self.out, self.put = dataset, job_id, out, put
        self.clock, self.wall = clock, wall
        self.start = clock()
        self.state: dict = {}
        self.last_put = self.last_tick = float("-inf")

    def elapsed(self) -> int:
        return round(self.clock() - self.start)

    def event(self, phase: str, **counters: object) -> None:
        """A phase boundary: always printed; the marker is refreshed unless one went up lately."""
        self._record(phase, counters, force_put=False)

    def tick(self, phase: str, **counters: object) -> None:
        """Per-item progress: printed (and put) at most once per ``TICK_SECONDS``."""
        if self.clock() - self.last_tick >= TICK_SECONDS:
            self._record(phase, counters, force_put=False)

    def finish(self, reason: str | None = None, **counters: object) -> None:
        """The last line; always uploaded, whatever the throttle says."""
        self._record("finished", {"stop_reason": reason or "none", **counters}, force_put=True)

    def _record(self, phase: str, counters: dict, *, force_put: bool) -> None:
        now = self.clock()
        self.last_tick = now
        self.out(f"luf: [{self.elapsed()}s] {phase} {_fmt(counters)}".rstrip())
        self.state = {
            "dataset": self.dataset,
            "job_id": self.job_id,
            "phase": phase,
            "counters": counters,
            "elapsed_seconds": self.elapsed(),
            "updated_epoch": round(self.wall()),
        }
        if force_put or now - self.last_put >= PUT_SECONDS:
            self.last_put = now
            try:
                self.put(self.state)
            except Exception as error:  # noqa: BLE001 - observability must never fail the job
                self.out(f"luf: progress marker not saved ({error!r})")


NULL = Progress(out=lambda _line: None)


def bucket_progress(remote: Remote, dataset: str, job_id: str = "") -> Progress:
    """A ``Progress`` whose marker goes to the bucket through its (retrying) put."""

    def put(state: dict) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "progress.json"
            local.write_text(json.dumps(state, sort_keys=True))
            remote.put([(local, progress_path(dataset))])

    return Progress(dataset=dataset, job_id=job_id, put=put)


def read_progress(remote: Remote, dataset: str) -> dict | None:
    """The live marker, ``None`` when there is none (or the bucket is unreachable): ONE bucket
    get, no listing."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "progress.json"
        try:
            remote.get([(progress_path(dataset), local)])
        except Exception:  # noqa: BLE001 - a missing marker, or an unreachable bucket
            return None
        return json.loads(local.read_text())
