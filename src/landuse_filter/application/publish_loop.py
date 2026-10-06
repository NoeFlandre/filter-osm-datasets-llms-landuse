"""The publish loop (ADR-0023): thin, everything effectful is injected."""

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from landuse_filter.adapters.g5k import RemoteError
from landuse_filter.adapters.remote import Remote
from landuse_filter.domain.publish_loop import FINISHED, SUBMIT, decide, is_done, pause

TRANSIENT = (RemoteError, TimeoutError, OSError)  # frontend/ssh and bucket failures


def status_path(dataset: str) -> str:
    return f"published/{dataset}.status.json"


def read_status(remote: Remote, dataset: str) -> dict | None:
    """The marker the last publish job left in the bucket, ``None`` when there is none (or it
    is not valid JSON). A bucket failure raises."""
    path = status_path(dataset)
    if path not in remote.ls("published/"):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "status.json"
        remote.get([(path, local)])
        try:
            return json.loads(local.read_text())
        except ValueError:
            return None


@dataclass
class LoopIO:
    """What the loop touches: the bucket marker, the live jobs, the submission."""

    status: Callable[[], dict | None]
    live: Callable[[], bool]  # a publish job of the dataset is waiting or running on the site
    submit: Callable[[], str]
    sleep: Callable[[float], None]
    log: Callable[[str], None]
    sweep: Callable[[], None] = lambda: None  # watchdog: delete stale waiting day jobs


def run_loop(
    io: LoopIO,
    revision: str,
    *,
    interval: float,
    cap: float,
    max_cycles: int | None = None,
) -> str:
    """Submit publish jobs until the marker says done; returns ``finished`` (or ``gave up``
    after ``max_cycles``, for tests). A failing step (a slow ``usagepolicycheck``, a frontend
    timeout, a bucket error) is logged and retried after a longer pause, never fatal."""
    failures = 0
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        try:
            io.sweep()
            done = is_done(io.status(), revision)
            action = decide(done=done, live=False if done else io.live())
            if action == SUBMIT:
                io.log(f"no publish job live: submitted {io.submit()}")
            failures = 0
        except TRANSIENT as exc:
            failures += 1
            io.log(f"publish-loop: {type(exc).__name__}: {exc} (retry {failures})")
            action = "retry"
        if action == FINISHED:
            io.log("publish-loop: the status marker says nothing is left to publish")
            return FINISHED
        io.sleep(pause(failures, interval, cap))
    return "gave up"
