"""Useful fraction of finished GPU jobs from their timeline records (pure).

A job record (``jobs/<site>/<job>.json``) written by a node that knows its timeline has
``engine_ready_at``, ``last_result_at``, ``started_at``, ``ended_at`` (ISO UTC strings),
``walltime_s`` and optionally ``env_ready_seconds``. Older records lack them and are skipped.

Generation seconds = ``last_result_at - engine_ready_at``. A job *stopped early* finished all
its assigned work (it was not signalled) with more than ``EARLY_MARGIN`` of its walltime left;
otherwise it ran to the checkpoint signal (or close enough to the end).
"""

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

EARLY_MARGIN = 0.10

Job = Mapping[str, Any]


def _gap(job: Job, start: str, end: str) -> float | None:
    if not job.get(start) or not job.get(end):
        return None
    return (datetime.fromisoformat(job[end]) - datetime.fromisoformat(job[start])).total_seconds()


def generation_seconds(job: Job) -> float | None:
    gap = _gap(job, "engine_ready_at", "last_result_at")
    return None if gap is None else max(0.0, gap)


def is_measured(job: Job) -> bool:
    return job.get("walltime_s", 0) > 0 and generation_seconds(job) is not None


def seconds_left(job: Job) -> float | None:
    """Walltime remaining when the node exited (setup before ``started_at`` included)."""
    gap = _gap(job, "started_at", "ended_at")
    if gap is None:
        return None
    return job["walltime_s"] - gap - (job.get("env_ready_seconds") or 0)


def stopped_early(job: Job) -> bool:
    left = seconds_left(job)
    return not job.get("stopped") and left is not None and left > EARLY_MARGIN * job["walltime_s"]


def _generated(job: Job) -> float:
    return generation_seconds(job) or 0.0


def _group(jobs: list[Job]) -> dict[str, Any]:
    walltime = sum(j["walltime_s"] for j in jobs)
    early = sum(map(stopped_early, jobs))
    return {
        "jobs": len(jobs),
        "useful_fraction": round(sum(map(_generated, jobs)) / walltime, 3),
        "stopped_early": round(early / len(jobs), 3),
        "ran_to_checkpoint": round(1 - early / len(jobs), 3),
    }


def _gpu(job: Job) -> str:
    return job.get("gpu_key") or job.get("gpu") or "unknown"


def _overall(measured: list[Job]) -> dict[str, Any]:
    if measured:
        return _group(measured)
    return {"jobs": 0, "useful_fraction": None, "stopped_early": None, "ran_to_checkpoint": None}


def useful_fraction(jobs: Iterable[Job]) -> dict[str, Any]:
    """Overall and per GPU type: useful fraction and the stopped-early / checkpoint shares."""
    measured = [j for j in jobs if is_measured(j)]
    by_gpu: dict[str, list[Job]] = {}
    for j in measured:
        by_gpu.setdefault(_gpu(j), []).append(j)
    return {
        "overall": _overall(measured),
        "by_gpu": {g: _group(v) for g, v in sorted(by_gpu.items())},
    }
