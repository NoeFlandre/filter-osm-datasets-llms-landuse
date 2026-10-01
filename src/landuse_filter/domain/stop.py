"""When a publish job must stop starting new work (ADR-0021, pure).

A job ends at its OAR walltime; it is warned by a signal 5 minutes before and may know its
deadline. Either way it stops *starting* files early enough to flush, refresh the card and
save its ledgers in the time that is left.
"""

import json
from collections.abc import Mapping

STOP_MARGIN_SECONDS = 360.0  # no new file this close to the deadline: 6 minutes to flush


def should_stop(now: float, deadline: float | None, margin: float = STOP_MARGIN_SECONDS) -> bool:
    """True once less than ``margin`` seconds remain before ``deadline`` (``None``: no limit)."""
    return deadline is not None and now >= deadline - margin


def stop_reason(
    now: float,
    deadline: float | None,
    signalled: str | None,
    margin: float = STOP_MARGIN_SECONDS,
) -> str | None:
    """Why to stop, if so: a received signal wins over the deadline."""
    if signalled:
        return f"signal {signalled}"
    if should_stop(now, deadline, margin):
        return "deadline"
    return None


def deadline_from_env(environ: Mapping[str, str]) -> float | None:
    """The job end time exported by ``scripts/node_job.sh`` (epoch seconds), when valid."""
    try:
        value = float(environ.get("LUF_JOB_DEADLINE_EPOCH", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def deadline_from_oarstat(text: str, now: float) -> int | None:
    """End time from ``oarstat -j <id> -J`` output: start (or ``now`` if not yet running) plus
    walltime. ``None`` when the output has no usable walltime."""
    try:
        data = json.loads(text)
        job = next(iter(data.values())) if "walltime" not in data else data
        walltime = float(job["walltime"])
        start = float(job.get("startTime") or 0)
    except (ValueError, KeyError, TypeError, StopIteration, AttributeError):
        return None
    if walltime <= 0:
        return None
    return int((start if start > 0 else now) + walltime)
