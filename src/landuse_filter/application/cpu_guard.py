"""Effects around the CPU-job boundary rules (ADR-0033): start check, watchdog, live filter."""

from collections.abc import Callable
from datetime import datetime, timedelta

from landuse_filter.adapters.g5k import Job, RemoteError
from landuse_filter.domain.cpu_job_guard import cpu_job_verdict, is_zombie, stale_day_job

POLL_INTERVAL = 5.0  # seconds between OAR status reads after oarsub
POLL_WINDOW = 120.0  # OAR may take this long to assign a start time
STARTED = ("Running", "Launching", "toLaunch", "Finishing", "Terminated")


class CpuJobCancelledError(RemoteError):
    """The job would start late or cross a boundary; it was deleted (the loop retries later)."""


def guard_start(
    *,
    job_id: str,
    submitted: datetime,
    walltime: timedelta,
    now: Callable[[], datetime],
    status: Callable[[str], tuple[str, int | None]],
    cancel: Callable[[str], None],
    sleep: Callable[[float], None],
) -> None:
    """After ``oarsub``: ask OAR when the job starts; delete it and raise if the verdict says no."""
    waited = 0.0
    while True:
        sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
        state, epoch = status(job_id)
        if state in STARTED or epoch or waited >= POLL_WINDOW:
            break
    if state in STARTED:
        start: datetime | None = now()
    else:
        start = datetime.fromtimestamp(epoch, tz=submitted.tzinfo) if epoch else None
    verdict = cpu_job_verdict(submitted=submitted, walltime=walltime, expected_start=start)
    if not verdict.allowed:
        cancel(job_id)
        raise CpuJobCancelledError(f"job {job_id} {verdict.reason}; cancelled")


def sweep(
    jobs: list[Job],
    name: str,
    *,
    now: datetime,
    cancel: Callable[[str], None],
    log: Callable[[str], None],
) -> None:
    """Watchdog: delete our stale Waiting jobs named ``name`` (they could start after 19:00)."""
    for job in jobs:
        if job.name != name or job.state != "Waiting" or job.submitted is None:
            continue
        submitted = datetime.fromtimestamp(job.submitted, tz=now.tzinfo)
        if stale_day_job(submitted=submitted, now=now):
            cancel(job.job_id)
            log(f"publish-loop: waiting day job {job.job_id} too old; cancelled")


def live(jobs: list[Job], name: str, *, now: datetime) -> bool:
    """A job of ``name`` is live unless it is a zombie (Waiting for hours: dead OAR record)."""
    return any(
        j.name == name
        and not is_zombie(
            state=j.state,
            submitted=None
            if j.submitted is None
            else datetime.fromtimestamp(j.submitted, tz=now.tzinfo),
            now=now,
        )
        for j in jobs
    )
