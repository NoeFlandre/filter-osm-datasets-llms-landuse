"""CPU jobs never cross 09:00/19:00 on a working day (ADR-0033, pure).

A queued job can start hours after ``oarsub``. The submission-time window is therefore not
enough: the *expected start* decides. Rules (Europe/Paris, see ``policy``):

- A job submitted in working-day daytime must start within ``START_TOLERANCE`` (quota-exempt
  jobs start < 10 min after submission); an unknown start counts as late.
- Whatever the time, [start, start + walltime] must not strictly contain a working-day 09:00,
  nor a working-day 19:00 unless the job was submitted the same day at/after 17:00.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from landuse_filter.domain.policy import DAY_END, DAY_START, is_daytime, is_working_day

EVENING_SUBMIT = time(17, 0)  # jobs submitted from here may cross 19:00
START_TOLERANCE = timedelta(minutes=10)
WATCHDOG_CUTOFF = time(16, 50)  # a still-waiting day job is deleted from here on
ZOMBIE_AGE = timedelta(hours=2)  # a job Waiting this long is a dead OAR record, not live


@dataclass(frozen=True, slots=True)
class CpuVerdict:
    allowed: bool
    reason: str


def _boundaries(start: datetime, end: datetime) -> list[tuple[datetime, time]]:
    days = [start.date() + timedelta(days=n) for n in range((end.date() - start.date()).days + 1)]
    return [
        (datetime.combine(day, edge, tzinfo=start.tzinfo), edge)
        for day in days
        if is_working_day(day)
        for edge in (DAY_START, DAY_END)
    ]


def crosses_forbidden(submitted: datetime, start: datetime, end: datetime) -> bool:
    """True when (start, end) strictly contains a forbidden working-day boundary."""
    evening = submitted.date() if submitted.time() >= EVENING_SUBMIT else None
    return any(
        start < moment < end and not (edge == DAY_END and moment.date() == evening)
        for moment, edge in _boundaries(start, end)
    )


def cpu_job_verdict(
    *, submitted: datetime, walltime: timedelta, expected_start: datetime | None
) -> CpuVerdict:
    """May a CPU job submitted at ``submitted`` and expected to start then stay queued?

    A night job without a predicted start is left to OAR's ``night`` type, which confines it.
    """
    if expected_start is None:
        if is_daytime(submitted):
            return CpuVerdict(
                allowed=False, reason="would start late: no predicted start in daytime"
            )
        return CpuVerdict(allowed=True, reason="night job, start left to the night type")
    if is_daytime(submitted) and expected_start - submitted > START_TOLERANCE:
        return CpuVerdict(allowed=False, reason="would start late")
    if crosses_forbidden(submitted, expected_start, expected_start + walltime):
        return CpuVerdict(allowed=False, reason="would cross a day/night boundary")
    return CpuVerdict(allowed=True, reason="ok")


def stale_day_job(*, submitted: datetime, now: datetime) -> bool:
    """Watchdog: must a still-Waiting job submitted at ``submitted`` be deleted at ``now``?

    Yes for a job submitted in working-day daytime before 17:00 once it waited more than
    ``START_TOLERANCE``, and from 16:50 on regardless. Night and evening jobs are never stale.
    """
    if not is_daytime(submitted) or submitted.time() >= EVENING_SUBMIT:
        return False
    if now - submitted > START_TOLERANCE:
        return True
    return now.date() == submitted.date() and now.time() >= WATCHDOG_CUTOFF


def is_zombie(*, state: str, submitted: datetime | None, now: datetime) -> bool:
    """A job listed Waiting for over ``ZOMBIE_AGE`` is a dead record ('already killed')."""
    return state == "Waiting" and submitted is not None and now - submitted > ZOMBIE_AGE
