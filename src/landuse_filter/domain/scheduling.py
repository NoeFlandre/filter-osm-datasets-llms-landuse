"""Ranking free Grid'5000 slots and assigning disjoint chunks to jobs."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta

STALE_BESTEFFORT_WAIT = timedelta(minutes=20)


def is_stale_besteffort(
    *,
    queue: str,
    state: str,
    submitted_at: float | None,
    now: float,
    max_wait: timedelta = STALE_BESTEFFORT_WAIT,
) -> bool:
    """A besteffort job still waiting at least ``max_wait`` after submission.

    Its GPUs were taken meanwhile, so it would hold its chunks and a per-site slot for
    hours. Other queues (night, exotic) legitimately wait in the schedule: never stale.
    """
    if queue != "besteffort" or state != "Waiting" or submitted_at is None:
        return False
    return now - submitted_at >= max_wait.total_seconds()


@dataclass(frozen=True, slots=True)
class Slot:
    """A GPU reservation we could obtain: where, how fast, how soon, how long."""

    site: str
    cluster: str
    gpu: str
    gpus_per_node: int
    free_nodes: int
    wait: timedelta
    walltime: timedelta
    job_type: str | None
    sentences_per_second: float  # per GPU, from the calibrated profile
    besteffort: bool = False  # submit as a preemptible besteffort job
    queued: bool = False  # no GPU free now: a bounded queue entry that may wait a while
    fallbacks: tuple[timedelta, ...] = ()  # shorter walltimes to retry in this same slot


def slot_for(  # noqa: PLR0913 - one value per fact the decision needs
    *,
    site: str,
    cluster: str,
    gpu: str,
    free: int,
    walltime: timedelta | None,
    job_type: str | None,
    sentences_per_second: float,
    besteffort: bool,
    queue_room: bool,
    queued_wait: timedelta,
    fallbacks: tuple[timedelta, ...] = (),
) -> Slot | None:
    """The slot a cluster offers, or ``None``.

    A free GPU gives an immediate slot. With none free we may queue one job, but only on
    clusters that are not besteffort-only (they cannot queue) and only while the site has
    room in its queue. No allowed walltime (night-only cluster by day) means no slot.
    """
    if walltime is None:
        return None
    queued = free <= 0
    if queued and not _may_queue(besteffort=besteffort, queue_room=queue_room):
        return None
    nodes, wait = (1, queued_wait) if queued else (free, timedelta(0))
    return Slot(
        site,
        cluster,
        gpu,
        1,
        nodes,
        wait,
        walltime,
        job_type,
        sentences_per_second,
        besteffort=besteffort,
        queued=queued,
        fallbacks=fallbacks,
    )


def _may_queue(*, besteffort: bool, queue_room: bool) -> bool:
    """Besteffort-only clusters cannot queue; others may while the site has queue room."""
    return queue_room and not besteffort


def useful_sentences(slot: Slot, setup: timedelta) -> float:
    """Sentences a one-node job in ``slot`` can finish, discounted by the wait."""
    working = max(0.0, (slot.walltime - setup).total_seconds())
    discount = 1.0 / (1.0 + slot.wait.total_seconds() / 3600.0)
    return slot.sentences_per_second * slot.gpus_per_node * working * discount


def rank_slots(slots: Iterable[Slot], setup: timedelta) -> list[Slot]:
    return sorted(
        (s for s in slots if useful_sentences(s, setup) > 0),
        key=lambda s: (-useful_sentences(s, setup), s.site, s.cluster),
    )


DEFAULT_CHUNK_OVERFLOW = 1.2  # work assigned per job = capacity x this (ADR-0018)


def assign_chunks(
    pending: Sequence[tuple[str, int]],
    taken: set[str],
    capacity: float,
    overflow: float = DEFAULT_CHUNK_OVERFLOW,
) -> list[str]:
    """Pick chunks (id, size) in plan order, skipping ones held by live jobs.

    ``overflow`` gives a job a little more work than its expected capacity, so a fast
    job does not idle; unfinished chunks return to the pool when the job ends.
    """
    budget = capacity * overflow
    picked: list[str] = []
    used = 0
    for chunk, size in pending:
        if chunk in taken:
            continue
        if picked and used + size > budget:
            break
        picked.append(chunk)
        used += size
    return picked


def walltime_ladder(
    *,
    window_max: timedelta,
    night: bool,
    day: timedelta,
    preferred: timedelta,
    fallback: timedelta,
    day_short: timedelta | None = None,
) -> tuple[timedelta, ...]:
    """Walltimes to try, in order, for one slot: the preferred one, then a shorter fallback.

    By day (``night`` false) the rungs are ``day`` then ``day_short`` (omitted: ``day`` alone).
    At night they are ``preferred`` then ``fallback``. Each is capped by the window maximum,
    the fallback never longer than the preferred one, duplicates and non-positive values
    dropped.
    """
    first, second = (preferred, fallback) if night else (day, day_short or day)
    rungs = [min(first, window_max), min(second, first, window_max)]
    return _distinct_positive(rungs)


def _distinct_positive(rungs: list[timedelta]) -> tuple[timedelta, ...]:
    return tuple(w for i, w in enumerate(rungs) if w > timedelta(0) and w not in rungs[:i])


LONG_FAILURES = 3
LONG_PAUSE = timedelta(hours=1)


def long_attempt_tripped(failures: int, limit: int = LONG_FAILURES) -> bool:
    """True once ``failures`` consecutive long attempts reach ``limit``."""
    return failures >= limit


def without_long(ladder: tuple[timedelta, ...]) -> tuple[timedelta, ...]:
    """The ladder with its long rung dropped: only the shortest walltime remains."""
    return ladder[-1:]
