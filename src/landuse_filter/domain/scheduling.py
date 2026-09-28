"""Ranking free Grid'5000 slots and assigning disjoint chunks to jobs."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta


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


def assign_chunks(
    pending: Sequence[tuple[str, int]], taken: set[str], capacity: float, overflow: float = 1.2
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
