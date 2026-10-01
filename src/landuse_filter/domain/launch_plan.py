"""Decide which slot gets which chunks before anything is submitted (ADR-0020, pure).

The decision is made once, single-threaded, in ranking order, so it is deterministic;
executing the resulting launches (possibly in parallel across sites) cannot change it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from landuse_filter.domain.scheduling import Slot, assign_chunks

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Planned(Generic[T]):
    index: int  # position in the ranking order: results are reported in this order
    slot: Slot
    payload: T  # whatever the caller needs to launch (the cluster)
    chunks: tuple[str, ...]


def plan_launches(  # noqa: PLR0913 - one value per fact the decision needs
    items: Sequence[tuple[Slot, T]],
    pending: Sequence[tuple[str, int]],
    taken: set[str],
    *,
    capacity: Callable[[Slot], float],
    overflow: float,
    max_total: int,
    max_per_site: int,
    total: int,
    per_site: dict[str, int],
) -> list[Planned[T]]:
    """One launch per free node of each ranked slot, within the job caps.

    Chunks are assigned in ranking order and never twice (``taken`` is not modified).
    Planning stops for good when no chunk is left, as the sequential loop does.
    """
    held = set(taken)
    counts = dict(per_site)
    plan: list[Planned[T]] = []
    for slot, payload in items:
        for _ in range(slot.free_nodes):
            if total + len(plan) >= max_total or counts.get(slot.site, 0) >= max_per_site:
                break
            chunks = assign_chunks(pending, held, capacity(slot), overflow)
            if not chunks:
                return plan
            held.update(chunks)
            counts[slot.site] = counts.get(slot.site, 0) + 1
            plan.append(Planned(len(plan), slot, payload, tuple(chunks)))
    return plan


def by_site(plan: Sequence[Planned[T]]) -> dict[str, list[Planned[T]]]:
    """The launches of each site, in plan order (one worker runs a site's jobs in turn)."""
    out: dict[str, list[Planned[T]]] = {}
    for p in plan:
        out.setdefault(p.slot.site, []).append(p)
    return out
