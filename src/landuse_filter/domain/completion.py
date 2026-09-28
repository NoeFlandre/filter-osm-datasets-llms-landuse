"""Resumption state of a chunk from the result parts uploaded so far."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class State(StrEnum):
    PENDING = "pending"
    PARTIAL = "partial"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class Part:
    """One uploaded batch of results: part id and the text hashes it covers."""

    part_id: str
    text_sha256s: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Progress:
    state: State
    missing: tuple[str, ...]
    owner: Mapping[str, str]  # text sha -> part id holding its canonical result
    foreign: tuple[str, ...]  # hashes found in parts but not in the chunk


def progress(expected: Sequence[str], parts: Iterable[Part]) -> Progress:
    """Merge parts in any order, with duplicates, to one canonical view.

    The canonical result for a text is the one in the lexicographically smallest part
    id, so the merge is order-independent.
    """
    wanted = set(expected)
    owner: dict[str, str] = {}
    foreign: set[str] = set()
    for part in sorted(parts, key=lambda p: p.part_id):
        for sha in part.text_sha256s:
            if sha not in wanted:
                foreign.add(sha)
            else:
                owner.setdefault(sha, part.part_id)
    missing = tuple(s for s in expected if s not in owner)
    state = State.COMPLETE if not missing else State.PARTIAL if owner else State.PENDING
    return Progress(state, missing, owner, tuple(sorted(foreign)))
