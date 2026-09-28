"""Canonical generation per text and its parsed decision, from verified parts."""

from collections.abc import Iterator

from landuse_filter.adapters.store import CorruptPartError, WorkStore
from landuse_filter.domain.parsing import Verdict, parse_generation


def canonical_generations(store: WorkStore, fp: str) -> Iterator[dict]:
    """One row per text sha: the result in the smallest part id (order-independent)."""
    seen: set[str] = set()
    for path in sorted(store.part_paths(fp), key=lambda p: (p.parent.name, p.stem)):
        try:
            rows = store.read_part(path).to_pylist()
        except CorruptPartError:
            continue
        for row in rows:
            if row["text_sha256"] not in seen:
                seen.add(row["text_sha256"])
                yield row


def verdict_of(row: dict) -> Verdict:
    return parse_generation(row["raw_output"], truncated=bool(row["truncated"]))


def decisions_by_sha(store: WorkStore, fp: str) -> dict[str, str | None]:
    """text sha -> yes/no, or None for failed (the benchmark's unparsed bucket)."""
    out: dict[str, str | None] = {}
    for row in canonical_generations(store, fp):
        decision = verdict_of(row).decision.value
        out[row["text_sha256"]] = decision if decision in ("yes", "no") else None
    return out
