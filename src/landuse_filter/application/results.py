"""Canonical generation per text and its parsed decision, from verified parts."""

from collections.abc import Iterator
from typing import TYPE_CHECKING

from landuse_filter.adapters.store import CorruptPartError, WorkStore

if TYPE_CHECKING:
    from landuse_filter.adapters.remote import Remote
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


def gathered_decisions(local: WorkStore, remote: "Remote | None", fp: str) -> dict[str, str | None]:
    """Decisions for namespace ``fp`` from local parts plus the bucket's, in a temp tree.

    Bucket parts are downloaded to a temporary directory (deleted afterwards), so the
    controller machine never keeps generations (scarce SSD).
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="luf-gate-"))
    try:
        merged = WorkStore(tmp)
        for path in local.part_paths(fp):
            target = merged.path(str(path.relative_to(local.root)))
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
        if remote is not None:
            wanted = [
                p
                for p in remote.ls(f"parts/{fp}/")
                if p.endswith(".parquet") and not merged.exists(p)
            ]
            remote.get([(p, merged.path(p)) for p in wanted])
        return decisions_by_sha(merged, fp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
