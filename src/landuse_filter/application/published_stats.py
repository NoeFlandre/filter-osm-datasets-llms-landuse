"""Deterministic statistics of the published tables, one record per published file.

The dataset card is computed from these records, never from whatever a single run built:
each ``labels/`` file contributes its decision and failure-reason counts, each
``generations/`` file its row count and rows per GPU type. Records are cached in
``published/<dataset>.stats.jsonl``; a published file without a record is (re)counted
from the Hub, so the card is always a function of the published data.
"""

import logging
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import pyarrow.parquet as pq

from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.geo import Located
from landuse_filter.domain import geomap
from landuse_filter.domain.gpu import gpu_key

Source = Path | IO[bytes]
Opener = Callable[[str], Source]  # repo path -> local file or Hub file object
Locator = Callable[[str, Source], Located]  # labels path, its source -> where the rows are


RECORD_FLUSH = 25  # files counted per ledger append
COUNT_WORKERS = 6  # files counted in parallel (each reads its labels file and map inputs)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileStats:
    """Counts of one published file; ``to_json``/``from_json`` are the ledger line format.

    A labels record has ``decisions`` and ``failures`` (plus ``cells``/``labelled``/``located``
    once the map was computed; ``cells is None`` marks a record made before, or without, a
    locator); a generations record has ``rows`` and ``gpus``.
    """

    path: str
    decisions: dict[str, int] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    rows: int = 0
    gpus: dict[str, int] = field(default_factory=dict)
    cells: dict[str, list[int]] | None = None
    labelled: int = 0
    located: int = 0

    @property
    def is_labels(self) -> bool:
        return self.path.startswith("labels/")

    def to_json(self) -> dict[str, Any]:
        if not self.is_labels:
            return {"path": self.path, "rows": self.rows, "gpus": self.gpus}
        record: dict[str, Any] = {
            "path": self.path,
            "decisions": self.decisions,
            "failures": self.failures,
        }
        if self.cells is not None:
            record |= {"cells": self.cells, "labelled": self.labelled, "located": self.located}
        return record

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> "FileStats":
        """Read a ledger line; every field but ``path`` is optional (older ledgers)."""
        return cls(
            path=record["path"],
            decisions=record.get("decisions", {}),
            failures=record.get("failures", {}),
            rows=record.get("rows", 0),
            gpus=record.get("gpus", {}),
            cells=record.get("cells"),
            labelled=record.get("labelled", 0),
            located=record.get("located", 0),
        )


@dataclass(frozen=True)
class PublishedStats:
    """The dataset card's numbers: the sum of every published file's :class:`FileStats`."""

    decisions: dict[str, int]
    failures: dict[str, int]
    gpus: dict[str, int]
    unique_texts: int
    cells: dict[str, tuple[int, int]]  # H3 cell -> (yes, no)
    labelled: int
    located: int


def file_stats(path: str, source: Source, locate: Locator | None = None) -> FileStats:
    """Counts of one published parquet file (``source``: local path or file object)."""
    if path.startswith("labels/"):
        table = pq.read_table(source, columns=["decision", "failure_reason"])
        decisions = Counter(table.column("decision").to_pylist())
        failures = Counter(
            r
            for d, r in zip(
                table.column("decision").to_pylist(),
                table.column("failure_reason").to_pylist(),
                strict=True,
            )
            if d == "failed" and r
        )
        if locate is None:
            return FileStats(path, decisions=dict(decisions), failures=dict(failures))
        if not isinstance(source, Path):
            source.seek(0)  # the decision columns above already consumed a Hub stream
        where = locate(path, source)
        return FileStats(
            path,
            decisions=dict(decisions),
            failures=dict(failures),
            cells=where.cells,
            labelled=where.labelled,
            located=where.located,
        )
    table = pq.read_table(source, columns=["gpu"])
    gpus = Counter(gpu_key(g) for g in table.column("gpu").to_pylist())
    return FileStats(path, rows=table.num_rows, gpus=dict(gpus))


def ledger(dataset: str) -> str:
    return f"published/{dataset}.stats.jsonl"


def complete(
    store: WorkStore,
    dataset: str,
    published: set[str],
    opener: Opener,
    locate: Locator | None = None,
    *,
    refresh: set[str] | frozenset[str] = frozenset(),
    on_progress: Callable[[], None] | None = None,
) -> list[FileStats]:
    """Records for every published labels/generations file, counting any that is missing.

    With ``locate``, a labels record without its map cells is counted again; paths in
    ``refresh`` (just re-uploaded) are counted again whatever the ledger holds. Records are
    appended every :data:`RECORD_FLUSH` files (then ``on_progress`` runs), so a job stopped
    midway keeps what it counted; superseded ledger lines are compacted away.
    """
    have = {r["path"]: FileStats.from_json(r) for r in store.compact_jsonl(ledger(dataset))}
    wanted = sorted(p for p in published if p.startswith(("labels/", "generations/")))
    missing = [p for p in wanted if _needs_count(p, have, refresh, locate)]
    _count_missing(
        missing,
        lambda p: _count(opener, p, locate),
        lambda fresh: _record(store, dataset, have, fresh),
        on_progress,
    )
    return [have[p] for p in wanted if p in have]


def _count_missing(
    missing: list[str],
    count: Callable[[str], FileStats | None],
    record: Callable[[list[FileStats]], None],
    on_progress: Callable[[], None] | None,
) -> None:
    with ThreadPoolExecutor(COUNT_WORKERS) as pool:
        for start in range(0, len(missing), RECORD_FLUSH):
            fresh = [r for r in pool.map(count, missing[start : start + RECORD_FLUSH]) if r]
            record(fresh)
            if on_progress:
                on_progress()  # ledger appends stay in this thread


def _needs_count(
    path: str,
    have: dict[str, FileStats],
    refresh: set[str] | frozenset[str],
    locate: Locator | None,
) -> bool:
    """Not in the ledger, just re-uploaded, or a labels record still lacking its map cells."""
    if path not in have or path in refresh:
        return True
    return bool(locate) and path.startswith("labels/") and have[path].cells is None


def _record(
    store: WorkStore, dataset: str, have: dict[str, FileStats], fresh: list[FileStats]
) -> None:
    if fresh:
        store.append_jsonl(ledger(dataset), [r.to_json() for r in fresh])
        have.update({r.path: r for r in fresh})


@contextmanager
def _opened(opener: Opener, path: str) -> Iterator[Source]:
    """The opened source, closed afterwards when it is a stream (a Path has nothing to close)."""
    source = opener(path)
    try:
        yield source
    finally:
        if not isinstance(source, Path) and (close := getattr(source, "close", None)):
            close()


def _count(opener: Opener, path: str, locate: Locator | None) -> FileStats | None:
    """Count one file; a failure is logged and skipped (the next run counts it again)."""
    try:
        with _opened(opener, path) as source:
            return file_stats(path, source, locate)
    except Exception:
        log.exception("counting %s failed; skipped until the next run", path)
        return None


def totals(records: list[FileStats]) -> PublishedStats:
    decisions: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    gpus: Counter[str] = Counter()
    for r in records:
        decisions.update(r.decisions)
        failures.update(r.failures)
        gpus.update(r.gpus)
    return PublishedStats(
        decisions=dict(decisions),
        failures=dict(failures),
        gpus=dict(gpus),
        unique_texts=sum(r.rows for r in records),
        cells=geomap.merge([r.cells or {} for r in records]),
        labelled=sum(r.labelled for r in records),
        located=sum(r.located for r in records),
    )
