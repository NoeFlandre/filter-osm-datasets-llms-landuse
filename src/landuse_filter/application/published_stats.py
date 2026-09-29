"""Deterministic statistics of the published tables, one record per published file.

The dataset card is computed from these records, never from whatever a single run built:
each ``labels/`` file contributes its decision and failure-reason counts, each
``generations/`` file its row count and rows per GPU type. Records are cached in
``published/<dataset>.stats.jsonl``; a published file without a record is (re)counted
from the Hub, so the card is always a function of the published data.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import IO

import pyarrow.parquet as pq

from landuse_filter.adapters.store import WorkStore
from landuse_filter.domain.gpu import gpu_key

Source = Path | IO[bytes]
Opener = Callable[[str], Source]  # repo path -> local file or Hub file object


def file_stats(path: str, source: Source) -> dict:
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
        return {"path": path, "decisions": dict(decisions), "failures": dict(failures)}
    table = pq.read_table(source, columns=["gpu"])
    gpus = Counter(gpu_key(g) for g in table.column("gpu").to_pylist())
    return {"path": path, "rows": table.num_rows, "gpus": dict(gpus)}


def ledger(dataset: str) -> str:
    return f"published/{dataset}.stats.jsonl"


def complete(store: WorkStore, dataset: str, published: set[str], opener: Opener) -> list[dict]:
    """Records for every published labels/generations file, counting any that is missing."""
    have = {r["path"]: r for r in store.read_jsonl(ledger(dataset))}
    wanted = sorted(p for p in published if p.startswith(("labels/", "generations/")))
    missing = [p for p in wanted if p not in have]
    if missing:
        fresh = [file_stats(p, opener(p)) for p in missing]
        store.append_jsonl(ledger(dataset), fresh)
        have.update({r["path"]: r for r in fresh})
    return [have[p] for p in wanted]


def totals(records: list[dict]) -> dict:
    decisions: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    gpus: Counter[str] = Counter()
    unique = 0
    for r in records:
        decisions.update(r.get("decisions", {}))
        failures.update(r.get("failures", {}))
        gpus.update(r.get("gpus", {}))
        unique += r.get("rows", 0)
    return {
        "decisions": dict(decisions),
        "failures": dict(failures),
        "gpus": dict(gpus),
        "unique_texts": unique,
    }
