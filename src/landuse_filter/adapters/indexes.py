"""On-disk indexes that keep progress tracking and resolution flat at any scale.

* :class:`ProgressIndex` - distinct generated text hashes per chunk, fed incrementally
  from part manifests, so the controller never re-reads every manifest each cycle.
* :class:`ResolutionIndex` - text hash -> parsed verdict of its canonical generation,
  so publishing never holds tens of millions of generations in memory.
"""

import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

from landuse_filter.domain.parsing import parse_generation

INSERT_BATCH = 10_000


def _key(location: str) -> str:
    """``<fp>/<chunk>/<part>.json``: a manifest's identity, whatever root it sits under."""
    return "/".join(location.replace("\\", "/").split("/")[-3:])


class ProgressIndex:
    """Generated hashes of the chunks still in flight, plus every manifest ever consumed.

    A chunk's hashes are dropped once it is complete (:meth:`forget`), so the index stays
    small however many sentences a dataset has; ``seen`` keeps the manifests from being
    fetched again.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS seen (path TEXT PRIMARY KEY);"
            "CREATE TABLE IF NOT EXISTS done (chunk TEXT, sha TEXT, PRIMARY KEY (chunk, sha));"
        )

    def ingest(self, manifests: Iterable[Path]) -> int:
        """Record manifests not seen before; returns how many were new."""
        new = 0
        with self.db:
            for path in manifests:
                cur = self.db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (_key(str(path)),))
                if not cur.rowcount:
                    continue
                new += 1
                shas = json.loads(path.read_text(encoding="utf-8"))["text_sha256s"]
                chunk = path.parent.name
                self.db.executemany(
                    "INSERT OR IGNORE INTO done VALUES (?, ?)", [(chunk, s) for s in shas]
                )
        return new

    def has_seen(self, manifest_location: str) -> bool:
        """Whether a manifest (repo path or local path) was already ingested."""
        row = self.db.execute("SELECT 1 FROM seen WHERE path = ?", (_key(manifest_location),))
        return row.fetchone() is not None

    def forget(self, chunks: Iterable[str]) -> int:
        """Drop the hashes of finished chunks (their manifests stay marked as seen)."""
        with self.db:
            return sum(
                self.db.execute("DELETE FROM done WHERE chunk = ?", (chunk,)).rowcount
                for chunk in chunks
            )

    def compact(self) -> None:
        """Give the freed pages back to the disk."""
        self.db.execute("VACUUM")

    def count(self, chunk: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM done WHERE chunk = ?", (chunk,)).fetchone()[0]

    def shas(self, chunk: str) -> set[str]:
        return {s for (s,) in self.db.execute("SELECT sha FROM done WHERE chunk = ?", (chunk,))}


class ResolutionIndex:
    """Mapping-like: ``get(text_sha256)`` -> (decision, mode, failure) or None."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS verdict "
            "(sha TEXT PRIMARY KEY, decision TEXT, mode TEXT, failure TEXT)"
        )

    def build(self, rows: Iterator[dict]) -> int:
        """Insert canonical generations in order; the first row per hash wins."""
        batch, n = [], 0
        with self.db:
            for row in rows:
                v = parse_generation(row["raw_output"], truncated=bool(row["truncated"]))
                batch.append(
                    (
                        row["text_sha256"],
                        v.decision.value,
                        v.mode and v.mode.value,
                        v.failure and v.failure.value,
                    )
                )
                if len(batch) >= INSERT_BATCH:
                    n += self._insert(batch)
                    batch = []
            n += self._insert(batch)
        return n

    def _insert(self, batch: list[tuple]) -> int:
        return self.db.executemany(
            "INSERT OR IGNORE INTO verdict VALUES (?, ?, ?, ?)", batch
        ).rowcount

    def get(self, sha: str) -> tuple[str, str | None, str | None] | None:
        return self.db.execute(
            "SELECT decision, mode, failure FROM verdict WHERE sha = ?", (sha,)
        ).fetchone()
