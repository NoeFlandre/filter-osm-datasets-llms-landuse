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


def _read_hashes(path: Path) -> list[str] | None:
    """The text hashes of a manifest, or ``None`` when the file is unreadable or malformed."""
    try:
        return list(json.loads(path.read_text(encoding="utf-8"))["text_sha256s"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


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
                if self.has_seen(str(path)):
                    continue
                shas = _read_hashes(path)
                if shas is None:  # a truncated download: drop it, the next pull fetches it again
                    path.unlink(missing_ok=True)
                    continue
                self.db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (_key(str(path)),))
                new += 1
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
    """Mapping-like: ``get(text_sha256)`` -> (decision, mode, failure) or None.

    Built either in one pass (:meth:`build`, first row per hash wins) or part by part
    (:meth:`add_part`): each verdict then remembers the smallest part key that produced it, so the
    outcome does not depend on the order parts are added in, and the index can travel through the
    bucket as a snapshot that later jobs extend instead of re-reading every part.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS verdict "
            "(sha TEXT PRIMARY KEY, decision TEXT, mode TEXT, failure TEXT, part TEXT)"
        )
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(verdict)")}
        if "part" not in columns:  # an index built before parts were tracked
            self.db.execute("ALTER TABLE verdict ADD COLUMN part TEXT")
        self.db.execute("CREATE TABLE IF NOT EXISTS parts (key TEXT PRIMARY KEY)")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT)")
        self.db.commit()

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
            "INSERT OR IGNORE INTO verdict (sha, decision, mode, failure) VALUES (?, ?, ?, ?)",
            batch,
        ).rowcount

    def add_part(
        self, key: str, verdicts: Iterable[tuple[str, str, str | None, str | None]]
    ) -> None:
        """Record a part's ``(sha, decision, mode, failure)`` rows; the smallest part key wins."""
        with self.db:
            self.db.executemany(
                "INSERT INTO verdict VALUES (?, ?, ?, ?, ?) ON CONFLICT(sha) DO UPDATE SET "
                "decision = excluded.decision, mode = excluded.mode, failure = excluded.failure, "
                "part = excluded.part WHERE excluded.part < verdict.part",
                [(*v, key) for v in verdicts],
            )
            self.db.execute("INSERT OR IGNORE INTO parts VALUES (?)", (key,))

    def parts(self) -> set[str]:
        """Keys of the parts already added."""
        return {k for (k,) in self.db.execute("SELECT key FROM parts")}

    def parts_of(self, shas: Iterable[str]) -> dict[str, str]:
        """sha -> key of the part holding its canonical generation (only parts-tracked rows)."""
        found: dict[str, str] = {}
        for sha in shas:
            row = self.db.execute("SELECT part FROM verdict WHERE sha = ?", (sha,)).fetchone()
            if row and row[0]:
                found[sha] = row[0]
        return found

    def meta(self, name: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def set_meta(self, name: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (name, value))

    def get(self, sha: str) -> tuple[str, str | None, str | None] | None:
        return self.db.execute(
            "SELECT decision, mode, failure FROM verdict WHERE sha = ?", (sha,)
        ).fetchone()
