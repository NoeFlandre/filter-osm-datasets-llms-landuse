"""Incremental, resumable planning of one input dataset into work chunks.

A crash between writing a chunk and recording it re-emits the *same* chunk id on the
next run (grouping is deterministic), so ``chunks.jsonl`` readers dedupe by id.

State lives in ``index/<dataset>.sqlite``: which input files were scanned, every
unique sentence text seen (first file index, chunk id once emitted) and per-file
counts. Re-running continues where it stopped; emitted chunks never change.
"""

import fnmatch
import hashlib
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import pyarrow as pa

from landuse_filter.adapters.readers import DATASET_RANK, SOURCES, Source
from landuse_filter.adapters.schema import CHUNK
from landuse_filter.adapters.store import WorkStore
from landuse_filter.domain.planning import Chunk, UniqueText, chunk_id, plan_chunks
from landuse_filter.domain.prompting import render_prompt

# Batch tokeniser: prompts -> token ids, same order (fast path, see adapters.tokenizer).
Encode = Callable[[list[str]], list[list[int]]]

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (idx INTEGER PRIMARY KEY, path TEXT UNIQUE, done INTEGER DEFAULT 0,
    sentences INTEGER, unsplit INTEGER, new_unique INTEGER);
CREATE TABLE IF NOT EXISTS texts (sha TEXT PRIMARY KEY, text TEXT, file_idx INTEGER, chunk_id TEXT);
CREATE INDEX IF NOT EXISTS texts_pending ON texts (chunk_id, file_idx, sha);
CREATE TABLE IF NOT EXISTS planner_meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
# Geographic re-planning (ADR-0014) adds these columns to indexes created before it.
MIGRATIONS = {
    "done": "ALTER TABLE texts ADD COLUMN done INTEGER DEFAULT 0",
    "cell": "ALTER TABLE texts ADD COLUMN cell TEXT",
}


def _cell_key(cell: str | None) -> str:
    """Deterministic tie-break between cells of one round (spreads them, not alphabetical)."""
    return hashlib.sha256((cell or "").encode()).hexdigest()[:16]


def _legacy_has_unassigned_completions(db: sqlite3.Connection) -> bool:
    columns = {row[1] for row in db.execute("PRAGMA table_info(texts)")}
    return "done" in columns and bool(
        db.execute("SELECT 1 FROM texts WHERE done = 1 AND chunk_id IS NULL LIMIT 1").fetchone()
    )


def _legacy_chunks_match(db: sqlite3.Connection, config_fp: str) -> bool:
    rows = db.execute(
        "SELECT chunk_id, sha FROM texts WHERE chunk_id IS NOT NULL ORDER BY chunk_id, sha"
    )
    for old_chunk, chunk_rows in groupby(rows, lambda row: row[0]):
        if chunk_id(config_fp, (row[1] for row in chunk_rows)) != old_chunk:
            return False
    return True


def _bind_fingerprint(db: sqlite3.Connection, config_fp: str) -> None:
    """Bind this index to one generation config, validating any legacy chunk ids first."""
    bound = db.execute("SELECT value FROM planner_meta WHERE name = 'config_fp'").fetchone()
    if bound:
        if bound[0] != config_fp:
            raise ValueError(f"Planner index is bound to fingerprint {bound[0]}, not {config_fp}")
        return

    if _legacy_has_unassigned_completions(db):
        raise ValueError("Cannot bind legacy planner index with unassigned completed texts")
    if not _legacy_chunks_match(db, config_fp):
        raise ValueError(
            f"Cannot bind legacy planner index: stored chunks do not match fingerprint {config_fp}"
        )

    with db:
        db.execute("INSERT INTO planner_meta (name, value) VALUES ('config_fp', ?)", (config_fp,))


@dataclass(frozen=True, slots=True)
class PlanReport:
    files_done: int
    files_total: int
    sentences: int
    unsplit: int
    unique_texts: int
    chunks_emitted: int


def list_input_files(source: Source, revision: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(source.repo_id, repo_type="dataset", revision=revision)
    return sorted(f for f in files if any(fnmatch.fnmatch(f, p) for p in source.patterns))


def download(source: Source, path: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source.repo_id, path, repo_type="dataset", revision=revision))


def _chunk_table(chunk: Chunk, ids: dict[str, list[int]], text_of: dict[str, str]) -> pa.Table:
    shas = list(chunk.text_sha256s)
    return pa.table(
        {
            "text_sha256": shas,
            "text": [text_of[s] for s in shas],
            "input_ids": [ids[s] for s in shas],
        },
        schema=CHUNK,
    )


def _plan_line(chunk: Chunk, texts: list[UniqueText]) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "order": list(chunk.order),
        "size": len(texts),
        "prompt_tokens": sum(t.prompt_tokens for t in texts),
    }


class Planner:
    def __init__(self, store: WorkStore, dataset: str, config_fp: str) -> None:
        self.store = store
        self.source = SOURCES[dataset]
        self.fp = config_fp
        db = store.path(f"index/{dataset}.sqlite")
        db.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db)
        try:
            self.db.executescript(SCHEMA)
            have = {row[1] for row in self.db.execute("PRAGMA table_info(texts)")}
            for column, statement in MIGRATIONS.items():
                if column not in have:
                    self.db.execute(statement)
            self.db.commit()
            _bind_fingerprint(self.db, self.fp)
        except Exception:
            self.db.close()
            raise

    def register(self, files: Sequence[str]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO files (idx, path) VALUES (?, ?)", list(enumerate(files))
            )

    def scan(self, fetch: Callable[[str], Path], limit: int | None = None) -> int:
        """Read pending files in order, recording unique texts; returns files scanned."""
        pending = self.db.execute(
            "SELECT idx, path FROM files WHERE done = 0 ORDER BY idx"
        ).fetchall()
        batch = pending[:limit] if limit else pending
        for idx, path in batch:
            self._scan_file(idx, path, fetch(path))
        return len(batch)

    def _scan_file(self, idx: int, path: str, local: Path) -> None:
        sentences = unsplit = new = 0
        with self.db:
            for ref in self.source.read(local, path):
                sentences += 1
                if ref.unsplit:
                    unsplit += 1
                    continue
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO texts (sha, text, file_idx) VALUES (?, ?, ?)",
                    (ref.text_sha256, ref.text, idx),
                )
                new += cur.rowcount
            self.db.execute(
                "UPDATE files SET done = 1, sentences = ?, unsplit = ?, new_unique = ? "
                "WHERE idx = ?",
                (sentences, unsplit, new, idx),
            )

    def emit(self, encode: Encode, template: str, chunk_size: int, *, final: bool) -> int:
        """Turn pending texts of scanned files into chunks; partial tail only if ``final``."""
        boundary = self.db.execute("SELECT MIN(idx) FROM files WHERE done = 0").fetchone()[0]
        query = "SELECT sha, text, file_idx FROM texts WHERE chunk_id IS NULL AND done = 0"
        args: tuple = ()
        if boundary is not None:
            query += " AND file_idx < ?"
            args = (boundary,)
        rows = self.db.execute(query + " ORDER BY file_idx, sha", args).fetchall()
        usable = len(rows) if final else len(rows) - len(rows) % chunk_size
        emitted = 0
        for start in range(0, usable, chunk_size):
            emitted += self._emit_group(
                rows[start : start + chunk_size], encode, template, chunk_size
            )
        return emitted

    # --- geographic re-planning -----------------------------------------------------

    @property
    def plan_path(self) -> str:
        return f"plans/{self.source.dataset}/{self.fp}/chunks.jsonl"

    def mark_done(self, shas: Iterable[str]) -> None:
        """Record texts that already have a generation (they are never planned again)."""
        with self.db:
            self.db.executemany("UPDATE texts SET done = 1 WHERE sha = ?", ((s,) for s in shas))

    def release_unfinished(self) -> set[str]:
        """Take every chunk with an ungenerated text out of the plan and free its texts.

        Finished chunks keep their plan line; the others are returned (their chunk files
        stay in the bucket, unused) so the texts can be planned again in a new order.
        """
        released = {
            cid
            for (cid,) in self.db.execute(
                "SELECT chunk_id FROM texts WHERE chunk_id IS NOT NULL "
                "GROUP BY chunk_id HAVING MIN(done) = 0"
            )
        }
        if not released:
            return released
        kept = [r for r in self.store.read_jsonl(self.plan_path) if r["chunk_id"] not in released]
        self.store.path(self.plan_path).unlink(missing_ok=True)
        self.store.append_jsonl(self.plan_path, kept)
        with self.db:
            self.db.executemany(
                "UPDATE texts SET chunk_id = NULL WHERE chunk_id = ?", ((c,) for c in released)
            )
        return released

    def set_cells(self, cells: Iterable[tuple[str, str]]) -> None:
        """Give texts their map cell (first location wins; texts may repeat across places)."""
        with self.db:
            self.db.executemany(
                "UPDATE texts SET cell = ? WHERE sha = ? AND cell IS NULL",
                ((cell, sha) for sha, cell in cells),
            )

    def emit_uniform(self, encode: Encode, template: str, chunk_size: int) -> int:
        """Plan every pending text in a geographically uniform order.

        Round ``j`` holds the ``j``-th text (by hash) of every cell, so any prefix of the plan
        takes an equal share from each cell still holding texts. Texts without a location get
        a hash-derived round, which spreads them evenly through the whole order.
        """
        order = self._uniform_order()
        emitted = 0
        while rows := order.fetchmany(chunk_size):
            emitted += self._emit_group(rows, encode, template, chunk_size)
        return emitted

    def _uniform_order(self) -> sqlite3.Cursor:
        self.db.create_function("cell_key", 1, _cell_key, deterministic=True)
        self.db.create_function("hash_round", 1, lambda sha: int(sha[:12], 16), deterministic=True)
        self.db.executescript(
            """
            DROP TABLE IF EXISTS temp.ranked;
            CREATE TEMP TABLE ranked AS
              SELECT sha, cell_key(cell) AS ck,
                     ROW_NUMBER() OVER (PARTITION BY cell ORDER BY sha) - 1 AS rnd
              FROM texts WHERE chunk_id IS NULL AND done = 0 AND cell IS NOT NULL;
            """
        )
        top = self.db.execute("SELECT COALESCE(MAX(rnd), 0) FROM temp.ranked").fetchone()[0]
        self.db.execute(
            "INSERT INTO temp.ranked SELECT sha, '', hash_round(sha) % ? FROM texts "
            "WHERE chunk_id IS NULL AND done = 0 AND cell IS NULL",
            (top + 1,),
        )
        self.db.execute("CREATE INDEX temp.ranked_order ON ranked (rnd, ck, sha)")
        return self.db.execute(
            "SELECT t.sha, t.text, t.file_idx FROM temp.ranked r JOIN texts t ON t.sha = r.sha "
            "ORDER BY r.rnd, r.ck, r.sha"
        )

    def _emit_group(self, rows: list, encode: Encode, template: str, chunk_size: int) -> int:
        batch = encode([render_prompt(template, text) for _, text, _ in rows])
        ids = {sha: batch[i] for i, (sha, _, _) in enumerate(rows)}
        rank = DATASET_RANK[self.source.dataset]
        texts = [UniqueText(sha, len(ids[sha]), (rank, idx)) for sha, _, idx in rows]
        (chunk,) = plan_chunks(texts, self.fp, chunk_size)
        table = _chunk_table(chunk, ids, {sha: text for sha, text, _ in rows})
        self.store.write_chunk(chunk.chunk_id, table)
        self.store.append_jsonl(
            f"plans/{self.source.dataset}/{self.fp}/chunks.jsonl",
            [_plan_line(chunk, texts)],
        )
        with self.db:
            self.db.executemany(
                "UPDATE texts SET chunk_id = ? WHERE sha = ?",
                [(chunk.chunk_id, s) for s, _, _ in rows],
            )
        return 1

    def recover_plan_lines(self) -> int:
        """Re-create plan lines the index assigned to chunks but the plan file lost.

        The order key is the chunk's earliest input file (priority only); prompt_tokens is
        not recoverable from the index and nothing downstream reads it.
        """
        path = f"plans/{self.source.dataset}/{self.fp}/chunks.jsonl"
        known = {row["chunk_id"] for row in self.store.read_jsonl(path)}
        rank = DATASET_RANK[self.source.dataset]
        rows = self.db.execute(
            "SELECT chunk_id, COUNT(*), MIN(file_idx) FROM texts "
            "WHERE chunk_id IS NOT NULL GROUP BY chunk_id"
        ).fetchall()
        lost = [
            {"chunk_id": cid, "order": [rank, first], "size": n, "prompt_tokens": 0}
            for cid, n, first in rows
            if cid not in known
        ]
        self.store.append_jsonl(path, lost)
        return len(lost)

    def report(self) -> PlanReport:
        f = self.db.execute(
            "SELECT COUNT(*), SUM(done), SUM(sentences), SUM(unsplit) FROM files"
        ).fetchone()
        t = self.db.execute("SELECT COUNT(*), COUNT(DISTINCT chunk_id) FROM texts").fetchone()
        return PlanReport(f[1] or 0, f[0], f[2] or 0, f[3] or 0, t[0], t[1])
