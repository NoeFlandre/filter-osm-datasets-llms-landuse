"""Incremental, resumable planning of one input dataset into work chunks.

A crash between writing a chunk and recording it re-emits the *same* chunk id on the
next run (grouping is deterministic), so ``chunks.jsonl`` readers dedupe by id.

State lives in ``index/<dataset>.sqlite``: which input files were scanned, every
unique sentence text seen (first file index, chunk id once emitted) and per-file
counts. Re-running continues where it stopped; emitted chunks never change.
"""

import fnmatch
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from landuse_filter.adapters.readers import DATASET_RANK, SOURCES, Source
from landuse_filter.adapters.schema import CHUNK
from landuse_filter.adapters.store import WorkStore
from landuse_filter.domain.planning import UniqueText, plan_chunks
from landuse_filter.domain.prompting import render_prompt

# Batch tokeniser: prompts -> token ids, same order (fast path, see adapters.tokenizer).
Encode = Callable[[list[str]], list[list[int]]]

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (idx INTEGER PRIMARY KEY, path TEXT UNIQUE, done INTEGER DEFAULT 0,
    sentences INTEGER, unsplit INTEGER, new_unique INTEGER);
CREATE TABLE IF NOT EXISTS texts (sha TEXT PRIMARY KEY, text TEXT, file_idx INTEGER, chunk_id TEXT);
CREATE INDEX IF NOT EXISTS texts_pending ON texts (chunk_id, file_idx, sha);
"""


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


class Planner:
    def __init__(self, store: WorkStore, dataset: str, config_fp: str) -> None:
        self.store = store
        self.source = SOURCES[dataset]
        self.fp = config_fp
        db = store.path(f"index/{dataset}.sqlite")
        db.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db)
        self.db.executescript(SCHEMA)

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
        query = "SELECT sha, text, file_idx FROM texts WHERE chunk_id IS NULL"
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

    def _emit_group(self, rows: list, encode: Encode, template: str, chunk_size: int) -> int:
        batch = encode([render_prompt(template, text) for _, text, _ in rows])
        ids = {sha: batch[i] for i, (sha, _, _) in enumerate(rows)}
        rank = DATASET_RANK[self.source.dataset]
        texts = [UniqueText(sha, len(ids[sha]), (rank, idx)) for sha, _, idx in rows]
        (chunk,) = plan_chunks(texts, self.fp, chunk_size)
        text_of = {sha: text for sha, text, _ in rows}
        table = pa.table(
            {
                "text_sha256": list(chunk.text_sha256s),
                "text": [text_of[s] for s in chunk.text_sha256s],
                "input_ids": [ids[s] for s in chunk.text_sha256s],
            },
            schema=CHUNK,
        )
        self.store.write_chunk(chunk.chunk_id, table)
        self.store.append_jsonl(
            f"plans/{self.source.dataset}/{self.fp}/chunks.jsonl",
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "order": list(chunk.order),
                    "size": len(rows),
                    "prompt_tokens": sum(t.prompt_tokens for t in texts),
                }
            ],
        )
        with self.db:
            self.db.executemany(
                "UPDATE texts SET chunk_id = ? WHERE sha = ?",
                [(chunk.chunk_id, s) for s, _, _ in rows],
            )
        return 1

    def report(self) -> PlanReport:
        f = self.db.execute(
            "SELECT COUNT(*), SUM(done), SUM(sentences), SUM(unsplit) FROM files"
        ).fetchone()
        t = self.db.execute("SELECT COUNT(*), COUNT(DISTINCT chunk_id) FROM texts").fetchone()
        return PlanReport(f[1] or 0, f[0], f[2] or 0, f[3] or 0, t[0], t[1])
