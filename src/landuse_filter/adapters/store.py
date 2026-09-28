"""The work store: a local directory mirrored to a private Hugging Face Bucket.

Layout (identical locally, on site spools and in the bucket)::

    plans/<dataset>/<fp>/chunks.jsonl        chunk id, order, size (append-only)
    chunks/<chunk_id>.parquet                text_sha256, text, input_ids
    parts/<fp>/<chunk_id>/<part_sha>.parquet generation rows
    assignments/<assignment_id>.json         chunk ids + serving config for one job
    jobs/<site>/<job_id>.json                job summaries
    gates/<serving_fp>[/<gpu>].json          benchmark / per-GPU gate results

Files are content-addressed or written once; ``write_atomic`` makes a crash leave either
the old file or the new one, never a torn one. A part's name is the sha256 of its
bytes, so a corrupt part is detected by re-hashing.
"""

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


class CorruptPartError(ValueError):
    """A part file's bytes no longer match the sha256 in its name."""


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def table_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue().to_pybytes()


class WorkStore:
    """File-level access to a work tree rooted at ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, relative: str) -> Path:
        return self.root / relative

    def write_json(self, relative: str, payload: Any) -> None:
        write_atomic(self.path(relative), json.dumps(payload, indent=1, sort_keys=True).encode())

    def read_json(self, relative: str) -> Any:
        return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def exists(self, relative: str) -> bool:
        return self.path(relative).exists()

    def append_jsonl(self, relative: str, rows: list[dict]) -> None:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_jsonl(self, relative: str) -> list[dict]:
        path = self.path(relative)
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a torn final line from a crash: everything before it is valid
        return rows

    # --- chunk inputs -------------------------------------------------------------

    def write_chunk(self, chunk_id: str, table: pa.Table) -> None:
        write_atomic(self.path(f"chunks/{chunk_id}.parquet"), table_bytes(table))

    def read_chunk(self, chunk_id: str) -> pa.Table:
        return pq.read_table(self.path(f"chunks/{chunk_id}.parquet"))

    # --- result parts -------------------------------------------------------------

    def write_part(self, fp: str, chunk_id: str, table: pa.Table) -> str:
        data = table_bytes(table)
        part = hashlib.sha256(data).hexdigest()
        target = self.path(f"parts/{fp}/{chunk_id}/{part}.parquet")
        if not target.exists():
            write_atomic(target, data)
        return part

    def part_paths(self, fp: str, chunk_id: str | None = None) -> Iterator[Path]:
        base = self.path(f"parts/{fp}")
        pattern = f"{chunk_id}/*.parquet" if chunk_id else "*/*.parquet"
        yield from sorted(base.glob(pattern))

    def read_part(self, path: Path) -> pa.Table:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != path.stem:
            raise CorruptPartError(str(path))
        return pq.read_table(pa.BufferReader(data))


class Bucket:
    """Mirror of a ``WorkStore`` in a private Hugging Face Bucket (ADR-0009)."""

    def __init__(self, bucket_id: str) -> None:
        self.bucket_id = bucket_id

    def ensure(self) -> None:
        from huggingface_hub import HfApi

        HfApi().create_bucket(self.bucket_id, private=True, exist_ok=True)

    def push(self, store: WorkStore, prefix: str = "") -> None:
        from huggingface_hub import HfApi

        HfApi().sync_bucket(
            str(store.path(prefix)), f"hf://buckets/{self.bucket_id}/{prefix}", quiet=True
        )

    def pull(self, store: WorkStore, prefix: str = "") -> None:
        from huggingface_hub import HfApi

        store.path(prefix).mkdir(parents=True, exist_ok=True)
        HfApi().sync_bucket(
            f"hf://buckets/{self.bucket_id}/{prefix}", str(store.path(prefix)), quiet=True
        )
