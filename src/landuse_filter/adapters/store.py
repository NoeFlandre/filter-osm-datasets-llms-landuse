"""The work store: a local directory mirrored to a private Hugging Face Bucket.

Layout (identical locally, on node scratch and in the bucket)::

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
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq


class CorruptPartError(ValueError):
    """A part file's bytes no longer match the sha256 in its name."""


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def table_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue().to_pybytes()


class CorruptJSONLError(ValueError):
    """A newline-terminated JSONL record could not be decoded safely."""

    def __init__(self, path: Path, line_number: int, byte_offset: int, detail: str) -> None:
        self.path = path
        self.line_number = line_number
        self.byte_offset = byte_offset
        super().__init__(f"{path}: {detail} at line {line_number}, byte offset {byte_offset}")


def _jsonl_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _iter_jsonl_records(file: BinaryIO, path: Path) -> Iterator[dict]:
    """Read complete records; only a malformed final fragment without LF may be torn."""
    line_number = 1
    byte_offset = 0
    while raw := file.readline():
        terminated = raw.endswith(b"\n")
        content = raw[:-1] if terminated else raw
        try:
            line = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            if not terminated:
                break
            raise CorruptJSONLError(
                path, line_number, byte_offset + exc.start, "invalid UTF-8"
            ) from exc
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if not terminated:
                break
            error_offset = byte_offset + len(line[: exc.pos].encode("utf-8"))
            raise CorruptJSONLError(
                path, line_number, error_offset, f"invalid JSON ({exc.msg})"
            ) from exc
        yield row
        byte_offset += len(raw)
        line_number += 1


def _last_line_start(file: BinaryIO, end: int) -> int:
    """Find the byte offset after the last newline without reading the full ledger."""
    position = end
    while position:
        start = max(0, position - 8192)
        file.seek(start)
        newline = file.read(position - start).rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        position = start
    return 0


def _prepare_jsonl_append(file: BinaryIO) -> None:
    """Drop a torn final record or separate a valid final record from the next append."""
    file.seek(0, os.SEEK_END)
    end = file.tell()
    if end == 0:
        return
    file.seek(end - 1)
    if file.read(1) == b"\n":
        return

    start = _last_line_start(file, end)
    file.seek(start)
    tail = file.read()
    try:
        json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        file.seek(start)
        file.truncate()
    else:
        file.seek(0, os.SEEK_END)
        file.write(b"\n")


class WorkStore:
    """File-level access to a work tree rooted at ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._validated_jsonl: dict[str, tuple[int, int, int, int, int]] = {}

    def path(self, relative: str) -> Path:
        return self.root / relative

    def write_json(self, relative: str, payload: Any) -> None:
        write_atomic(self.path(relative), json.dumps(payload, indent=1, sort_keys=True).encode())

    def read_json(self, relative: str) -> Any:
        return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def admitted_gates(self) -> dict[str, Any]:
        """Admission gate of every GPU type whose benchmark was admitted, by GPU key."""
        gates: dict[str, Any] = {}
        for path in sorted(self.path("gates/admission").glob("*.json")):
            record = self.read_json(str(path.relative_to(self.root)))
            if record.get("status") == "admitted":
                gates[path.stem] = record["gate"]
        return gates

    def exists(self, relative: str) -> bool:
        return self.path(relative).exists()

    def append_jsonl(self, relative: str, rows: list[dict]) -> None:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows)
        existed = path.exists()
        with path.open("r+b" if existed else "w+b") as f:
            if existed and self._validated_jsonl.get(relative) != _jsonl_signature(path):
                for _ in _iter_jsonl_records(f, path):
                    pass
            _prepare_jsonl_append(f)
            f.seek(0, os.SEEK_END)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        self._validated_jsonl[relative] = _jsonl_signature(path)

    def compact_jsonl(self, relative: str, key: str = "path") -> list[dict]:
        """Read a ledger; when superseded lines (same ``key``, last wins) pile up, rewrite it
        with one line per key. Returns the surviving rows in first-seen order."""
        rows = self.read_jsonl(relative)
        latest = {r[key]: r for r in rows}
        if len(rows) > 2 * len(latest) + 100:
            write_atomic(
                self.path(relative),
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in latest.values()).encode(),
            )
        return list(latest.values())

    def read_jsonl(self, relative: str) -> list[dict]:
        path = self.path(relative)
        if not path.exists():
            return []
        with path.open("rb") as f:
            rows = list(_iter_jsonl_records(f, path))
        self._validated_jsonl[relative] = _jsonl_signature(path)
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
        # An existing file with the same name is only kept if its bytes still match:
        # regenerating identical content must repair a corrupt copy (regression).
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != part:
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
