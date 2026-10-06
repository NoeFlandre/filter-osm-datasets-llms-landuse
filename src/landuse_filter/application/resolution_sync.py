"""The resolution index as a bucket snapshot that every publish job extends (ADR-0026).

Re-reading tens of thousands of result parts at the start of each publish job cost the whole
day walltime. The index (text hash -> parsed verdict + the part that produced it, plus the set of
parts already read) is checkpointed to ``index/resolve-<fp>.sqlite``; a job downloads it, lists
the bucket's parts and only reads the ones it has not seen, in batches that are downloaded while
the previous batch is parsed (in parallel processes), deleting each part once indexed.
"""

import hashlib
import multiprocessing
import os
import shutil
import time
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from landuse_filter.adapters.indexes import ResolutionIndex
from landuse_filter.adapters.remote import Remote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.job_progress import NULL, Progress
from landuse_filter.domain.parsing import PARSER_VERSION, parse_generation

BATCH = 1000  # parts downloaded (and parsed) at a time
CHECKPOINT_SECONDS = 300.0  # snapshot upload cadence while catching up
DOWNLOAD_STREAMS = 4  # concurrent bucket downloads inside a batch
Verdicts = list[tuple[str, str, str | None, str | None]]


def snapshot_path(fp: str) -> str:
    return f"index/resolve-{fp}.sqlite"


def part_key(path: str) -> str:
    """``parts/<fp>/<chunk>/<part>.parquet`` -> ``<chunk>/<part>``."""
    return path.removesuffix(".parquet").split("/", 2)[2]


def part_path(fp: str, key: str) -> str:
    return f"parts/{fp}/{key}.parquet"


def read_verdicts(path: str) -> Verdicts | None:
    """The parsed rows of one part (first row per hash), or ``None`` when it is corrupt.

    Module level so a process pool can run it."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != Path(path).stem:
        return None
    try:
        rows = pq.read_table(pa.BufferReader(data)).to_pylist()
    except (pa.ArrowException, OSError):
        return None
    seen: set[str] = set()
    out: Verdicts = []
    for row in rows:
        sha = row["text_sha256"]
        if sha in seen:
            continue
        seen.add(sha)
        v = parse_generation(row["raw_output"], truncated=bool(row["truncated"]))
        out.append((sha, v.decision.value, v.mode and v.mode.value, v.failure and v.failure.value))
    return out


def _restore_snapshot(
    remote: Remote, scratch: WorkStore, fp: str, progress: Progress = NULL
) -> ResolutionIndex:
    """The bucket's snapshot when it was made by the same parser, else an empty index."""
    target = scratch.path(snapshot_path(fp))
    target.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path(fp) in remote.ls("index/"):
        staged = target.with_name(target.name + ".download")
        progress.event("snapshot_download_start", file=snapshot_path(fp))
        t0 = time.monotonic()
        remote.get([(snapshot_path(fp), staged)])
        progress.event(
            "snapshot_download_done",
            bytes=staged.stat().st_size,
            seconds=round(time.monotonic() - t0),
        )
        shutil.copyfile(staged, target)  # a writable copy: the download may be a cache link
        target.chmod(0o644)
        staged.unlink()
    index = ResolutionIndex(target)
    if index.meta("parser") != PARSER_VERSION:
        index.db.close()
        target.unlink()
        index = ResolutionIndex(target)
        index.set_meta("parser", PARSER_VERSION)
    return index


def _save(
    remote: Remote, scratch: WorkStore, index: ResolutionIndex, fp: str, progress: Progress = NULL
) -> None:
    index.db.commit()
    size, t0 = scratch.path(snapshot_path(fp)).stat().st_size, time.monotonic()
    remote.put([(scratch.path(snapshot_path(fp)), snapshot_path(fp))])
    progress.event("snapshot_upload", bytes=size, seconds=round(time.monotonic() - t0))


def _download(remote: Remote, scratch: WorkStore, fp: str, keys: list[str]) -> list[Path]:
    paths = [part_path(fp, k) for k in keys]
    size = -(-len(paths) // DOWNLOAD_STREAMS)
    chunks = [paths[i : i + size] for i in range(0, len(paths), size)]
    with ThreadPoolExecutor(DOWNLOAD_STREAMS) as pool:
        list(pool.map(lambda c: remote.get([(p, scratch.path(p)) for p in c]), chunks))
    return [scratch.path(p) for p in paths]


def _new_part_keys(remote: Remote, index: ResolutionIndex, fp: str) -> list[str]:
    """Keys of the bucket's parts the index has not read yet, sorted."""
    seen = index.parts()
    on_bucket = {part_key(p) for p in remote.ls(f"parts/{fp}/") if p.endswith(".parquet")}
    return sorted(k for k in on_bucket if k not in seen)


def _prefetched(
    batches: list[list[str]],
    download: Callable[[list[str]], list[Path]],
    io: ThreadPoolExecutor,
    should_stop: Callable[[], bool],
) -> Generator[tuple[list[str], list[Path]]]:
    """``(keys, paths)`` per batch, the next batch downloading while the caller indexes.

    Ends before the next batch on a stop request; closing it cancels the pending download."""
    inflight = [io.submit(download, b) for b in batches[:1]]
    try:
        for i, keys in enumerate(batches):
            if should_stop():
                return
            paths = inflight.pop().result()
            inflight += [io.submit(download, b) for b in batches[i + 1 : i + 2]]
            yield keys, paths
    finally:
        for future in inflight:
            future.cancel()


class _Checkpointer:
    """Uploads the snapshot every ``seconds`` of catch-up, and once more if anything is unsaved."""

    def __init__(self, save: Callable[[], None], seconds: float) -> None:
        self.save, self.seconds = save, seconds
        self.last, self.dirty = time.monotonic(), False

    def batch_indexed(self) -> None:
        self.dirty = True
        if time.monotonic() - self.last >= self.seconds:
            self.flush()

    def flush(self) -> None:
        self.save()
        self.last, self.dirty = time.monotonic(), False


def restore_resolution(
    remote: Remote,
    scratch: WorkStore,
    fp: str,
    *,
    should_stop: Callable[[], bool] = lambda: False,
    workers: int | None = None,
    checkpoint_seconds: float = CHECKPOINT_SECONDS,
    progress: Progress = NULL,
) -> ResolutionIndex:
    """The resolution index of every part on the bucket, reading only parts not seen before.

    The snapshot is uploaded every ``checkpoint_seconds`` and at the end, so a job stopped
    midway still moved the next one forward. A stop request ends the catch-up early; the index
    then holds fewer verdicts, which only delays publications."""
    index = _restore_snapshot(remote, scratch, fp, progress)
    new = _new_part_keys(remote, index, fp)
    progress.event("parts_listed", seen=len(index.parts()), new=len(new))
    t0, rows, done = time.monotonic(), 0, 0
    batches = [new[i : i + BATCH] for i in range(0, len(new), BATCH)]
    checkpoint = _Checkpointer(
        lambda: _save(remote, scratch, index, fp, progress), checkpoint_seconds
    )
    try:
        with (
            ProcessPoolExecutor(
                workers or min(16, os.cpu_count() or 1),
                mp_context=multiprocessing.get_context("spawn"),
            ) as procs,
            ThreadPoolExecutor(1) as io,
            closing(
                _prefetched(batches, lambda ks: _download(remote, scratch, fp, ks), io, should_stop)
            ) as downloaded,
        ):
            for keys, paths in downloaded:
                rows += _index_batch(index, keys, paths, procs)
                done += len(keys)
                progress.tick(
                    "parts_read",
                    read=done,
                    total=len(new),
                    rows_per_s=round(rows / max(time.monotonic() - t0, 1e-9)),
                )
                checkpoint.batch_indexed()
    finally:
        if checkpoint.dirty or not _on_bucket(remote, fp):
            checkpoint.flush()
    progress.event("resolution_ready", parts_read=done, rows=rows)
    return index


def _on_bucket(remote: Remote, fp: str) -> bool:
    return snapshot_path(fp) in remote.ls("index/")


def _index_batch(
    index: ResolutionIndex, keys: Iterable[str], paths: list[Path], procs: ProcessPoolExecutor
) -> int:
    """Index a batch; the number of verdict rows it added."""
    rows = 0
    for key, path, verdicts in zip(
        keys, paths, procs.map(read_verdicts, map(str, paths), chunksize=8), strict=True
    ):
        if verdicts is not None:  # a corrupt part is skipped, as the canonical reader does
            index.add_part(key, verdicts)
            rows += len(verdicts)
        path.unlink(missing_ok=True)
    return rows


def fetch_parts(remote: Remote, scratch: WorkStore, fp: str, keys: Iterable[str]) -> None:
    """Download the given parts (by key) that are not on the scratch yet."""
    wanted = [part_path(fp, k) for k in sorted(set(keys))]
    missing = [p for p in wanted if not scratch.exists(p)]
    if missing:
        remote.get([(p, scratch.path(p)) for p in missing])
