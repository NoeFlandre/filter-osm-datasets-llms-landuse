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
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from landuse_filter.adapters.indexes import ResolutionIndex
from landuse_filter.adapters.remote import Remote
from landuse_filter.adapters.store import WorkStore
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


def _restore_snapshot(remote: Remote, scratch: WorkStore, fp: str) -> ResolutionIndex:
    """The bucket's snapshot when it was made by the same parser, else an empty index."""
    target = scratch.path(snapshot_path(fp))
    target.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path(fp) in remote.ls("index/"):
        staged = target.with_name(target.name + ".download")
        remote.get([(snapshot_path(fp), staged)])
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


def _save(remote: Remote, scratch: WorkStore, index: ResolutionIndex, fp: str) -> None:
    index.db.commit()
    remote.put([(scratch.path(snapshot_path(fp)), snapshot_path(fp))])


def _download(remote: Remote, scratch: WorkStore, fp: str, keys: list[str]) -> list[Path]:
    paths = [part_path(fp, k) for k in keys]
    size = -(-len(paths) // DOWNLOAD_STREAMS)
    chunks = [paths[i : i + size] for i in range(0, len(paths), size)]
    with ThreadPoolExecutor(DOWNLOAD_STREAMS) as pool:
        list(pool.map(lambda c: remote.get([(p, scratch.path(p)) for p in c]), chunks))
    return [scratch.path(p) for p in paths]


def restore_resolution(
    remote: Remote,
    scratch: WorkStore,
    fp: str,
    *,
    should_stop: Callable[[], bool] = lambda: False,
    workers: int | None = None,
    checkpoint_seconds: float = CHECKPOINT_SECONDS,
) -> ResolutionIndex:
    """The resolution index of every part on the bucket, reading only parts not seen before.

    The snapshot is uploaded every ``checkpoint_seconds`` and at the end, so a job stopped
    midway still moved the next one forward. A stop request ends the catch-up early; the index
    then holds fewer verdicts, which only delays publications."""
    index = _restore_snapshot(remote, scratch, fp)
    seen = index.parts()
    new = sorted(
        k
        for k in {part_key(p) for p in remote.ls(f"parts/{fp}/") if p.endswith(".parquet")}
        if k not in seen
    )
    batches = [new[i : i + BATCH] for i in range(0, len(new), BATCH)]
    last, dirty = time.monotonic(), False
    try:
        with (
            ProcessPoolExecutor(
                workers or min(16, os.cpu_count() or 1),
                mp_context=multiprocessing.get_context("spawn"),
            ) as procs,
            ThreadPoolExecutor(1) as io,
        ):
            pending = io.submit(_download, remote, scratch, fp, batches[0]) if batches else None
            for i, keys in enumerate(batches):
                if should_stop():
                    break
                assert pending is not None  # noqa: S101 - a batch implies a download
                paths = pending.result()
                pending = (
                    io.submit(_download, remote, scratch, fp, batches[i + 1])
                    if i + 1 < len(batches)
                    else None
                )
                _index_batch(index, keys, paths, procs)
                dirty = True
                if time.monotonic() - last >= checkpoint_seconds:
                    _save(remote, scratch, index, fp)
                    last, dirty = time.monotonic(), False
            if pending is not None:
                pending.cancel()
    finally:
        if dirty or not _on_bucket(remote, fp):
            _save(remote, scratch, index, fp)
    return index


def _on_bucket(remote: Remote, fp: str) -> bool:
    return snapshot_path(fp) in remote.ls("index/")


def _index_batch(
    index: ResolutionIndex, keys: Iterable[str], paths: list[Path], procs: ProcessPoolExecutor
) -> None:
    for key, path, verdicts in zip(
        keys, paths, procs.map(read_verdicts, map(str, paths), chunksize=8), strict=True
    ):
        if verdicts is not None:  # a corrupt part is skipped, as the canonical reader does
            index.add_part(key, verdicts)
        path.unlink(missing_ok=True)


def fetch_parts(remote: Remote, scratch: WorkStore, fp: str, keys: Iterable[str]) -> None:
    """Download the given parts (by key) that are not on the scratch yet."""
    wanted = [part_path(fp, k) for k in sorted(set(keys))]
    missing = [p for p in wanted if not scratch.exists(p)]
    if missing:
        remote.get([(p, scratch.path(p)) for p in missing])
