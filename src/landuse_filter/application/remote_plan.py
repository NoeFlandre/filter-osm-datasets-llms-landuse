"""Planning as a Grid'5000 job: scan on node-local scratch, publish chunks to the bucket.

The laptop never downloads input shards. The planner index (SQLite) is checkpointed to
the bucket, so a killed planning job resumes from the last checkpoint; chunk files and
plan lines are uploaded as they are emitted and deleted locally.
"""

import shutil
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from landuse_filter.adapters.remote import Remote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.plan import Encode, Planner

CHECKPOINT_SECONDS = 900.0


def _index(dataset: str) -> str:
    return f"index/{dataset}.sqlite"


def restore_index(remote: Remote, scratch: WorkStore, dataset: str) -> bool:
    """Download the index checkpoint as a fresh, writable file.

    The bucket download can be read-only (linked from a cache), which made every
    resumed planning job fail with "attempt to write a readonly database"
    (regression: Grenoble job 3123094).
    """
    if _index(dataset) not in remote.ls("index/"):
        return False
    target = scratch.path(_index(dataset))
    staged = target.with_name(target.name + ".download")
    remote.get([(_index(dataset), staged)])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(staged, target)
    target.chmod(0o644)
    staged.unlink()
    return True


def restore_plan(remote: Remote, scratch: WorkStore, dataset: str, fp: str) -> None:
    """Start from the bucket's plan lines: the file is append-only across nodes.

    A resumed job used to begin with an empty scratch file and overwrite the bucket's,
    losing every earlier chunk line (wiki: 1,586 of 26,389 survived).
    """
    plan = f"plans/{dataset}/{fp}/chunks.jsonl"
    if plan in remote.ls(f"plans/{dataset}/{fp}/"):
        remote.get([(plan, scratch.path(plan))])
        scratch.path(plan).chmod(0o644)


def publish_new_chunks(remote: Remote, scratch: WorkStore, dataset: str, fp: str) -> int:
    """Upload emitted chunk files and the plan lines; drop local chunk copies."""
    chunks = (
        sorted(scratch.path("chunks").glob("*.parquet")) if scratch.path("chunks").exists() else []
    )
    remote.put([(p, f"chunks/{p.name}") for p in chunks])
    plan = f"plans/{dataset}/{fp}/chunks.jsonl"
    if scratch.exists(plan):
        remote.put([(scratch.path(plan), plan)])
    for p in chunks:
        p.unlink()
    return len(chunks)


def checkpoint(remote: Remote, planner: Planner, scratch: WorkStore, dataset: str) -> None:
    planner.db.commit()
    remote.put([(scratch.path(_index(dataset)), _index(dataset))])


def run_plan(  # noqa: PLR0913 - one use case, explicit collaborators
    remote: Remote,
    scratch: WorkStore,
    dataset: str,
    fp: str,
    *,
    files: list[str],
    fetch: Callable[[str], Path],
    encode: Encode,
    template: str,
    chunk_size: int,
    should_stop: Callable[[], bool],
    forget: Callable[[Path], None] = lambda _p: None,
) -> dict:
    """Scan every file (resumably), emitting and publishing chunks as it goes."""
    restore_index(remote, scratch, dataset)
    restore_plan(remote, scratch, dataset, fp)
    planner = Planner(scratch, dataset, fp)
    planner.register(files)
    planner.recover_plan_lines()
    last = time.monotonic()
    fetched: list[Path] = []

    def fetch_and_track(path: str) -> Path:
        fetched.append(fetch(path))
        return fetched[-1]

    while not should_stop():
        if planner.scan(fetch_and_track, limit=1) == 0:
            break
        while fetched:
            forget(fetched.pop())
        planner.emit(encode, template, chunk_size, final=False)
        publish_new_chunks(remote, scratch, dataset, fp)
        if time.monotonic() - last > CHECKPOINT_SECONDS:
            checkpoint(remote, planner, scratch, dataset)
            last = time.monotonic()
    done = planner.report()
    if done.files_done == done.files_total:
        planner.emit(encode, template, chunk_size, final=True)
        publish_new_chunks(remote, scratch, dataset, fp)
    checkpoint(remote, planner, scratch, dataset)
    report = asdict(planner.report())
    scratch.write_json(f"plans/{dataset}/{fp}/report.json", report)
    remote.put(
        [(scratch.path(f"plans/{dataset}/{fp}/report.json"), f"plans/{dataset}/{fp}/report.json")]
    )
    return report
