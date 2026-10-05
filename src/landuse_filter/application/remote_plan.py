"""Planning as a Grid'5000 job: scan on node-local scratch, publish chunks to the bucket.

The laptop never downloads input shards. The planner index (SQLite) is checkpointed to
the bucket, so a killed planning job resumes from the last checkpoint; chunk files and
plan lines are uploaded as they are emitted and deleted locally.
"""

import json
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from landuse_filter.adapters.remote import Remote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.plan import Encode, Planner
from landuse_filter.application.progress import NULL, Progress

CHECKPOINT_SECONDS = 900.0


def _index(dataset: str) -> str:
    return f"index/{dataset}.sqlite"


def restore_index(
    remote: Remote, scratch: WorkStore, dataset: str, progress: Progress = NULL
) -> bool:
    """Download the index checkpoint as a fresh, writable file.

    The bucket download can be read-only (linked from a cache), which made every
    resumed planning job fail with "attempt to write a readonly database"
    (regression: Grenoble job 3123094).
    """
    if _index(dataset) not in remote.ls("index/"):
        return False
    progress.event("restore_index_start", file=_index(dataset))
    t0 = time.monotonic()
    target = scratch.path(_index(dataset))
    staged = target.with_name(target.name + ".download")
    remote.get([(_index(dataset), staged)])
    progress.event(
        "restore_index_done", bytes=staged.stat().st_size, seconds=round(time.monotonic() - t0)
    )
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


@dataclass(frozen=True)
class PlanInputs:
    """What planning (and re-planning) reads and how it cuts chunks."""

    files: list[str]  # input files of the dataset, in planner order
    fetch: Callable[[str], Path]  # download one input file
    encode: Encode  # batch tokeniser of rendered prompts
    template: str
    chunk_size: int
    forget: Callable[[Path], None] = lambda _p: None  # drop a downloaded file once read


def run_plan(
    remote: Remote,
    scratch: WorkStore,
    dataset: str,
    fp: str,
    inputs: PlanInputs,
    *,
    should_stop: Callable[[], bool],
) -> dict:
    """Scan every file (resumably), emitting and publishing chunks as it goes."""
    files, fetch, forget = inputs.files, inputs.fetch, inputs.forget
    encode, template, chunk_size = inputs.encode, inputs.template, inputs.chunk_size
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


Locate = Callable[
    [str, str, Path], Iterable[tuple[str, str]]
]  # dataset, path, local -> (sha, cell)


def generated_shas(remote: Remote, scratch: WorkStore, fp: str) -> set[str]:
    """Texts that already have a generation, from the manifests of the bucket's parts."""
    paths = [p for p in remote.ls(f"parts/{fp}/") if p.endswith(".json")]
    shas: set[str] = set()
    for start in range(0, len(paths), 500):
        batch = paths[start : start + 500]
        remote.get([(p, scratch.path(p)) for p in batch])
        for p in batch:
            shas.update(json.loads(scratch.path(p).read_text(encoding="utf-8"))["text_sha256s"])
            scratch.path(p).unlink()
    return shas


def run_replan(
    remote: Remote,
    scratch: WorkStore,
    dataset: str,
    fp: str,
    inputs: PlanInputs,
    *,
    locate: Locate,
) -> dict:
    """Plan every not-yet-generated text again in a geographically uniform order.

    Finished chunks keep their plan line; the open ones leave the plan and their unfinished
    texts are re-chunked (ADR-0014). Needs a completed planner index.
    """
    if not restore_index(remote, scratch, dataset):
        raise FileNotFoundError(f"no planner index for {dataset} in the bucket")
    restore_plan(remote, scratch, dataset, fp)
    planner = Planner(scratch, dataset, fp)
    progress = planner.report()
    if progress.files_done != progress.files_total:
        raise RuntimeError(f"{dataset}: planning is not complete ({progress.files_done} files)")
    planner.mark_done(generated_shas(remote, scratch, fp))
    released = planner.release_unfinished()
    for path in inputs.files:
        local = inputs.fetch(path)
        planner.set_cells(locate(dataset, path, local))
        inputs.forget(local)
    pending = planner.db.execute(
        "SELECT COUNT(*), COUNT(cell) FROM texts WHERE chunk_id IS NULL AND done = 0"
    ).fetchone()
    chunks = planner.emit_uniform(inputs.encode, inputs.template, inputs.chunk_size)
    publish_new_chunks(remote, scratch, dataset, fp)
    checkpoint(remote, planner, scratch, dataset)
    return {
        "released_chunks": len(released),
        "replanned_texts": pending[0],
        "located_texts": pending[1],
        "new_chunks": chunks,
    }
