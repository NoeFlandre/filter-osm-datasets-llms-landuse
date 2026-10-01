"""``luf node run``: executed on a reserved GPU node by scripts/node_job.sh."""

import asyncio
import os
import signal
import socket
import time
from typing import TYPE_CHECKING

from landuse_filter import config
from landuse_filter.adapters.store import WorkStore

if TYPE_CHECKING:
    from landuse_filter.adapters.remote import BucketRemote
from landuse_filter.application.node import Runner
from landuse_filter.application.node_timeline import Timeline, timeline_record
from landuse_filter.domain.gpu import GpuSpec, gpu_key, ineligibility


class Stop:
    def __init__(self) -> None:
        self.requested = False
        for sig in (signal.SIGTERM, signal.SIGUSR2, signal.SIGINT):
            signal.signal(sig, self._handle)

    def _handle(self, signum: int, _frame: object) -> None:
        self.requested = True
        print(f"luf: signal {signum}: stopping after flush", flush=True)  # noqa: T201

    def __call__(self) -> bool:
        return self.requested


def detect_gpu() -> GpuSpec:
    from landuse_filter.adapters.engine import gpu_names

    name, memory, cap = (x.strip() for x in gpu_names()[0].split(","))
    major, _, minor = cap.partition(".")
    return GpuSpec(name, (int(major), int(minor or 0)), int(float(memory)))


def workspace(assignment: dict) -> tuple[WorkStore, "BucketRemote"]:
    """Node-local scratch (chunks, parts) and the bucket they come from and go to.

    Everything bulky stays on node-local scratch and is uploaded as it is produced;
    the NFS spool only carries the assignment and the job summary.
    """
    bucket = assignment.get("bucket")
    if not bucket:
        raise ValueError("assignment has no bucket: spool mode was removed (ADR-0009)")
    from landuse_filter.adapters.remote import BucketRemote
    from landuse_filter.application.sync import fetch, fetch_manifests

    scratch = WorkStore(config.scratch_dir())
    remote = BucketRemote(bucket)
    fetch(remote, scratch, [f"chunks/{c}.parquet" for c in assignment["chunks"]])
    for chunk in assignment["chunks"]:
        fetch_manifests(remote, scratch, f"parts/{assignment['fp']}/{chunk}/")
    return scratch, remote


def run(assignment_id: str) -> int:
    from landuse_filter.adapters.engine import SGLangEngine

    timeline = Timeline()
    spool = WorkStore(config.work_dir())
    assignment = spool.read_json(f"assignments/{assignment_id}.json")
    store, remote = workspace(assignment)
    spec = detect_gpu()
    reason = ineligibility(spec)
    if reason:
        print(f"luf: GPU {spec.model} not eligible: {reason}", flush=True)  # noqa: T201
        return 5
    stop = Stop()
    t0 = time.monotonic()
    engine = SGLangEngine(assignment["engine_kwargs"], assignment["sampling"])
    engine_ready = time.monotonic()
    load_seconds = engine_ready - t0
    provenance = {
        **assignment["provenance"],
        "sglang_version": engine.version,
        "gpu": spec.model,
        "site": assignment["site"],
        "oar_job_id": os.environ.get("OAR_JOB_ID", "local"),
        "assignment_id": assignment_id,
    }
    from landuse_filter.application.sync import prune_local_part, upload_part

    def on_part(chunk: str, part: str, shas: list[str]) -> None:
        upload_part(remote, store, assignment["fp"], chunk, part_id=part, shas=shas)
        prune_local_part(store, assignment["fp"], chunk, part)

    runner = Runner(
        store,
        engine,
        assignment["fp"],
        provenance,
        window=assignment["window"],
        should_stop=stop,
        on_part=on_part,
    )
    try:
        stats = asyncio.run(runner.run(assignment["chunks"]))
    finally:
        engine.shutdown()
    summary_path = f"jobs/{assignment['site']}/{provenance['oar_job_id']}.json"
    spool.write_json(
        summary_path,
        {
            "assignment_id": assignment_id,
            "host": socket.gethostname(),
            "gpu": spec.model,
            "gpu_key": gpu_key(spec.model),
            "load_seconds": round(load_seconds, 1),
            "completed": stats.completed,
            "generated_tokens": stats.generated_tokens,
            "failed": stats.failed,
            "sentences_per_second": round(stats.sentences_per_second, 3),
            "chunks_done": stats.chunks_done,
            "stopped": stop.requested,
            **timeline_record(
                timeline,
                engine_ready=engine_ready,
                first_result=stats.first_result,
                last_result=stats.last_result,
                walltime_s=int(assignment.get("walltime_s") or 0),
                assigned_texts=sum(store.read_chunk(c).num_rows for c in assignment["chunks"]),
                environ=os.environ,
            ),
        },
    )
    remote.put([(spool.path(summary_path), summary_path)])
    print(f"luf: done {stats.completed} sentences, {stats.sentences_per_second:.2f}/s", flush=True)  # noqa: T201
    return 0
