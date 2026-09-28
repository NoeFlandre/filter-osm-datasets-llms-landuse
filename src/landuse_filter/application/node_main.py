"""``luf node run``: executed on a reserved GPU node by scripts/node_job.sh."""

import asyncio
import os
import signal
import socket
import time
from pathlib import Path

from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.node import Runner
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


def run(assignment_id: str) -> int:
    from landuse_filter.adapters.engine import SGLangEngine

    store = WorkStore(Path(os.environ.get("LUF_WORK", "work")))
    assignment = store.read_json(f"assignments/{assignment_id}.json")
    spec = detect_gpu()
    reason = ineligibility(spec)
    if reason:
        print(f"luf: GPU {spec.model} not eligible: {reason}", flush=True)  # noqa: T201
        return 5
    stop = Stop()
    t0 = time.monotonic()
    engine = SGLangEngine(assignment["engine_kwargs"], assignment["sampling"])
    load_seconds = time.monotonic() - t0
    provenance = {
        **assignment["provenance"],
        "sglang_version": engine.version,
        "gpu": spec.model,
        "site": assignment["site"],
        "oar_job_id": os.environ.get("OAR_JOB_ID", "local"),
        "assignment_id": assignment_id,
    }
    runner = Runner(
        store, engine, assignment["fp"], provenance, window=assignment["window"], should_stop=stop
    )
    try:
        stats = asyncio.run(runner.run(assignment["chunks"]))
    finally:
        engine.shutdown()
    store.write_json(
        f"jobs/{assignment['site']}/{provenance['oar_job_id']}.json",
        {
            "assignment_id": assignment_id,
            "host": socket.gethostname(),
            "gpu": spec.model,
            "gpu_key": gpu_key(spec.model),
            "load_seconds": round(load_seconds, 1),
            "completed": stats.completed,
            "generated_tokens": stats.generated_tokens,
            "sentences_per_second": round(stats.sentences_per_second, 3),
            "chunks_done": stats.chunks_done,
            "stopped": stop.requested,
        },
    )
    print(f"luf: done {stats.completed} sentences, {stats.sentences_per_second:.2f}/s", flush=True)  # noqa: T201
    return 0
