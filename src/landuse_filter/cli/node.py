"""`luf node` commands."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from landuse_filter.cli import (
    OPS,
    _store,
    _template,
    node_app,
)

if TYPE_CHECKING:
    pass


@node_app.command("run")
def node_run(
    assignment: str = typer.Option(..., help="Assignment id in $LUF_WORK/assignments."),
) -> None:
    """Run an assignment on this node's GPU (called by scripts/node_job.sh)."""
    from landuse_filter.application.node_main import run

    raise typer.Exit(run(assignment))


@node_app.command("plan")
def node_plan(
    dataset: str = typer.Option(...),
    revision: str = typer.Option(...),
    bucket: str = typer.Option(OPS.bucket),
    chunk_size: int = typer.Option(2000, min=1),
) -> None:
    """Scan a dataset on this node's scratch and publish chunks to the bucket."""
    import signal

    from landuse_filter import config
    from landuse_filter.adapters.readers import SOURCES
    from landuse_filter.adapters.remote import BucketRemote
    from landuse_filter.adapters.tokenizer import chat_encoder
    from landuse_filter.application.plan import download, list_input_files
    from landuse_filter.application.remote_plan import run_plan

    stop = {"requested": False}
    for sig in (signal.SIGTERM, signal.SIGUSR2):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    source = SOURCES[dataset]
    scratch = _store(Path(os.environ.get("LUF_SCRATCH", "/tmp/luf-scratch")))  # noqa: S108
    report = run_plan(
        BucketRemote(bucket),
        scratch,
        dataset,
        config.GENERATION_FP,
        files=list_input_files(source, revision),
        fetch=lambda path: download(source, path, revision),
        encode=chat_encoder(config.MODEL_ID, config.MODEL_REVISION),
        template=_template(),
        chunk_size=chunk_size,
        should_stop=lambda: stop["requested"],
        forget=lambda p: p.resolve().unlink(missing_ok=True),
    )
    typer.echo(json.dumps(report))


@node_app.command("calibrate")
def node_calibrate(
    chunk: str = typer.Option(..., help="Chunk id whose prompts drive the sweep."),
    windows: str = typer.Option("16,32,64,128"),
    max_prompts: int = typer.Option(300, help="Prompts per level (keeps a sweep well inside 1 h)."),
    bucket: str = typer.Option(OPS.bucket),
) -> None:
    """Sweep concurrency on this GPU; write a candidate profile (speed only)."""
    from dataclasses import asdict

    from landuse_filter import config
    from landuse_filter.adapters.engine import SGLangEngine
    from landuse_filter.adapters.remote import BucketRemote
    from landuse_filter.application.calibrate import best, sweep
    from landuse_filter.application.node_main import detect_gpu
    from landuse_filter.application.sync import fetch
    from landuse_filter.domain.gpu import gpu_key

    levels = [int(w) for w in windows.split(",")]
    scratch = _store(Path(os.environ.get("LUF_SCRATCH", "/tmp/luf-scratch")))  # noqa: S108
    remote = BucketRemote(bucket)
    fetch(remote, scratch, [f"chunks/{chunk}.parquet"])
    prompts = scratch.read_chunk(chunk).column("input_ids").to_pylist()[:max_prompts]
    spec = detect_gpu()
    cfg = config.reference_config()
    engine = SGLangEngine(
        config.engine_kwargs(cfg, {"max_running_requests": max(levels)}), cfg["sampling"]
    )
    try:
        points = sweep(
            engine,
            prompts,
            levels,
            on_point=lambda p: typer.echo(
                json.dumps({**asdict(p), "sps": round(p.sentences_per_second, 3)})
            ),
        )
    finally:
        engine.shutdown()
    pick = best(points)
    key = gpu_key(spec.model)
    profile = {
        "gpu": key,
        "max_running_requests": pick.window,
        "mem_fraction_static": 0.75,
        "sentences_per_second": round(pick.sentences_per_second, 3),
        "calibrated": True,
    }
    path = f"profiles/{key}.json"
    scratch.write_json(path, profile)
    scratch.write_json(
        f"calibration/{key}.json", {"points": [asdict(p) for p in points], "gpu": spec.model}
    )
    remote.put(
        [
            (scratch.path(path), path),
            (scratch.path(f"calibration/{key}.json"), f"calibration/{key}.json"),
        ]
    )
    typer.echo(json.dumps(profile))


@node_app.command("publish")
def node_publish(
    dataset: str = typer.Option(...),
    revision: str = typer.Option(...),
    bucket: str = typer.Option(OPS.bucket),
) -> None:
    """Build and upload the -landuse dataset on this node's scratch."""
    from dataclasses import asdict

    from landuse_filter import config
    from landuse_filter.adapters.remote import BucketRemote
    from landuse_filter.application.remote_publish import run_publish

    scratch = _store(Path(os.environ.get("LUF_SCRATCH", "/tmp/luf-scratch")))  # noqa: S108
    report = run_publish(BucketRemote(bucket), scratch, dataset, revision, config.GENERATION_FP)
    typer.echo(json.dumps(asdict(report)))
