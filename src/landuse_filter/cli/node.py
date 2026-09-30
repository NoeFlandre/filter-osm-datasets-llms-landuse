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


@node_app.command("replan")
def node_replan(
    dataset: str = typer.Option(...),
    revision: str = typer.Option(...),
    bucket: str = typer.Option(OPS.bucket),
    chunk_size: int = typer.Option(2000, min=1),
) -> None:
    """Plan every not-yet-generated text again, round-robin over H3 cells (ADR-0014)."""
    from landuse_filter import config
    from landuse_filter.adapters.hexmap import cell_of
    from landuse_filter.adapters.readers import SOURCES
    from landuse_filter.adapters.remote import BucketRemote
    from landuse_filter.adapters.tokenizer import chat_encoder
    from landuse_filter.application.locate import locator
    from landuse_filter.application.plan import download, list_input_files
    from landuse_filter.application.remote_plan import run_replan

    source = SOURCES[dataset]
    scratch = _store(Path(os.environ.get("LUF_SCRATCH", "/tmp/luf-scratch")))  # noqa: S108
    report = run_replan(
        BucketRemote(bucket),
        scratch,
        dataset,
        config.GENERATION_FP,
        files=list_input_files(source, revision),
        fetch=lambda path: download(source, path, revision),
        locate=locator(dataset, lambda path: download(source, path, revision), cell_of),
        encode=chat_encoder(config.MODEL_ID, config.MODEL_REVISION),
        template=_template(),
        chunk_size=chunk_size,
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


@node_app.command("repair")
def node_repair(
    dataset: str = typer.Option(...),
    bucket: str = typer.Option(OPS.bucket),
) -> None:
    """Remove duplicate rows from the published generations/ tables, then reset the card cache."""
    import pyarrow.parquet as pq

    from landuse_filter.adapters import hub
    from landuse_filter.adapters.remote import BucketRemote
    from landuse_filter.application.publish import output_repo
    from landuse_filter.application.remote_publish import reset_card_cache
    from landuse_filter.application.repair import dedupe_generations

    repo = output_repo(dataset)
    paths = sorted(p for p in hub.remote_files(repo) if p.startswith("generations/"))
    local = {p: hub.download_all(repo, "main", [p])[0][0] for p in paths}
    by_local = {path: repo_path for repo_path, path in local.items()}
    scratch = _store(Path(os.environ.get("LUF_SCRATCH", "/tmp/luf-scratch")))  # noqa: S108
    changed = dedupe_generations(list(local.values()))
    rewrites, deletions = [], []
    for path, table in changed.items():
        if table is None:
            deletions.append(by_local[path])
            continue
        out = scratch.path(f"repair/{path.name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
        rewrites.append((out, by_local[path]))
    if rewrites:
        hub.upload(repo, rewrites, "Remove duplicate generation rows")
    if deletions:
        hub.delete(repo, deletions, "Remove emptied generation files")
    reset_card_cache(BucketRemote(bucket), scratch, dataset)
    typer.echo(json.dumps({"rewritten": len(rewrites), "deleted": len(deletions)}))
