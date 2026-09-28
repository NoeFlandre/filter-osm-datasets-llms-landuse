"""`luf plan / publish / status` commands."""

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from landuse_filter.cli import (
    JSON_OUT,
    WORK,
    _emit,
    _store,
    _template,
    app,
)

if TYPE_CHECKING:
    pass


@app.command()
def plan(
    dataset: str = typer.Option(..., help="Input dataset name, e.g. osm-polygon-description-tag."),
    revision: str = typer.Option(..., help="Pinned input dataset commit sha."),
    work: Path = WORK,
    chunk_size: int = typer.Option(2000, min=1),
    limit_files: int | None = typer.Option(None, help="Scan at most N more files this run."),
    final: bool = typer.Option(False, help="Also emit the last partial chunk."),
    scan_only: bool = typer.Option(False, help="Only scan (sizing); emit no chunks."),
    as_json: bool = JSON_OUT,
) -> None:
    """Scan input files (resumable) and emit work chunks."""
    from dataclasses import asdict

    from landuse_filter import config
    from landuse_filter.adapters.readers import SOURCES
    from landuse_filter.application.plan import Planner, download, list_input_files

    if dataset not in SOURCES:
        raise typer.BadParameter(f"unknown dataset; choose from {sorted(SOURCES)}")
    source = SOURCES[dataset]
    planner = Planner(_store(work), dataset, config.GENERATION_FP)
    planner.register(list_input_files(source, revision))
    planner.scan(lambda path: download(source, path, revision), limit=limit_files)
    if not scan_only:
        from landuse_filter.adapters.tokenizer import chat_encoder

        encode = chat_encoder(config.MODEL_ID, config.MODEL_REVISION)
        planner.emit(encode, _template(), chunk_size, final=final)
    _emit(asdict(planner.report()), as_json)


@app.command()
def publish(
    dataset: str = typer.Option(..., help="Input dataset name."),
    revision: str = typer.Option(..., help="Pinned input revision (same as planning)."),
    work: Path = WORK,
    dry_run: bool = typer.Option(False, help="Build labels locally; upload nothing."),
    as_json: bool = JSON_OUT,
) -> None:
    """Mirror the input and upload labels/generations for every fully generated file."""
    from dataclasses import asdict

    from landuse_filter.application.publish import publish as run_publish

    report = run_publish(_store(work), dataset, revision, dry_run=dry_run)
    _emit(asdict(report), as_json)
    if report.new_files == 0:
        raise typer.Exit(4)


@app.command()
def status(
    work: Path = WORK, datasets: str = typer.Option("benchmark"), as_json: bool = JSON_OUT
) -> None:
    """Progress per dataset: chunks complete / pending, live assignments, throughput."""
    from landuse_filter.application.status import summarize

    _emit(summarize(_store(work), datasets.split(",")), as_json)
