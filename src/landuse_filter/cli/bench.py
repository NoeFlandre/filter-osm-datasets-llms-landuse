"""`luf bench` commands."""

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from landuse_filter.cli import (
    JSON_OUT,
    WORK,
    _emit,
    _store,
    _template,
    bench_app,
)

if TYPE_CHECKING:
    pass


def _bench_root(work: Path) -> Path:
    from landuse_filter.adapters.benchmark import download

    return download(work / "benchmark")


@bench_app.command("plan")
def bench_plan(work: Path = WORK, chunk_size: int = typer.Option(500, min=1)) -> None:
    """Turn the 25,500-item benchmark into work chunks (production path)."""
    from landuse_filter import config
    from landuse_filter.adapters.benchmark import read_items
    from landuse_filter.adapters.tokenizer import chat_encoder
    from landuse_filter.application.bench_plan import plan_benchmark

    items = read_items(_bench_root(work))
    encode = chat_encoder(config.MODEL_ID, config.MODEL_REVISION)
    n = plan_benchmark(
        _store(work),
        items,
        encode,
        template=_template(),
        fp=config.GENERATION_FP,
        chunk_size=chunk_size,
    )
    typer.echo(f"{len(items)} items -> {n} chunks")


@bench_app.command("admit")
def bench_admit(
    gpu: str = typer.Option(..., help="GPU key, e.g. l40s (see `luf g5k inventory`)."),
    work: Path = WORK,
    resamples: int = typer.Option(10_000),
    as_json: bool = JSON_OUT,
) -> None:
    """One-time admission of a GPU type: its full-benchmark run (namespace gpu-<key>)
    must pass the same non-inferiority gate as parity."""
    from dataclasses import asdict

    from landuse_filter import config
    from landuse_filter.adapters.benchmark import read_reference
    from landuse_filter.application.bench import compare
    from landuse_filter.application.results import decisions_by_sha

    store = _store(work)
    item_sha = store.read_json(f"plans/benchmark/{config.GENERATION_FP}/items.json")
    by_sha = decisions_by_sha(store, f"{config.GENERATION_FP}-gpu-{gpu}")
    candidate = {i: by_sha[s] for i, s in item_sha.items() if s in by_sha}
    if len(candidate) < len(item_sha):
        typer.echo(f"incomplete: {len(candidate)}/{len(item_sha)}", err=True)
        raise typer.Exit(4)
    reference = [r for r in read_reference(_bench_root(work)) if r.item_id in item_sha]
    gate = compare(reference, candidate, resamples)
    status = "admitted" if gate.passed else "rejected"
    store.write_json(f"gates/admission/{gpu}.json", {"status": status, "gate": asdict(gate)})
    _emit({"gpu": gpu, "status": status, **asdict(gate)}, as_json)
    if not gate.passed:
        raise typer.Exit(3)


@bench_app.command("budget")
def bench_budget(
    work: Path = WORK,
    caps: str = typer.Option("1024,1536,2048,2560,3072,3584,3840,4000"),
    resamples: int = typer.Option(10_000),
    as_json: bool = JSON_OUT,
) -> None:
    """Offline max_new_tokens simulation on the reference run, through the full gate."""
    from dataclasses import asdict

    from landuse_filter.adapters.benchmark import read_reference
    from landuse_filter.application.bench import simulate_budgets

    reference = list(read_reference(_bench_root(work)))
    rows = simulate_budgets(reference, [int(c) for c in caps.split(",")], resamples)
    payload = [{"cap": r.cap, "truncation_rate": r.truncation_rate, **asdict(r.gate)} for r in rows]
    _emit(payload, as_json)


@bench_app.command("compare")
def bench_compare(
    work: Path = WORK,
    resamples: int = typer.Option(10_000),
    label: str = typer.Option("reference-config", help="Name of the gate file to write."),
    namespace: str | None = typer.Option(
        None, help="Gate a candidate namespace instead of production."
    ),
    as_json: bool = JSON_OUT,
) -> None:
    """Gate our benchmark generations against the published reference run."""
    from dataclasses import asdict

    from landuse_filter import config
    from landuse_filter.adapters.benchmark import read_reference
    from landuse_filter.application.bench import compare, macro_scores
    from landuse_filter.application.results import decisions_by_sha

    store = _store(work)
    reference = list(read_reference(_bench_root(work)))
    item_sha = store.read_json(f"plans/benchmark/{config.GENERATION_FP}/items.json")
    fp = f"{config.GENERATION_FP}-{namespace}" if namespace else config.GENERATION_FP
    by_sha = decisions_by_sha(store, fp)
    candidate = {item: by_sha[sha] for item, sha in item_sha.items() if sha in by_sha}
    if len(candidate) < len(item_sha):
        typer.echo(f"incomplete: {len(candidate)}/{len(item_sha)} items generated", err=True)
        raise typer.Exit(4)
    gate = compare(reference, candidate, resamples)
    from landuse_filter.adapters.benchmark import ReferencePrediction

    ours = [
        ReferencePrediction(r.item_id, r.language, r.expected, candidate[r.item_id], 0, False, "")
        for r in reference
    ]
    payload = {
        "gate": asdict(gate),
        "ours": macro_scores(ours),
        "reference": macro_scores(reference),
    }
    store.write_json(f"gates/{label}.json", payload)
    _emit(payload, as_json)
    if not gate.passed:
        raise typer.Exit(3)
