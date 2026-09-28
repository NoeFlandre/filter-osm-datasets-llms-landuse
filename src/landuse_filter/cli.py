"""``luf``: plan, run, gate, orchestrate and publish the land-use labelling.

Exit codes: 0 ok, 1 failure, 2 usage error, 3 gate failed, 4 nothing to do,
5 GPU not eligible.
"""

import hashlib
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from landuse_filter import __version__

if TYPE_CHECKING:
    from landuse_filter.adapters.store import WorkStore
    from landuse_filter.application.controller import Controller, Settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
bench_app = typer.Typer(no_args_is_help=True, help="Benchmark parity: plan, budget, compare, gate.")
g5k_app = typer.Typer(no_args_is_help=True, help="Grid'5000: inventory, controller, jobs, storage.")
node_app = typer.Typer(no_args_is_help=True, help="Commands run on a reserved GPU node.")
app.add_typer(bench_app, name="bench")
app.add_typer(g5k_app, name="g5k")
app.add_typer(node_app, name="node")

PROMPT = Path(__file__).resolve().parents[2] / "data" / "prompt.txt"
WORK = typer.Option(Path(os.environ.get("LUF_WORK", "work")), "--work", help="Local work tree.")
JSON_OUT = typer.Option(False, "--json", help="Machine-readable output.")


def _store(work: Path) -> "WorkStore":
    from landuse_filter.adapters.store import WorkStore

    return WorkStore(work)


def _template() -> str:
    from landuse_filter.domain.prompting import check_prompt_digest

    data = PROMPT.read_bytes()
    check_prompt_digest(hashlib.sha256(data).hexdigest())
    return data.decode("utf-8")


def _emit(payload: object, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, indent=1, default=str))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            typer.echo(f"{key}: {value}")
    else:
        typer.echo(str(payload))


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def fingerprint(as_json: bool = JSON_OUT) -> None:
    """Show the production generation fingerprint and serving config."""
    from landuse_filter import config

    _emit(
        {"config_fingerprint": config.GENERATION_FP, "config": config.reference_config()}, as_json
    )


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


# --- benchmark ------------------------------------------------------------------


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
    by_sha = decisions_by_sha(store, config.GENERATION_FP)
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


# --- node -----------------------------------------------------------------------


@node_app.command("run")
def node_run(
    assignment: str = typer.Option(..., help="Assignment id in $LUF_WORK/assignments."),
) -> None:
    """Run an assignment on this node's GPU (called by scripts/node_job.sh)."""
    from landuse_filter.application.node_main import run

    raise typer.Exit(run(assignment))


# --- grid'5000 ------------------------------------------------------------------

SITES = "grenoble,lille,lyon,nancy,rennes,sophia,toulouse,luxembourg"


def _controller(work: Path, settings: "Settings") -> "Controller":
    from landuse_filter.application.controller import Controller

    return Controller(_store(work), settings, log=lambda m: typer.echo(m, err=True))


@g5k_app.command("inventory")
def g5k_inventory(
    work: Path = WORK, site: str = typer.Option("nancy", help="Frontend to query from.")
) -> None:
    """Refresh the GPU cluster inventory of all sites (Reference API)."""
    from landuse_filter.adapters import g5k
    from landuse_filter.application.controller import admission, eligible, load_clusters

    script = Path(__file__).parent / "adapters" / "remote" / "inventory.py"
    store = _store(work)
    store.write_json("inventory.json", g5k.inventory(site, script))
    for c in load_clusters(store):
        if eligible(c):
            typer.echo(
                f"{c.site:10} {c.name:12} {c.gpu:32} {c.gpus_per_node}x{c.nodes:<3} "
                f"{'abaca' if c.production else 'default'}{' exotic' if c.exotic else ''}  "
                f"{admission(store, c.gpu).value}"
            )


@g5k_app.command("run")
def g5k_run(
    datasets: str = typer.Option(
        ..., help="Comma-separated, in priority order (benchmark first for parity)."
    ),
    work: Path = WORK,
    sites: str = typer.Option(SITES),
    gpu_models: str = typer.Option(
        "", help="Comma-separated gpu keys to allow (default: admitted models)."
    ),
    max_jobs: int = typer.Option(12),
    max_jobs_per_site: int = typer.Option(4),
    walltime_minutes: int = typer.Option(60),
    besteffort: bool = typer.Option(False),
    interval: int = typer.Option(300, help="Seconds between cycles."),
    once: bool = typer.Option(False, help="Run a single cycle and exit."),
) -> None:
    """The controller loop: reconcile, pull results, submit where GPUs are free now."""
    from landuse_filter.application.controller import Settings

    settings = Settings(
        datasets=datasets.split(","),
        sites=sites.split(","),
        max_jobs_total=max_jobs,
        max_jobs_per_site=max_jobs_per_site,
        walltime=timedelta(minutes=walltime_minutes),
        besteffort=besteffort,
        gpu_models=[g for g in gpu_models.split(",") if g],
    )
    ctl = _controller(work, settings)
    while True:
        if ctl.store.exists("PAUSED"):
            ctl.settings.paused = True
        report = ctl.cycle()
        typer.echo(json.dumps(report))
        if once or (report["pending_chunks"] == 0 and report["live_jobs"] == 0):
            break
        time.sleep(interval)


@g5k_app.command("pause")
def g5k_pause(
    work: Path = WORK, cancel: bool = typer.Option(False, help="Also oardel our live jobs.")
) -> None:
    """Stop submitting new jobs (running jobs drain unless --cancel)."""
    store = _store(work)
    store.path("PAUSED").parent.mkdir(parents=True, exist_ok=True)
    store.path("PAUSED").write_text("paused\n")
    if cancel:
        from landuse_filter.application.controller import Settings

        sites = [str(s) for s in SITES.split(",")]
        ctl = _controller(work, Settings(datasets=[], sites=sites))
        typer.echo("\n".join(ctl.cancel_all()) or "no live jobs")


@g5k_app.command("resume")
def g5k_resume(work: Path = WORK) -> None:
    """Allow the controller to submit again."""
    _store(work).path("PAUSED").unlink(missing_ok=True)


@g5k_app.command("storage")
def g5k_storage(sites: str = typer.Option(SITES)) -> None:
    """Home quota usage per site and size of the project's ~/luf tree."""
    from landuse_filter.adapters import g5k

    for site in sites.split(","):
        try:
            typer.echo(f"{site}: {g5k.home_usage(site).strip()}")
        except g5k.RemoteError as exc:
            typer.echo(f"{site}: unreachable ({exc})", err=True)


store_app = typer.Typer(no_args_is_help=True, help="Mirror the work tree to a private HF Bucket.")
app.add_typer(store_app, name="store")
BUCKET = typer.Option("NoeFlandre/landuse-filter-work", help="Private Hugging Face Bucket id.")


@store_app.command("push")
def store_push(work: Path = WORK, bucket: str = BUCKET) -> None:
    """Upload the local work tree (plans, chunks, parts, ledger, gates) to the bucket."""
    from landuse_filter.adapters.store import Bucket

    target = Bucket(bucket)
    target.ensure()
    target.push(_store(work))
    typer.echo(f"pushed {work} -> hf://buckets/{bucket}")


@store_app.command("pull")
def store_pull(work: Path = WORK, bucket: str = BUCKET) -> None:
    """Restore the work tree from the bucket (e.g. on a new controller machine)."""
    from landuse_filter.adapters.store import Bucket

    Bucket(bucket).pull(_store(work))
    typer.echo(f"pulled hf://buckets/{bucket} -> {work}")


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


def main() -> None:  # pragma: no cover
    sys.exit(app())
