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
from landuse_filter.adapters.settings_file import load as load_settings

OPS = load_settings()  # luf.toml (or $LUF_CONFIG): sites, bucket, walltimes, caps

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


# --- node -----------------------------------------------------------------------


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
    windows: str = typer.Option("16,32,64,128,256"),
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
    prompts = scratch.read_chunk(chunk).column("input_ids").to_pylist()
    spec = detect_gpu()
    cfg = config.reference_config()
    engine = SGLangEngine(
        config.engine_kwargs(cfg, {"max_running_requests": max(levels)}), cfg["sampling"]
    )
    try:
        points = sweep(engine, prompts, levels)
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


# --- grid'5000 ------------------------------------------------------------------

SITES = ",".join(OPS.sites)


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
    max_jobs: int = typer.Option(OPS.max_jobs),
    max_jobs_per_site: int = typer.Option(OPS.max_jobs_per_site),
    walltime_minutes: int = typer.Option(OPS.walltime_minutes),
    besteffort: bool = typer.Option(False),
    window: int | None = typer.Option(
        None, help="Candidate concurrency (tuning); default: GPU profile."
    ),
    namespace: str | None = typer.Option(
        None, help="Store a candidate config's results under <fp>-<namespace>."
    ),
    bucket: str | None = typer.Option(
        None, help="Private HF Bucket for chunks and parts (keeps local disks empty)."
    ),
    interval: int = typer.Option(OPS.interval_seconds, help="Seconds between cycles."),
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
        window=window,
        namespace=namespace,
        bucket=bucket,
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


@g5k_app.command("cpu-job")
def g5k_cpu_job(
    mode: str = typer.Argument(..., help="plan or publish"),
    site: str = typer.Option(..., help="Site to run the CPU job on."),
    dataset: str = typer.Option(...),
    revision: str = typer.Option(...),
    walltime_minutes: int = typer.Option(60),
) -> None:
    """Submit one resumable planning or publishing job (default queue, one CPU node)."""
    if mode not in ("plan", "publish"):
        raise typer.BadParameter("mode must be plan or publish")
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from landuse_filter.adapters import g5k
    from landuse_filter.application.controller import commit, git_archive
    from landuse_filter.domain.capacity import walltime_text
    from landuse_filter.domain.policy import allowed_window

    window = allowed_window(datetime.now(ZoneInfo("Europe/Paris")), starts_now=True)
    wall = (
        min(timedelta(minutes=walltime_minutes), window.max_walltime)
        if window
        else timedelta(hours=1)
    )
    code_commit = commit()
    code = g5k.deploy_code(site, code_commit, git_archive(code_commit))
    g5k.ssh(site, f"mkdir -p {g5k.REMOTE_ROOT}/logs")
    g5k.policy_check(site)
    args = (
        ["-q", "default"]
        + (["-t", window.job_type] if window and window.job_type else [])
        + [
            "-p",
            "gpu_count = 0",
            "-l",
            f"host=1,walltime={walltime_text(wall)}",
            "--checkpoint",
            "300",
            "-n",
            f"{g5k.JOB_PREFIX}plan-{dataset[:20]}",
            "-O",
            f"{g5k.REMOTE_ROOT}/logs/%jobid%.out",
            "-E",
            f"{g5k.REMOTE_ROOT}/logs/%jobid%.err",
            f"{code}/scripts/node_job.sh {code} {mode} {dataset} {revision}",
        ]
    )
    job_id = g5k.submit(site, args)
    g5k.policy_check(site)
    typer.echo(f"{mode} job {job_id} on {site} ({walltime_text(wall)})")


@g5k_app.command("calibrate-job")
def g5k_calibrate_job(
    site: str = typer.Option(...),
    cluster: str = typer.Option(...),
    chunk: str = typer.Option(..., help="A benchmark chunk id (prompts for the sweep)."),
    work: Path = WORK,
) -> None:
    """Submit one GPU calibration job on ``cluster`` (1 h, starts now or is cancelled)."""
    from landuse_filter.adapters import g5k
    from landuse_filter.application.controller import commit, git_archive, load_clusters
    from landuse_filter.domain.capacity import oarsub_arguments

    target = next(c for c in load_clusters(_store(work)) if c.site == site and c.name == cluster)
    code_commit = commit()
    code = g5k.deploy_code(site, code_commit, git_archive(code_commit))
    g5k.ssh(site, f"mkdir -p {g5k.REMOTE_ROOT}/logs")
    g5k.policy_check(site)
    args = oarsub_arguments(
        target,
        timedelta(hours=1),
        None,
        f"{g5k.JOB_PREFIX}calib-{cluster}",
        command=f"{code}/scripts/node_job.sh {code} calibrate {chunk}",
    )
    job_id = g5k.submit(site, args)
    g5k.policy_check(site)
    typer.echo(f"calibration job {job_id} on {site}/{cluster}")


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
BUCKET = typer.Option(OPS.bucket, help="Private Hugging Face Bucket id.")


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


@g5k_app.command("clean")
def g5k_clean(
    sites: str = typer.Option(SITES),
    work: Path = WORK,
    apply: bool = typer.Option(False, help="Actually delete (default: dry run)."),
) -> None:
    """Delete stale project files under ~/luf on each site (old code, envs, logs, synced parts)."""
    from landuse_filter.adapters import g5k
    from landuse_filter.application.controller import commit
    from landuse_filter.domain.cleanup import Entry, Keep, cleanup_plan

    store = _store(work)
    assignments = [
        store.read_json(f"assignments/{p.name}") for p in store.path("assignments").glob("*.json")
    ]
    live = [a for a in assignments if a.get("state") in ("submitting", "submitted")]
    commits = {a["provenance"]["code_commit"] for a in live} | {commit()}
    lock = hashlib.sha256(
        (Path(__file__).resolve().parents[2] / "uv.lock").read_bytes()
    ).hexdigest()[:12]
    for site in sites.split(","):
        try:
            listing = g5k.project_listing(site)
        except g5k.RemoteError as exc:
            typer.echo(f"{site}: unreachable ({exc})", err=True)
            continue
        synced = frozenset(
            p.removeprefix("luf/work/parts/")
            for p, _ in listing
            if p.startswith("luf/work/parts/")
            and (
                store.exists(p.removeprefix("luf/work/"))
                or store.exists(p.removeprefix("luf/work/").replace(".parquet", ".json"))
            )
        )
        plan = cleanup_plan(
            [Entry(p, a) for p, a in listing], Keep(frozenset(commits), lock, synced)
        )
        typer.echo(f"{site}: {len(plan)} path(s) {'deleted' if apply else 'would be deleted'}")
        for p in plan[:20]:
            typer.echo(f"  {p}")
        if apply and plan:
            g5k.remove(site, plan)


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
