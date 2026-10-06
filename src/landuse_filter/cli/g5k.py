"""`luf g5k` commands."""

import hashlib
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from landuse_filter.cli import (
    OPS,
    SITES,
    WORK,
    _store,
    g5k_app,
)
from landuse_filter.domain.policy_check import PolicyCheck

if TYPE_CHECKING:
    from landuse_filter.adapters.store import WorkStore
    from landuse_filter.application.controller import Controller, Settings


def _controller(work: Path, settings: "Settings") -> "Controller":
    from landuse_filter.application.controller import Controller

    return Controller(_store(work), settings, log=lambda m: typer.echo(m, err=True))


RUN_MAX_QUEUED_PER_SITE = 1
ADMISSION_MAX_JOBS = 5
ADMISSION_MAX_JOBS_PER_SITE = 3
ADMISSION_MAX_QUEUED_PER_SITE = 2
STALE_BESTEFFORT_MINUTES = 20

WalltimeMinutes = Annotated[int, typer.Option()]
DayWalltimeMinutes = Annotated[
    int | None,
    typer.Option(
        help="Preferred (long) day walltime, retried with --walltime-minutes if it cannot "
        "start (default: --walltime-minutes).",
    ),
]
DayLongMaxFailures = Annotated[
    int,
    typer.Option(help="Consecutive failed long day attempts on a cluster before a 1 h pause."),
]
ChunkOverflow = Annotated[
    float,
    typer.Option(
        min=1.0,
        help="Work given to a job = expected capacity x this (>= 1.0); unfinished chunks "
        "return to the pool at the checkpoint signal.",
    ),
]
SubmitWorkers = Annotated[
    int,
    typer.Option(
        min=1,
        help="Sites submitted to in parallel (one worker per site; 1 = one job after "
        "another). See ADR-0020.",
    ),
]
PolicyCheckOption = Annotated[
    PolicyCheck,
    typer.Option(
        help="usagepolicycheck cadence: per-job (before and after every submission) or "
        "per-batch (once per site before its first submission of a cycle and once after "
        "its last; see ADR-0019).",
    ),
]
NightWalltimeMinutes = Annotated[
    int, typer.Option(help="Preferred (long) walltime for night/weekend jobs.")
]
NightFallbackWalltimeMinutes = Annotated[
    int,
    typer.Option(
        help="Shorter night walltime retried in the same slot if the long job cannot start."
    ),
]
NightMaxQueuedPerSite = Annotated[
    int | None,
    typer.Option(help="Waiting jobs per site at night/weekend (default: --max-queued-per-site)."),
]


def build_settings(
    *,
    datasets: list[str],
    sites: str,
    max_jobs: int,
    max_jobs_per_site: int,
    max_queued_per_site: int,
    walltime_minutes: int,
    day_walltime_minutes: int | None,
    night_walltime_minutes: int,
    night_fallback_walltime_minutes: int,
    stale_besteffort_minutes: int,
    policy_check: PolicyCheck,
    day_long_max_failures: int,
    chunk_overflow: float,
    submit_workers: int,
    night_max_queued_per_site: int | None,
    besteffort: bool,
    gpu_models: list[str],
    namespace: str | None,
    bucket: str,
    immediate_in_night: bool = True,
    window: int | None = None,
) -> "Settings":
    """The one mapping from CLI options to `Settings` (units converted here)."""
    from landuse_filter.application.controller import Settings

    return Settings(
        sites=sites.split(","),
        max_jobs_total=max_jobs,
        max_jobs_per_site=max_jobs_per_site,
        max_queued_per_site=max_queued_per_site,
        walltime=timedelta(minutes=walltime_minutes),
        day_walltime=timedelta(minutes=day_walltime_minutes or walltime_minutes),
        night_walltime=timedelta(minutes=night_walltime_minutes),
        night_fallback_walltime=timedelta(minutes=night_fallback_walltime_minutes),
        stale_besteffort_wait=timedelta(minutes=stale_besteffort_minutes),
        policy_check=policy_check.value,
        background_ingest=True,
        datasets=datasets,
        day_long_max_failures=day_long_max_failures,
        chunk_overflow=chunk_overflow,
        submit_workers=submit_workers,
        night_max_queued_per_site=night_max_queued_per_site,
        besteffort=besteffort,
        gpu_models=gpu_models,
        namespace=namespace,
        bucket=bucket,
        immediate_in_night=immediate_in_night,
        window=window,
    )


@g5k_app.command("inventory")
def g5k_inventory(
    work: Path = WORK, site: str = typer.Option("nancy", help="Frontend to query from.")
) -> None:
    """Refresh the GPU cluster inventory of all sites (Reference API)."""
    from landuse_filter.adapters import g5k
    from landuse_filter.application.inventory import admission, eligible, load_clusters

    script = Path(__file__).parents[1] / "adapters" / "frontend" / "inventory.py"
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
    walltime_minutes: WalltimeMinutes = OPS.walltime_minutes,
    day_walltime_minutes: DayWalltimeMinutes = OPS.day_walltime_minutes,
    day_long_max_failures: DayLongMaxFailures = OPS.day_long_max_failures,
    chunk_overflow: ChunkOverflow = OPS.chunk_overflow,
    submit_workers: SubmitWorkers = OPS.submit_workers,
    policy_check: PolicyCheckOption = PolicyCheck(OPS.policy_check),
    night_walltime_minutes: NightWalltimeMinutes = OPS.night_walltime_minutes,
    night_fallback_walltime_minutes: NightFallbackWalltimeMinutes = (
        OPS.night_fallback_walltime_minutes
    ),
    besteffort: bool = typer.Option(False),
    stale_besteffort_minutes: int = typer.Option(
        STALE_BESTEFFORT_MINUTES,
        min=1,
        help="Cancel our besteffort jobs still waiting this long after submission "
        "(their GPUs were taken); night/exotic jobs are never reaped. See ADR-0025.",
    ),
    max_queued_per_site: int = typer.Option(
        RUN_MAX_QUEUED_PER_SITE,
        help="Waiting jobs allowed per site when nothing is free (start predicted < 2 h).",
    ),
    night_max_queued_per_site: NightMaxQueuedPerSite = None,
    immediate_in_night: bool = typer.Option(
        True,
        help="At night/weekend also submit immediate-start jobs (no -t night, <= 1 h) "
        "where GPUs are free now, besides the queued night jobs. See ADR-0027.",
    ),
    window: int | None = typer.Option(
        None, help="Candidate concurrency (tuning); default: GPU profile."
    ),
    namespace: str | None = typer.Option(
        None, help="Store a candidate config's results under <fp>-<namespace>."
    ),
    bucket: str = typer.Option(OPS.bucket, help="Private HF Bucket for chunks and parts."),
    interval: int = typer.Option(OPS.interval_seconds, help="Seconds between cycles."),
    once: bool = typer.Option(False, help="Run a single cycle and exit."),
) -> None:
    """The controller loop: reconcile, pull results, submit where GPUs are free now."""
    settings = build_settings(
        datasets=datasets.split(","),
        sites=sites,
        max_jobs=max_jobs,
        max_jobs_per_site=max_jobs_per_site,
        max_queued_per_site=max_queued_per_site,
        walltime_minutes=walltime_minutes,
        day_walltime_minutes=day_walltime_minutes,
        day_long_max_failures=day_long_max_failures,
        chunk_overflow=chunk_overflow,
        policy_check=policy_check,
        submit_workers=submit_workers,
        night_walltime_minutes=night_walltime_minutes,
        night_fallback_walltime_minutes=night_fallback_walltime_minutes,
        night_max_queued_per_site=night_max_queued_per_site,
        immediate_in_night=immediate_in_night,
        besteffort=besteffort,
        stale_besteffort_minutes=stale_besteffort_minutes,
        gpu_models=[g for g in gpu_models.split(",") if g],
        window=window,
        namespace=namespace,
        bucket=bucket,
    )
    ctl = _controller(work, settings)
    from landuse_filter.application.controller import run_loop

    run_loop(ctl, interval=interval, once=once, emit=typer.echo)


@g5k_app.command("run-admission")
def g5k_run_admission(
    gpus: str = typer.Option(..., help="Comma-separated GPU keys to admit (full benchmark each)."),
    work: Path = WORK,
    sites: str = typer.Option(SITES),
    max_jobs: int = typer.Option(ADMISSION_MAX_JOBS, help="Per GPU type."),
    max_jobs_per_site: int = typer.Option(ADMISSION_MAX_JOBS_PER_SITE),
    walltime_minutes: WalltimeMinutes = OPS.walltime_minutes,
    day_walltime_minutes: DayWalltimeMinutes = OPS.day_walltime_minutes,
    day_long_max_failures: DayLongMaxFailures = OPS.day_long_max_failures,
    chunk_overflow: ChunkOverflow = OPS.chunk_overflow,
    submit_workers: SubmitWorkers = OPS.submit_workers,
    policy_check: PolicyCheckOption = PolicyCheck(OPS.policy_check),
    night_walltime_minutes: NightWalltimeMinutes = OPS.night_walltime_minutes,
    night_fallback_walltime_minutes: NightFallbackWalltimeMinutes = (
        OPS.night_fallback_walltime_minutes
    ),
    stale_besteffort_minutes: int = typer.Option(
        STALE_BESTEFFORT_MINUTES,
        min=1,
        help="Cancel besteffort jobs still waiting this long (ADR-0025).",
    ),
    max_queued_per_site: int = typer.Option(
        ADMISSION_MAX_QUEUED_PER_SITE,
        help="Waiting jobs allowed per site when nothing is free (night/weekend).",
    ),
    night_max_queued_per_site: NightMaxQueuedPerSite = None,
    interval: int = typer.Option(OPS.interval_seconds),
) -> None:
    """One process for every GPU-type admission run (namespace gpu-<key>), sharing one
    view of each site per cycle; afterwards run `luf bench admit --gpu <key>`."""
    from landuse_filter.application.controller import Controller, run_many
    from landuse_filter.application.site_cache import SiteCache

    cache = SiteCache()
    store = _store(work)
    controllers = [
        Controller(
            store,
            build_settings(
                datasets=["benchmark"],
                sites=sites,
                max_jobs=max_jobs,
                max_jobs_per_site=max_jobs_per_site,
                max_queued_per_site=max_queued_per_site,
                walltime_minutes=walltime_minutes,
                day_walltime_minutes=day_walltime_minutes,
                day_long_max_failures=day_long_max_failures,
                chunk_overflow=chunk_overflow,
                policy_check=policy_check,
                submit_workers=submit_workers,
                night_walltime_minutes=night_walltime_minutes,
                night_fallback_walltime_minutes=night_fallback_walltime_minutes,
                night_max_queued_per_site=night_max_queued_per_site,
                besteffort=True,
                stale_besteffort_minutes=stale_besteffort_minutes,
                gpu_models=[g],
                namespace=f"gpu-{g}",
                bucket=OPS.bucket,
            ),
            log=lambda m: typer.echo(m, err=True),
            sites=cache,
        )
        for g in gpus.split(",")
    ]
    run_many(controllers, cache, interval=interval, emit=typer.echo)


LOCKFILE = Path(__file__).resolve().parents[3] / "uv.lock"

# sagittaire's ancient CPUs crash modern wheels (SIGILL) and its /tmp is tiny.
CPU_JOB_PROPERTY = "gpu_count = 0 AND cluster != 'sagittaire'"


@g5k_app.command("cpu-job")
def g5k_cpu_job(
    mode: str = typer.Argument(..., help="plan, replan, publish, card or repair"),
    site: str = typer.Option(..., help="Site to run the CPU job on."),
    dataset: str = typer.Option(...),
    revision: str = typer.Option(...),
    walltime_minutes: int = typer.Option(60),
) -> None:
    """Submit one resumable planning or publishing job (default queue, one CPU node)."""
    from landuse_filter.application.cpu_guard import CpuJobCancelledError

    try:
        job_id, wall = _submit_cpu_job(mode, site, dataset, revision, walltime_minutes)
    except CpuJobCancelledError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{mode} job {job_id} on {site} ({wall})")


def _submit_cpu_job(
    mode: str, site: str, dataset: str, revision: str, minutes: int
) -> tuple[str, str]:
    """Deploy the code and submit one CPU job; returns (job id, walltime text)."""
    if mode not in ("plan", "replan", "publish", "card", "repair"):
        raise typer.BadParameter("mode must be plan, replan, publish, card or repair")
    import time
    from datetime import datetime

    from landuse_filter.adapters import g5k
    from landuse_filter.application.controller import PARIS, commit, git_archive
    from landuse_filter.application.cpu_guard import guard_start
    from landuse_filter.domain.capacity import walltime_text
    from landuse_filter.domain.policy import allowed_window
    from landuse_filter.domain.publish_loop import job_name

    window = allowed_window(datetime.now(PARIS), starts_now=True)
    wall = min(timedelta(minutes=minutes), window.max_walltime) if window else timedelta(hours=1)
    code_commit = commit()
    code = g5k.deploy_code(site, code_commit, git_archive(code_commit))
    g5k.ssh(site, f"mkdir -p {g5k.REMOTE_ROOT}/logs")
    g5k.policy_check(site)
    args = (
        ["-q", "default"]
        + (["-t", window.job_type] if window and window.job_type else [])
        + [
            "-p",
            CPU_JOB_PROPERTY,
            "-l",
            f"host=1,walltime={walltime_text(wall)}",
            "--checkpoint",
            "300",
            "-n",
            job_name(g5k.JOB_PREFIX, mode, dataset),
            "-O",
            f"{g5k.REMOTE_ROOT}/logs/%jobid%.out",
            "-E",
            f"{g5k.REMOTE_ROOT}/logs/%jobid%.err",
            f"{code}/scripts/node_job.sh {code} {mode} {dataset} {revision}",
        ]
    )
    submitted = datetime.now(PARIS)
    job_id = g5k.submit(site, args)
    guard_start(
        job_id=job_id,
        submitted=submitted,
        walltime=wall,
        now=lambda: datetime.now(PARIS),
        status=lambda j: g5k.scheduled_start(site, j),
        cancel=lambda j: g5k.cancel(site, j),
        sleep=time.sleep,
    )
    g5k.policy_check(site)
    return job_id, walltime_text(wall)


@g5k_app.command("publish-loop")
def g5k_publish_loop(
    dataset: str = typer.Option(...),
    revision: str = typer.Option(...),
    site: str = typer.Option(..., help="Site to submit the publish jobs on."),
    walltime_minutes: int = typer.Option(
        60, help="Walltime asked for each job (policy permitting)."
    ),
    interval_seconds: int = typer.Option(300, min=1, help="Pause between two checks."),
    max_interval_seconds: int = typer.Option(
        1800, min=1, help="Longest pause after repeated failures (slow frontend, timeouts)."
    ),
    bucket: str = typer.Option(OPS.bucket),
) -> None:
    """Resubmit `cpu-job publish` whenever none is live, until the bucket status says done."""
    import time
    from datetime import datetime

    from landuse_filter.adapters import g5k
    from landuse_filter.adapters.remote import BucketRemote
    from landuse_filter.application.controller import PARIS
    from landuse_filter.application.cpu_guard import live as live_jobs
    from landuse_filter.application.cpu_guard import sweep
    from landuse_filter.application.publish_loop import LoopIO, read_status, run_loop
    from landuse_filter.domain.publish_loop import job_name

    remote = BucketRemote(bucket)
    name = job_name(g5k.JOB_PREFIX, "publish", dataset)
    io = LoopIO(
        status=lambda: read_status(remote, dataset),
        live=lambda: live_jobs(g5k.our_jobs(site), name, now=datetime.now(PARIS)),
        sweep=lambda: sweep(
            g5k.our_jobs(site),
            name,
            now=datetime.now(PARIS),
            cancel=lambda j: g5k.cancel(site, j),
            log=lambda m: typer.echo(m, err=True),
        ),
        submit=lambda: _submit_cpu_job("publish", site, dataset, revision, walltime_minutes)[0],
        sleep=time.sleep,
        log=lambda m: typer.echo(m, err=True),
    )
    result = run_loop(
        io, revision, interval=interval_seconds, cap=max(max_interval_seconds, interval_seconds)
    )
    typer.echo(result)


@g5k_app.command("publish-status")
def g5k_publish_status(
    dataset: str = typer.Option(...),
    bucket: str = typer.Option(OPS.bucket),
) -> None:
    """Print the live phase of the running publish/card job (one small bucket download)."""
    import time

    from landuse_filter.adapters.remote import BucketRemote
    from landuse_filter.application.job_progress import read_progress

    state = read_progress(BucketRemote(bucket), dataset)
    if state is None:
        typer.echo(f"no progress marker for {dataset} (no job yet, or the bucket is unreachable)")
        raise typer.Exit(1)
    age = round(time.time() - state["updated_epoch"])
    counters = " ".join(f"{k}={v}" for k, v in state["counters"].items())
    typer.echo(
        f"job {state['job_id'] or '-'}  phase {state['phase']}  "
        f"elapsed {state['elapsed_seconds']}s  updated {age}s ago\n{counters}"
    )


@g5k_app.command("calibrate-job")
def g5k_calibrate_job(
    site: str = typer.Option(...),
    cluster: str = typer.Option(...),
    chunk: str = typer.Option(..., help="A benchmark chunk id (prompts for the sweep)."),
    work: Path = WORK,
) -> None:
    """Submit one GPU calibration job on ``cluster`` (1 h, starts now or is cancelled)."""
    from datetime import datetime

    from landuse_filter.adapters import g5k
    from landuse_filter.application.controller import PARIS, commit, git_archive
    from landuse_filter.application.inventory import load_clusters
    from landuse_filter.domain.capacity import oarsub_arguments
    from landuse_filter.domain.policy import allowed_window

    target = next(c for c in load_clusters(_store(work)) if c.site == site and c.name == cluster)
    # Same usage-policy window as the controller: -t night outside daytime, otherwise a
    # 1 h immediate job (regression: an untyped job at 18:40 was scheduled for 08:02).
    window = None if target.production else allowed_window(datetime.now(PARIS), starts_now=True)
    wall = min(timedelta(hours=1), window.max_walltime) if window else timedelta(hours=1)
    code_commit = commit()
    code = g5k.deploy_code(site, code_commit, git_archive(code_commit))
    g5k.ssh(site, f"mkdir -p {g5k.REMOTE_ROOT}/logs")
    g5k.policy_check(site)
    args = oarsub_arguments(
        target,
        wall,
        window.job_type if window else None,
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


def _synced_parts(store: "WorkStore", listing: list[tuple[str, float]]) -> frozenset[str]:
    """Parts on a site's disk whose result (parquet or manifest) this controller already has."""
    prefix = "luf/work/"
    return frozenset(
        p.removeprefix("luf/work/parts/")
        for p, _ in listing
        if p.startswith("luf/work/parts/")
        and (
            store.exists(p.removeprefix(prefix))
            or store.exists(p.removeprefix(prefix).replace(".parquet", ".json"))
        )
    )


def _live_commits(store: "WorkStore") -> set[str]:
    """Code commits that live assignments still run, plus the commit jobs would run now."""
    from landuse_filter.application.controller import commit

    assignments = [
        store.read_json(f"assignments/{p.name}") for p in store.path("assignments").glob("*.json")
    ]
    live = [a for a in assignments if a.get("state") in ("submitting", "submitted")]
    return {a["provenance"]["code_commit"] for a in live} | {commit()}


@g5k_app.command("clean")
def g5k_clean(
    sites: str = typer.Option(SITES),
    work: Path = WORK,
    apply: bool = typer.Option(False, help="Actually delete (default: dry run)."),
) -> None:
    """Delete stale project files under ~/luf on each site (old code, envs, logs, synced parts)."""
    from landuse_filter.adapters import g5k
    from landuse_filter.domain.cleanup import Entry, Keep, cleanup_plan

    store = _store(work)
    commits = frozenset(_live_commits(store))
    lock = hashlib.sha256(LOCKFILE.read_bytes()).hexdigest()[:12]
    for site in sites.split(","):
        try:
            listing = g5k.project_listing(site)
        except g5k.RemoteError as exc:
            typer.echo(f"{site}: unreachable ({exc})", err=True)
            continue
        keep = Keep(commits, lock, _synced_parts(store, listing))
        plan = cleanup_plan([Entry(p, a) for p, a in listing], keep)
        typer.echo(f"{site}: {len(plan)} path(s) {'deleted' if apply else 'would be deleted'}")
        for p in plan[:20]:
            typer.echo(f"  {p}")
        if apply and plan:
            g5k.remove(site, plan)
