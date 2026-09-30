"""`luf g5k` commands."""

import hashlib
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from landuse_filter.cli import (
    OPS,
    SITES,
    WORK,
    _store,
    g5k_app,
)

if TYPE_CHECKING:
    from landuse_filter.adapters.store import WorkStore
    from landuse_filter.application.controller import Controller, Settings


def _controller(work: Path, settings: "Settings") -> "Controller":
    from landuse_filter.application.controller import Controller

    return Controller(_store(work), settings, log=lambda m: typer.echo(m, err=True))


@g5k_app.command("inventory")
def g5k_inventory(
    work: Path = WORK, site: str = typer.Option("nancy", help="Frontend to query from.")
) -> None:
    """Refresh the GPU cluster inventory of all sites (Reference API)."""
    from landuse_filter.adapters import g5k
    from landuse_filter.application.inventory import admission, eligible, load_clusters

    script = Path(__file__).parent / "adapters" / "frontend" / "inventory.py"
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
    night_walltime_minutes: int = typer.Option(
        OPS.night_walltime_minutes, help="Walltime cap for night/weekend jobs."
    ),
    besteffort: bool = typer.Option(False),
    max_queued_per_site: int = typer.Option(
        1, help="Waiting jobs allowed per site when nothing is free (start predicted < 2 h)."
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
    from landuse_filter.application.controller import Settings

    settings = Settings(
        datasets=datasets.split(","),
        sites=sites.split(","),
        max_jobs_total=max_jobs,
        max_jobs_per_site=max_jobs_per_site,
        max_queued_per_site=max_queued_per_site,
        walltime=timedelta(minutes=walltime_minutes),
        night_walltime=timedelta(minutes=night_walltime_minutes),
        besteffort=besteffort,
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
    max_jobs: int = typer.Option(5, help="Per GPU type."),
    max_jobs_per_site: int = typer.Option(3),
    walltime_minutes: int = typer.Option(OPS.walltime_minutes),
    night_walltime_minutes: int = typer.Option(
        OPS.night_walltime_minutes, help="Walltime cap for night/weekend jobs."
    ),
    max_queued_per_site: int = typer.Option(
        2, help="Waiting jobs allowed per site when nothing is free (night/weekend)."
    ),
    interval: int = typer.Option(OPS.interval_seconds),
) -> None:
    """One process for every GPU-type admission run (namespace gpu-<key>), sharing one
    view of each site per cycle; afterwards run `luf bench admit --gpu <key>`."""
    from landuse_filter.application.controller import Controller, Settings, run_many
    from landuse_filter.application.site_cache import SiteCache

    cache = SiteCache()
    store = _store(work)
    controllers = [
        Controller(
            store,
            Settings(
                datasets=["benchmark"],
                sites=sites.split(","),
                max_jobs_total=max_jobs,
                max_jobs_per_site=max_jobs_per_site,
                max_queued_per_site=max_queued_per_site,
                walltime=timedelta(minutes=walltime_minutes),
                night_walltime=timedelta(minutes=night_walltime_minutes),
                besteffort=True,
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
    mode: str = typer.Argument(..., help="plan, replan, publish or repair"),
    site: str = typer.Option(..., help="Site to run the CPU job on."),
    dataset: str = typer.Option(...),
    revision: str = typer.Option(...),
    walltime_minutes: int = typer.Option(60),
) -> None:
    """Submit one resumable planning or publishing job (default queue, one CPU node)."""
    if mode not in ("plan", "replan", "publish", "repair"):
        raise typer.BadParameter("mode must be plan, replan, publish or repair")
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
            CPU_JOB_PROPERTY,
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
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from landuse_filter.adapters import g5k
    from landuse_filter.application.controller import commit, git_archive
    from landuse_filter.application.inventory import load_clusters
    from landuse_filter.domain.capacity import oarsub_arguments
    from landuse_filter.domain.policy import allowed_window

    target = next(c for c in load_clusters(_store(work)) if c.site == site and c.name == cluster)
    # Same usage-policy window as the controller: -t night outside daytime, otherwise a
    # 1 h immediate job (regression: an untyped job at 18:40 was scheduled for 08:02).
    window = (
        None
        if target.production
        else allowed_window(datetime.now(ZoneInfo("Europe/Paris")), starts_now=True)
    )
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
