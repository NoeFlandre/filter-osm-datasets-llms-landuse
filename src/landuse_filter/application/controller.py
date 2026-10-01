"""Laptop-side controller: the single writer of chunk -> job assignments (ADR-0007).

One ``cycle`` is idempotent and restartable: it reconciles the ledger with the live
OAR jobs, pulls finished parts from site spools, verifies them, and, unless paused,
submits new short jobs where GPUs are free *now*. All state is files under the work
tree, so killing the controller at any point loses nothing.
"""

import json
import subprocess
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from landuse_filter import config
from landuse_filter.adapters import g5k
from landuse_filter.adapters.remote import BucketRemote, Remote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.assignment import Assignment, CycleReport
from landuse_filter.application.inventory import admission, eligible, load_clusters, profile_for
from landuse_filter.application.memory import ClusterMemory
from landuse_filter.application.site_cache import SiteCache
from landuse_filter.application.staging import Transport
from landuse_filter.application.work_progress import WorkProgress
from landuse_filter.domain.capacity import Cluster, free_gpus, oarsub_arguments
from landuse_filter.domain.fingerprint import config_fingerprint, serving_fingerprint
from landuse_filter.domain.gpu import Admission, gpu_key
from landuse_filter.domain.policy import Window, allowed_window, is_daytime
from landuse_filter.domain.prompting import PROMPT_SHA256
from landuse_filter.domain.scheduling import (
    Slot,
    assign_chunks,
    rank_slots,
    slot_for,
    walltime_ladder,
)

PARIS = ZoneInfo("Europe/Paris")
# oarsub's refusal when the account has no Abaca priority on a cluster.
BESTEFFORT_ONLY = "only access the required resources in besteffort"
SETUP = timedelta(minutes=8)
LATE_START = timedelta(minutes=15)
BACKOFF = timedelta(minutes=30)
POST_SUBMIT_WAIT = 20.0
PULL_INTERVAL = 180.0  # seconds: ingest results again if a cycle of submissions runs longer
# Bounded queue (#20): when nothing is free, a site may hold a few of our waiting jobs
# predicted to start within QUEUED_START; they rank below free slots.
QUEUED_START = timedelta(hours=2)
QUEUED_WAIT = timedelta(hours=1)


@dataclass
class Settings:
    datasets: list[str]
    sites: list[str]
    max_jobs_total: int = 12
    max_jobs_per_site: int = 4
    max_queued_per_site: int = 0
    night_max_queued_per_site: int | None = None  # None: same as the day value
    walltime: timedelta = timedelta(hours=1)
    night_walltime: timedelta = timedelta(hours=2)
    night_fallback_walltime: timedelta = timedelta(minutes=30)
    besteffort: bool = False
    gpu_models: list[str] = field(default_factory=list)  # allow-list of gpu keys; empty = admitted
    window: int | None = None  # candidate concurrency (tuning, issue #17); None = GPU profile
    namespace: str | None = None  # results of a candidate config live under <fp>-<namespace>
    bucket: str = "NoeFlandre/landuse-filter-work"  # private HF Bucket: chunks, parts (ADR-0009)
    paused: bool = False


DEPLOY_REF = "origin/main"
# Absolute: a controller's working directory can vanish when the volume remounts
# (regression: every controller died on `git rev-parse` with status 128).
REPO = Path(__file__).resolve().parents[3]


def commit(ref: str = DEPLOY_REF) -> str:
    """The commit jobs run: the latest merged main, never a local feature branch.

    Background loops share the working checkout, which may be on a branch; deploying
    HEAD would ship unreviewed code to Grid'5000.
    """
    git = ["git", "-C", str(REPO)]
    subprocess.run([*git, "fetch", "-q", "origin", "main"], capture_output=True, check=False)

    def rev_parse() -> str:
        return subprocess.run(
            [*git, "rev-parse", ref], capture_output=True, text=True, check=True
        ).stdout.strip()

    for _ in range(ARCHIVE_ATTEMPTS - 1):
        try:  # another fetch updating the ref at the same moment makes rev-parse fail (128)
            return rev_parse()
        except subprocess.CalledProcessError:
            time.sleep(ARCHIVE_RETRY_DELAY)
    return rev_parse()


class Controller:
    def __init__(
        self,
        store: WorkStore,
        settings: Settings,
        log: Callable[[str], None] = print,
        sites: "SiteCache | None" = None,
        remote: "Remote | None" = None,
    ) -> None:
        self.store = store
        self.sites = sites  # shared per-cycle view when several controllers run together
        self.settings = settings
        self.log = log
        self.cfg = config.reference_config()
        self.fp = config_fingerprint(self.cfg)
        self.now = datetime.now(PARIS)
        self.memory = ClusterMemory(store)
        self.transport = Transport(
            store, remote or BucketRemote(settings.bucket), settings.bucket, log
        )
        self.progress = WorkProgress(
            store,
            plan_fp=self.fp,
            work_fp=self.work_fp,
            datasets=settings.datasets,
            complete_log=self.complete_log,
            log=log,
        )

    # --- ledger -------------------------------------------------------------------

    @property
    def work_fp(self) -> str:
        """Where this controller's results go: production fp, or a candidate namespace."""
        return f"{self.fp}-{self.settings.namespace}" if self.settings.namespace else self.fp

    @property
    def complete_log(self) -> str:
        return (
            "complete.jsonl"
            if not self.settings.namespace
            else f"complete-{self.settings.namespace}.jsonl"
        )

    def ledger(self) -> list[Assignment]:
        return [
            Assignment.from_json(self.store.read_json(f"assignments/{p.name}"))
            for p in sorted(self.store.path("assignments").glob("*.json"))
        ]

    def save(self, a: Assignment) -> None:
        self.store.write_json(f"assignments/{a.id}.json", a.to_json())

    def live(self) -> list[Assignment]:
        """This controller's live assignments: its namespace, on its sites.

        Several controllers (e.g. one GPU-admission run per GPU type) may share a work
        tree; each only reconciles its own assignments, so none releases another's work.
        Per-site job caps still count every ``luf-`` job on the site.
        """
        return [
            a
            for a in self.ledger()
            if a.state in ("submitting", "submitted")
            and a.fp == self.work_fp
            and a.site in self.settings.sites
        ]

    # --- cycle ---------------------------------------------------------------------

    def cycle(self, now: datetime | None = None) -> CycleReport:
        now = now or datetime.now(PARIS)
        self.now = now
        jobs = self.reconcile()
        self.pull()
        pending = self.progress.pending()
        report = CycleReport(len(pending), sum(len(v) for v in jobs.values()))
        if self.settings.paused or not pending:
            return report
        report.submitted = self.submit(now, jobs, pending)
        return report

    def reconcile(self) -> dict[str, list[g5k.Job]]:
        jobs = {site: self._our_jobs(site) for site in self.settings.sites}
        self.drop_drifted(jobs)
        by_name = {j.name: j for js in jobs.values() for j in js}
        for a in self.live():
            job = by_name.get(a.name)
            if job and not a.job_id:
                a.job_id, a.state = job.job_id, "submitted"  # crash after oarsub: adopt
            elif not job:
                a.state = "ended"
            self.save(a)
        return jobs

    def _our_jobs(self, site: str) -> list[g5k.Job]:
        """Our jobs on a site; when it is unreachable, the ones we believe are still there."""
        try:
            return self.sites.our_jobs(site) if self.sites else g5k.our_jobs(site)
        except g5k.RemoteError as exc:
            self.log(f"{site}: unreachable ({exc}); keeping its assignments")
            return [
                g5k.Job(site, a.job_id, a.name, "Unknown", "")
                for a in self.live()
                if a.site == site and a.job_id
            ]

    def drop_drifted(self, jobs: dict[str, list[g5k.Job]]) -> None:
        """Cancel waiting jobs whose predicted start slipped past their tolerance."""
        mine = {a.name: a for a in self.live()}
        for site, site_jobs in jobs.items():
            for job in list(site_jobs):
                a = mine.get(job.name)
                if a is None or job.state != "Waiting" or not job.scheduled_start:
                    continue
                tolerance = a.late_tolerance(LATE_START.total_seconds())
                if job.scheduled_start <= self.now.timestamp() + tolerance:
                    continue
                try:
                    g5k.cancel(site, job.job_id)
                except g5k.RemoteError as exc:
                    self.log(f"{site}: could not cancel drifted job {job.job_id}: {exc}")
                    continue
                site_jobs.remove(job)
                self.memory.back_off(site, a.cluster, self.now + BACKOFF)
                self.log(f"{site}: job {job.job_id} start drifted; cancelled and backing off")

    def pull(self) -> None:
        self.transport.pull(self.progress)

    # --- submission -----------------------------------------------------------------

    def candidate_slots(
        self,
        now: datetime,
        jobs: dict[str, list[g5k.Job]],
        allow: Callable[[str], bool] | None = None,
    ) -> list[tuple[Slot, Cluster]]:
        allow = allow or self.allowed_gpu
        clusters = [
            c for c in load_clusters(self.store) if c.site in self.settings.sites and eligible(c)
        ]
        out = [
            pair
            for site in self.settings.sites
            for pair in self._site_slots(site, clusters, now, jobs, allow)
        ]
        ranked = rank_slots([s for s, _ in out], SETUP)
        by_key = {(s.site, s.cluster): c for s, c in out}
        return [(s, by_key[(s.site, s.cluster)]) for s in ranked]

    def _site_slots(
        self,
        site: str,
        clusters: list[Cluster],
        now: datetime,
        jobs: dict[str, list[g5k.Job]],
        allow: Callable[[str], bool],
    ) -> list[tuple[Slot, Cluster]]:
        """The slots of one site's usable clusters (none when it is full or unreadable)."""
        usable = [c for c in clusters if c.site == site and allow(c.gpu) and self.accessible(c)]
        if not usable or len(jobs.get(site, [])) >= self.settings.max_jobs_per_site:
            return []
        nodes = self._site_nodes(site)
        if nodes is None:
            return []
        slots = ((self._cluster_slot(c, nodes, now, jobs), c) for c in usable)
        return [(slot, c) for slot, c in slots if slot]

    def _site_nodes(self, site: str) -> dict | None:
        """The site's live node states, or ``None`` (logged) when they cannot be read."""
        try:
            status = self.sites.site_status(site) if self.sites else g5k.site_status(site)
            return status["nodes"]
        except (g5k.RemoteError, KeyError, ValueError) as exc:
            self.log(f"{site}: status failed: {exc}")
            return None

    def _cluster_slot(
        self, c: Cluster, nodes: dict, now: datetime, jobs: dict[str, list[g5k.Job]]
    ) -> Slot | None:
        if self.memory.backed_off(c, now):
            return None
        window = allowed_window(now, starts_now=True) if not c.production else None
        ladder, job_type = self.ladder_for(c, window)
        wall = ladder[0] if ladder else None
        besteffort = self.memory.besteffort_only(c)
        free = free_gpus(
            c,
            nodes,
            # A regular job preempts besteffort ones; a besteffort job needs free GPUs.
            besteffort_counts=not besteffort,
            now=now.timestamp(),
            walltime_s=wall.total_seconds() if wall else 0.0,
        )
        return slot_for(
            site=c.site,
            cluster=c.name,
            gpu=c.gpu,
            free=free,
            walltime=wall,
            job_type=job_type,
            sentences_per_second=profile_for(self.store, c.gpu).sentences_per_second,
            besteffort=besteffort,
            queue_room=self.queue_room(c.site, jobs, night=not is_daytime(now)),
            queued_wait=QUEUED_WAIT,
            fallbacks=tuple(ladder[1:]),
        )

    def queue_room(self, site: str, jobs: dict[str, list[g5k.Job]], *, night: bool = False) -> bool:
        waiting = sum(j.state == "Waiting" for j in jobs.get(site, []))
        return waiting < self.queue_limit(night=night)

    def queue_limit(self, *, night: bool) -> int:
        night_limit = self.settings.night_max_queued_per_site
        if night and night_limit is not None:
            return night_limit
        return self.settings.max_queued_per_site

    def starts_soon(self, site: str, job_id: str, tolerance: timedelta = LATE_START) -> bool:
        """OAR is the oracle: a job predicted to start > LATE_START from now is not a free slot."""
        import time

        time.sleep(POST_SUBMIT_WAIT)
        state, start = g5k.scheduled_start(site, job_id)
        if state in ("Running", "Launching", "toLaunch", "Finishing", "Terminated"):
            return True
        return start is not None and start - time.time() <= tolerance.total_seconds()

    def accessible(self, cluster: Cluster) -> bool:
        """False for clusters that only admit us in besteffort, unless besteffort is on."""
        return self.settings.besteffort or not self.memory.besteffort_only(cluster)

    def allowed_gpu(self, gpu: str) -> bool:
        if self.settings.gpu_models:
            return gpu_key(gpu) in self.settings.gpu_models
        return admission(self.store, gpu) is Admission.ADMITTED

    def walltime_for(
        self, cluster: Cluster, window: Window | None
    ) -> tuple[timedelta | None, str | None]:
        ladder, job_type = self.ladder_for(cluster, window)
        return (ladder[0] if ladder else None), job_type

    def ladder_for(
        self, cluster: Cluster, window: Window | None
    ) -> tuple[tuple[timedelta, ...], str | None]:
        """Walltimes to try for a slot (preferred first) and the OAR job type."""
        if cluster.production:
            return (self.settings.walltime,), None
        if window is None:
            return (), None
        ladder = walltime_ladder(
            window_max=window.max_walltime,
            night=window.job_type is not None,
            day=self.settings.walltime,
            preferred=self.settings.night_walltime,
            fallback=self.settings.night_fallback_walltime,
        )
        return ladder, window.job_type

    def submit(
        self, now: datetime, jobs: dict[str, list[g5k.Job]], pending: list[tuple[str, int]]
    ) -> list[str]:
        submitted: list[str] = []
        taken = self._taken_chunks()
        total = sum(len(v) for v in jobs.values())
        per_site = {s: len(v) for s, v in jobs.items()}
        code_commit = commit()
        last_pull = time.monotonic()
        for slot, cluster in self.candidate_slots(now, jobs):
            for _ in range(slot.free_nodes):
                if (
                    total >= self.settings.max_jobs_total
                    or per_site.get(slot.site, 0) >= self.settings.max_jobs_per_site
                ):
                    break
                if time.monotonic() - last_pull >= PULL_INTERVAL:
                    self.pull()  # each submission waits ~20 s: do not let results pile up
                    last_pull = time.monotonic()
                job_id, chunks = self.launch_ladder(slot, cluster, pending, taken, code_commit)
                if not chunks:
                    return submitted
                if not job_id:
                    break  # this slot refused us; try the next cluster
                taken.update(chunks)
                total += 1
                per_site[slot.site] = per_site.get(slot.site, 0) + 1
                submitted.append(f"{slot.site}/{cluster.name}:{job_id}")
        return submitted

    def launch_ladder(
        self,
        slot: Slot,
        cluster: Cluster,
        pending: list[tuple[str, int]],
        taken: set[str],
        code_commit: str,
    ) -> tuple[str | None, list[str]]:
        """Try the slot's walltimes in order; only the last failure backs the cluster off.

        Every attempt re-assigns chunks for its own walltime; a failed attempt has already
        released its chunks (its assignment is no longer live). Returns the job id (``None``
        when every walltime failed) and the chunks of the last attempt (empty: nothing left).
        """
        walltimes = (slot.walltime, *slot.fallbacks)
        chunks: list[str] = []
        for i, wall in enumerate(walltimes):
            attempt = replace(slot, walltime=wall)
            chunks = assign_chunks(pending, taken, _capacity(attempt))
            if not chunks:
                return None, []
            job_id = self.launch(
                attempt, cluster, chunks, code_commit, final=i == len(walltimes) - 1
            )
            if job_id:
                return job_id, chunks
        return None, chunks

    def _taken_chunks(self) -> set[str]:
        """Chunks held by a live work job of this namespace."""
        return {
            c for a in self.live() if a.kind == "work" and a.fp == self.work_fp for c in a.chunks
        }

    def assignment(
        self,
        slot: Slot,
        chunks: list[str],
        code_commit: str,
        *,
        fp: str | None = None,
        kind: str = "work",
    ) -> Assignment:
        prof = profile_for(self.store, slot.gpu)
        speed = prof.engine_args()
        if self.settings.window:
            speed["max_running_requests"] = self.settings.window
        aid = uuid.uuid4().hex[:12]
        return Assignment(
            id=aid,
            name=f"{g5k.JOB_PREFIX}{aid}",
            site=slot.site,
            cluster=slot.cluster,
            gpu=slot.gpu,
            chunks=chunks,
            kind=kind,
            state="submitting",
            fp=fp or self.work_fp,
            window=self.settings.window or prof.max_running_requests,
            engine_kwargs=config.engine_kwargs(self.cfg, speed),
            sampling=dict(self.cfg["sampling"]),
            walltime_s=int(slot.walltime.total_seconds()),
            late_after_s=int((QUEUED_START if slot.queued else LATE_START).total_seconds()),
            provenance={
                "config_fingerprint": self.fp,
                "serving_fingerprint": serving_fingerprint(
                    {**self.cfg, "engine": {**self.cfg["engine"], **speed}}
                ),
                "model": self.cfg["model"],
                "draft": self.cfg["draft"],
                "prompt_sha256": PROMPT_SHA256,
                "code_commit": code_commit,
            },
        )

    def launch(
        self,
        slot: Slot,
        cluster: Cluster,
        chunks: list[str],
        code_commit: str,
        *,
        fp: str | None = None,
        kind: str = "work",
        final: bool = True,
    ) -> str | None:
        a = self.assignment(slot, chunks, code_commit, fp=fp, kind=kind)
        site = slot.site
        try:
            code = g5k.deploy_code(site, code_commit, git_archive(code_commit))
            self.transport.stage(site, a)
            g5k.policy_check(site)
            self.save(a)  # before oarsub: crash-safe
            command = f"{code}/scripts/node_job.sh {code} {a.id}"
            args = oarsub_arguments(
                cluster,
                slot.walltime,
                slot.job_type,
                a.name,
                command=command,
                besteffort=slot.besteffort,
            )
            a.job_id = g5k.submit(site, args)
            if self.sites:
                self.sites.invalidate(site)
            a.state = "submitted"
            self.save(a)
            if not self.starts_soon(site, a.job_id, QUEUED_START if slot.queued else LATE_START):
                g5k.cancel(site, a.job_id)
                a.state = "cancelled_late_start"
                self.save(a)
                self._late_cancelled(a, cluster, final=final)
                return None
            a.submitted_at = datetime.now(PARIS).isoformat(timespec="seconds")
            self.save(a)
            g5k.policy_check(site)
            self.log(
                f"submitted {a.name} on {site}/{cluster.name} "
                f"({len(chunks)} chunks): job {a.job_id}"
            )
            return a.job_id
        except g5k.RemoteError as exc:
            if BESTEFFORT_ONLY in str(exc):
                self.memory.remember_besteffort_only(site, cluster.name)
            a.state = "failed_submit"
            a.error = str(exc)[-500:]
            self.save(a)
            self.log(f"{site}: submission failed: {exc}")
            return None

    def _late_cancelled(self, a: Assignment, cluster: Cluster, *, final: bool) -> None:
        if final:
            self.memory.back_off(cluster.site, cluster.name, self.now + BACKOFF)
        self.log(
            f"{a.site}/{cluster.name}: job {a.job_id} would start late; cancelled"
            + (", backing off" if final else ", retrying with a shorter walltime")
        )

    # --- control ---------------------------------------------------------------------

    def cancel_all(self) -> list[str]:
        cancelled = []
        for site in self.settings.sites:
            for job in g5k.our_jobs(site):
                g5k.cancel(site, job.job_id)
                cancelled.append(f"{site}:{job.job_id}")
        return cancelled


def _capacity(slot: Slot) -> float:
    """Sentences a job in ``slot`` is expected to finish once its environment is up."""
    return slot.sentences_per_second * max(0.0, (slot.walltime - SETUP).total_seconds())


ARCHIVE_ATTEMPTS = 3
ARCHIVE_RETRY_DELAY = 5.0


def _tar_of(ref: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO), "archive", "--format=tar", ref],
        capture_output=True,
        check=True,
    ).stdout


class ArchiveCache:
    """``git archive`` of a ref, read once per commit and retried on transient faults.

    Retried because the external drive can fault under load (``git archive`` died with
    SIGBUS), and cached because every cycle used to re-read a commit already deployed.
    The reader and the sleep are injectable so the policy is testable without git.
    """

    def __init__(
        self,
        read: Callable[[str], bytes] = _tar_of,
        sleep: Callable[[float], None] = time.sleep,
        attempts: int = ARCHIVE_ATTEMPTS,
        delay: float = ARCHIVE_RETRY_DELAY,
    ) -> None:
        self.read = read
        self.sleep = sleep
        self.attempts = attempts
        self.delay = delay
        self.cache: dict[str, bytes] = {}

    def __call__(self, ref: str) -> bytes:
        if ref not in self.cache:
            for attempt in range(self.attempts):
                try:
                    self.cache[ref] = self.read(ref)
                    break
                except subprocess.CalledProcessError:
                    if attempt == self.attempts - 1:
                        raise
                    self.sleep(self.delay)
        return self.cache[ref]


_ARCHIVES = ArchiveCache()


def git_archive(ref: str) -> bytes:
    """The repository at ``ref`` as a tar (process-wide cache)."""
    return _ARCHIVES(ref)


def log_failure(log: Callable[[str], None], headline: str, exc: Exception) -> None:
    """Headline on one line, then the traceback on clearly marked ``  | `` lines.

    Keep-alive handlers swallow the error; without the traceback the failing line in
    the code is unknowable from the production log.
    """
    log(f"{headline} ({type(exc).__name__}: {exc})")
    for line in "".join(traceback.format_exception(exc)).rstrip().splitlines():
        log(f"  | {line}")


def run_loop(
    ctl: "Controller",
    *,
    interval: float,
    once: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    emit: Callable[[str], None] = print,
) -> None:
    """Run cycles until nothing is left, surviving any transient failure.

    A failed cycle (network, Hub, git, a vanished working directory) is logged and
    retried at the next interval; the loop never dies because of one bad cycle
    (regression: every controller crashed on a git error when the volume remounted).
    """
    while True:
        if ctl.store.exists("PAUSED"):
            ctl.settings.paused = True
        try:
            report = ctl.cycle()
        except Exception as exc:  # noqa: BLE001 - keep the controller alive; state is on disk
            log_failure(ctl.log, f"cycle failed; retrying in {interval:.0f}s", exc)
            if once:
                return
            sleep(interval)
            continue
        emit(json.dumps(report.to_json()))
        if once or (report.pending_chunks == 0 and report.live_jobs == 0):
            return
        sleep(interval)


def run_many(
    controllers: list["Controller"],
    sites: SiteCache,
    *,
    interval: float,
    sleep: Callable[[float], None] = time.sleep,
    emit: Callable[[str], None] = print,
) -> None:
    """One process, several namespaces: each cycle every controller runs once over a
    shared per-cycle site view; finished namespaces drop out; failures never kill it."""
    active = list(controllers)
    while active:
        sites.reset()
        for ctl in list(active):
            try:
                report = ctl.cycle()
            except Exception as exc:  # noqa: BLE001 - keep the others running; state is on disk
                log_failure(ctl.log, f"{ctl.work_fp}: cycle failed", exc)
                continue
            emit(json.dumps({"namespace": ctl.settings.namespace, **report.to_json()}))
            # Done when this namespace has no pending chunks and no live jobs of its own
            # (report.live_jobs counts every luf- job on its sites).
            if report.pending_chunks == 0 and not ctl.live():
                active.remove(ctl)
        if active:
            sleep(interval)
