"""Laptop-side controller: the single writer of chunk -> job assignments (ADR-0007).

One ``cycle`` is idempotent and restartable: it reconciles the ledger with the live
OAR jobs, pulls finished parts from site spools, verifies them, and, unless paused,
submits new short jobs where GPUs are free *now*. All state is files under the work
tree, so killing the controller at any point loses nothing.
"""

import subprocess
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from landuse_filter import config
from landuse_filter.adapters import g5k
from landuse_filter.adapters.remote import BucketRemote
from landuse_filter.adapters.store import CorruptPartError, WorkStore
from landuse_filter.application.bench_plan import SMOKE, smoke_fp
from landuse_filter.application.sync import fetch_manifests
from landuse_filter.domain.capacity import Cluster, free_gpus, oarsub_arguments
from landuse_filter.domain.completion import Part, State, progress
from landuse_filter.domain.fingerprint import config_fingerprint, serving_fingerprint
from landuse_filter.domain.gpu import Admission, GpuSpec, Profile, gpu_key, ineligibility
from landuse_filter.domain.policy import Window, allowed_window
from landuse_filter.domain.prompting import PROMPT_SHA256
from landuse_filter.domain.scheduling import Slot, assign_chunks, rank_slots

PARIS = ZoneInfo("Europe/Paris")
# oarsub's refusal when the account has no Abaca priority on a cluster.
BESTEFFORT_ONLY = "only access the required resources in besteffort"
SETUP = timedelta(minutes=8)
LATE_START = timedelta(minutes=15)
BACKOFF = timedelta(minutes=30)
POST_SUBMIT_WAIT = 20.0


@dataclass
class Settings:
    datasets: list[str]
    sites: list[str]
    max_jobs_total: int = 12
    max_jobs_per_site: int = 4
    max_queued_per_site: int = 0
    walltime: timedelta = timedelta(hours=1)
    night_walltime: timedelta = timedelta(hours=2)
    besteffort: bool = False
    gpu_models: list[str] = field(default_factory=list)  # allow-list of gpu keys; empty = admitted
    admit: list[str] = field(default_factory=list)  # gpu keys to run the one-time smoke gate for
    window: int | None = None  # candidate concurrency (tuning, issue #17); None = GPU profile
    namespace: str | None = None  # results of a candidate config live under <fp>-<namespace>
    bucket: str | None = None  # private HF Bucket holding chunks and parts (no bulk data locally)
    paused: bool = False


def load_clusters(store: WorkStore) -> list[Cluster]:
    return [
        Cluster(
            c["site"],
            c["cluster"],
            c["gpu"],
            c["memory_mib"],
            tuple(c["compute_capability"]),
            c["gpus_per_node"],
            c["nodes"],
            tuple(c["queues"]),
            c["exotic"],
            c.get("vendor", "Nvidia"),
            c.get("arch", "x86_64"),
        )
        for c in store.read_json("inventory.json")
    ]


def eligible(cluster: Cluster) -> bool:
    spec = GpuSpec(cluster.gpu, cluster.compute_capability, cluster.memory_mib)
    # SGLang/FlashInfer wheels are x86_64 only (excludes e.g. Lyon's GH200 nodes).
    return (
        cluster.vendor.lower() == "nvidia"
        and cluster.arch == "x86_64"
        and cluster.submittable
        and ineligibility(spec) is None
    )


def admission(store: WorkStore, gpu: str) -> Admission:
    path = f"gates/admission/{gpu_key(gpu)}.json"
    return Admission(store.read_json(path)["status"]) if store.exists(path) else Admission.PENDING


def profile_for(store: WorkStore, gpu: str) -> Profile:
    path = f"profiles/{gpu_key(gpu)}.json"
    return (
        Profile(**store.read_json(path))
        if store.exists(path)
        else Profile(gpu_key(gpu), max_running_requests=16)
    )


def commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


class Controller:
    def __init__(
        self, store: WorkStore, settings: Settings, log: Callable[[str], None] = print
    ) -> None:
        self.store = store
        self.settings = settings
        self.log = log
        self.cfg = config.reference_config()
        self.fp = config_fingerprint(self.cfg)
        self.now = datetime.now(PARIS)

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

    def ledger(self) -> list[dict]:
        return [
            self.store.read_json(f"assignments/{p.name}")
            for p in sorted(self.store.path("assignments").glob("*.json"))
        ]

    def live(self) -> list[dict]:
        return [a for a in self.ledger() if a.get("state") in ("submitting", "submitted")]

    # --- work ----------------------------------------------------------------------

    def plan_lines(self) -> list[dict]:
        seen: set[str] = set()
        lines = []
        for dataset in self.settings.datasets:
            for row in self.store.read_jsonl(f"plans/{dataset}/{self.fp}/chunks.jsonl"):
                if row["chunk_id"] not in seen:
                    seen.add(row["chunk_id"])
                    lines.append(row)
        return lines

    def done_count(self, chunk_id: str, fp: str | None = None) -> int:
        """Distinct texts of a chunk already generated (manifests + verified local parts)."""
        from landuse_filter.application.sync import manifest_shas

        fp = fp or self.work_fp
        shas = manifest_shas(self.store, fp, chunk_id)
        for path in self.store.part_paths(fp, chunk_id):
            try:
                shas.update(self.store.read_part(path).column("text_sha256").to_pylist())
            except CorruptPartError:
                self.quarantine(path)
        return len(shas)

    def chunk_state(self, chunk_id: str, fp: str | None = None) -> State:
        expected = self.store.read_chunk(chunk_id).column("text_sha256").to_pylist()
        parts = []
        for path in self.store.part_paths(fp or self.work_fp, chunk_id):
            try:
                parts.append(
                    Part(
                        path.stem,
                        tuple(self.store.read_part(path).column("text_sha256").to_pylist()),
                    )
                )
            except CorruptPartError:
                self.quarantine(path)
        return progress(expected, parts).state

    def quarantine(self, path: Path) -> None:
        target = self.store.path("quarantine") / path.relative_to(self.store.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        self.log(f"quarantined corrupt part {path.name}")

    def pending_chunks(self) -> list[tuple[str, int]]:
        complete = {r["chunk_id"] for r in self.store.read_jsonl(self.complete_log)}
        pending, newly = [], []
        for row in self.plan_lines():
            cid = row["chunk_id"]
            if cid in complete:
                continue
            if self.done_count(cid) >= row["size"]:
                newly.append({"chunk_id": cid})
            else:
                pending.append((cid, row["size"]))
        if newly:
            self.store.append_jsonl(self.complete_log, newly)
        return pending

    # --- cycle ---------------------------------------------------------------------

    def cycle(self, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(PARIS)
        self.now = now
        jobs = self.reconcile()
        self.pull()
        pending = self.pending_chunks()
        report = {
            "pending_chunks": len(pending),
            "live_jobs": sum(len(v) for v in jobs.values()),
            "submitted": [],
        }
        if self.settings.paused or not (pending or self.settings.admit):
            return report
        report["submitted"] = self.submit_smoke(now, jobs) + self.submit(now, jobs, pending)
        return report

    def reconcile(self) -> dict[str, list[g5k.Job]]:
        jobs: dict[str, list[g5k.Job]] = {}
        for site in self.settings.sites:
            try:
                jobs[site] = g5k.our_jobs(site)
            except g5k.RemoteError as exc:
                self.log(f"{site}: unreachable ({exc}); keeping its assignments")
                jobs[site] = [
                    g5k.Job(site, a["job_id"], a["name"], "Unknown", "")
                    for a in self.live()
                    if a["site"] == site and a.get("job_id")
                ]
        self.drop_drifted(jobs)
        by_name = {j.name: j for js in jobs.values() for j in js}
        for a in self.live():
            job = by_name.get(a["name"])
            if job and not a.get("job_id"):
                a.update(job_id=job.job_id, state="submitted")  # crash after oarsub: adopt
            elif not job:
                a["state"] = "ended"
            self.store.write_json(f"assignments/{a['id']}.json", a)
        return jobs

    def drop_drifted(self, jobs: dict[str, list[g5k.Job]]) -> None:
        """Cancel waiting jobs whose predicted start slipped past LATE_START."""
        clusters = {a["name"]: (a["site"], a["cluster"]) for a in self.live()}
        horizon = self.now.timestamp() + LATE_START.total_seconds()
        for site, site_jobs in jobs.items():
            for job in list(site_jobs):
                late = (
                    job.state == "Waiting" and job.scheduled_start and job.scheduled_start > horizon
                )
                if not late or job.name not in clusters:
                    continue
                try:
                    g5k.cancel(site, job.job_id)
                except g5k.RemoteError as exc:
                    self.log(f"{site}: could not cancel drifted job {job.job_id}: {exc}")
                    continue
                site_jobs.remove(job)
                self.store.write_json(
                    f"backoff/{site}_{clusters[job.name][1]}.json",
                    {"until": (self.now + BACKOFF).isoformat()},
                )
                self.log(f"{site}: job {job.job_id} start drifted; cancelled and backing off")

    def remote(self) -> "BucketRemote | None":
        return BucketRemote(self.settings.bucket) if self.settings.bucket else None

    def pull(self) -> None:
        remote = self.remote()
        if remote is None:
            self.pull_spools()
            return
        fetch_manifests(remote, self.store, f"parts/{self.work_fp}/")
        fetch_manifests(remote, self.store, "jobs/")

    def pull_spools(self) -> None:
        for site in self.settings.sites:
            if not any(a["site"] == site and a.get("job_id") for a in self.ledger()):
                continue
            try:
                g5k.ssh(site, f"mkdir -p {g5k.REMOTE_ROOT}/work/parts {g5k.REMOTE_ROOT}/work/jobs")
            except g5k.RemoteError as exc:
                self.log(f"{site}: unreachable for pull: {exc}")
                continue
            for sub in ("parts", "jobs"):
                self.store.path(sub).mkdir(parents=True, exist_ok=True)
                try:
                    g5k.rsync(
                        f"{site}:{g5k.REMOTE_ROOT}/work/{sub}/",
                        f"{self.store.path(sub)}/",
                        extra=("--ignore-existing",),
                    )
                except g5k.RemoteError as exc:
                    self.log(f"{site}: pull {sub} failed: {exc}")

    # --- submission -----------------------------------------------------------------

    def candidate_slots(
        self,
        now: datetime,
        jobs: dict[str, list[g5k.Job]],
        allow: Callable[[str], bool] | None = None,
    ) -> list[tuple[Slot, Cluster]]:
        allow = allow or self.allowed_gpu
        out = []
        clusters = [
            c for c in load_clusters(self.store) if c.site in self.settings.sites and eligible(c)
        ]
        for site in self.settings.sites:
            site_clusters = [
                c for c in clusters if c.site == site and allow(c.gpu) and self.accessible(c)
            ]
            if not site_clusters or len(jobs.get(site, [])) >= self.settings.max_jobs_per_site:
                continue
            try:
                nodes = g5k.site_status(site)["nodes"]
            except (g5k.RemoteError, KeyError, ValueError) as exc:
                self.log(f"{site}: status failed: {exc}")
                continue
            for c in site_clusters:
                window = allowed_window(now, starts_now=True) if not c.production else None
                wall, job_type = self.walltime_for(c, window)
                if self.backed_off(c, now):
                    continue
                free = free_gpus(
                    c,
                    nodes,
                    besteffort_counts=False,
                    now=now.timestamp(),
                    walltime_s=wall.total_seconds() if wall else 0.0,
                )
                if free <= 0 or wall is None:
                    continue
                prof = profile_for(self.store, c.gpu)
                out.append(
                    (
                        Slot(
                            site,
                            c.name,
                            c.gpu,
                            1,
                            free,
                            timedelta(0),
                            wall,
                            job_type,
                            prof.sentences_per_second,
                        ),
                        c,
                    )
                )
        ranked = rank_slots([s for s, _ in out], SETUP)
        by_key = {(s.site, s.cluster): c for s, c in out}
        return [(s, by_key[(s.site, s.cluster)]) for s in ranked]

    def starts_soon(self, site: str, job_id: str) -> bool:
        """OAR is the oracle: a job predicted to start > LATE_START from now is not a free slot."""
        import time

        time.sleep(POST_SUBMIT_WAIT)
        state, start = g5k.scheduled_start(site, job_id)
        if state in ("Running", "Launching", "toLaunch", "Finishing", "Terminated"):
            return True
        return start is not None and start - time.time() <= LATE_START.total_seconds()

    def back_off(self, cluster: Cluster) -> None:
        self.store.write_json(
            f"backoff/{cluster.site}_{cluster.name}.json",
            {"until": (self.now + BACKOFF).isoformat()},
        )

    def backed_off(self, cluster: Cluster, now: datetime) -> bool:
        path = f"backoff/{cluster.site}_{cluster.name}.json"
        return (
            self.store.exists(path)
            and datetime.fromisoformat(self.store.read_json(path)["until"]) > now
        )

    def accessible(self, cluster: Cluster) -> bool:
        """False for clusters that only admit us in besteffort, unless besteffort is on."""
        return self.settings.besteffort or not self.store.exists(
            f"access/{cluster.site}_{cluster.name}.json"
        )

    def allowed_gpu(self, gpu: str) -> bool:
        if self.settings.gpu_models:
            return gpu_key(gpu) in self.settings.gpu_models
        return admission(self.store, gpu) is Admission.ADMITTED

    def walltime_for(
        self, cluster: Cluster, window: Window | None
    ) -> tuple[timedelta | None, str | None]:
        if cluster.production:
            return self.settings.walltime, None
        if window is None:
            return None, None
        cap = self.settings.walltime if window.job_type is None else self.settings.night_walltime
        return min(cap, window.max_walltime), window.job_type

    def submit(
        self, now: datetime, jobs: dict[str, list[g5k.Job]], pending: list[tuple[str, int]]
    ) -> list[str]:
        submitted: list[str] = []
        taken = {
            c
            for a in self.live()
            if a.get("kind", "work") == "work" and a["fp"] == self.work_fp
            for c in a["chunks"]
        }
        total = sum(len(v) for v in jobs.values())
        per_site = {s: len(v) for s, v in jobs.items()}
        code_commit = commit()
        for slot, cluster in self.candidate_slots(now, jobs):
            for _ in range(slot.free_nodes):
                if (
                    total >= self.settings.max_jobs_total
                    or per_site.get(slot.site, 0) >= self.settings.max_jobs_per_site
                ):
                    break
                capacity = slot.sentences_per_second * max(
                    0.0, (slot.walltime - SETUP).total_seconds()
                )
                chunks = assign_chunks(pending, taken, capacity)
                if not chunks:
                    return submitted
                job_id = self.launch(slot, cluster, chunks, code_commit)
                if not job_id:
                    break  # this slot refused us; try the next cluster
                taken.update(chunks)
                total += 1
                per_site[slot.site] = per_site.get(slot.site, 0) + 1
                submitted.append(f"{slot.site}/{cluster.name}:{job_id}")
        return submitted

    # --- one-time GPU admission (issue #18) -------------------------------------------

    def smoke_pending(self, gpu: str) -> list[str]:
        base = f"{self.fp}-smoke"
        fp = smoke_fp(self.fp, gpu_key(gpu))
        rows = self.store.read_jsonl(f"plans/{SMOKE}/{base}/chunks.jsonl")
        return [r["chunk_id"] for r in rows if self.done_count(r["chunk_id"], fp) < r["size"]]

    def needs_smoke(self, gpu: str) -> bool:
        live = {a["gpu"] for a in self.live() if a.get("kind") == "smoke"}
        return (
            gpu_key(gpu) in self.settings.admit
            and admission(self.store, gpu) is Admission.PENDING
            and gpu not in live
            and bool(self.smoke_pending(gpu))
        )

    def submit_smoke(self, now: datetime, jobs: dict[str, list[g5k.Job]]) -> list[str]:
        if not self.settings.admit:
            return []
        submitted, done = [], set()
        for slot, cluster in self.candidate_slots(now, jobs, allow=self.needs_smoke):
            if slot.gpu in done:
                continue
            fp = smoke_fp(self.fp, gpu_key(slot.gpu))
            job_id = self.launch(
                slot, cluster, self.smoke_pending(slot.gpu), commit(), fp=fp, kind="smoke"
            )
            if job_id:
                done.add(slot.gpu)
                submitted.append(f"smoke:{slot.site}/{cluster.name}:{job_id}")
        return submitted

    def assignment(
        self,
        slot: Slot,
        chunks: list[str],
        code_commit: str,
        *,
        fp: str | None = None,
        kind: str = "work",
    ) -> dict:
        prof = profile_for(self.store, slot.gpu)
        speed = prof.engine_args()
        if self.settings.window:
            speed["max_running_requests"] = self.settings.window
        aid = uuid.uuid4().hex[:12]
        return {
            "id": aid,
            "name": f"{g5k.JOB_PREFIX}{aid}",
            "site": slot.site,
            "cluster": slot.cluster,
            "gpu": slot.gpu,
            "chunks": chunks,
            "kind": kind,
            "fp": fp or self.work_fp,
            "state": "submitting",
            "job_id": None,
            "window": self.settings.window or prof.max_running_requests,
            "engine_kwargs": config.engine_kwargs(self.cfg, speed),
            "sampling": self.cfg["sampling"],
            "walltime_s": int(slot.walltime.total_seconds()),
            "provenance": {
                "config_fingerprint": self.fp,
                "serving_fingerprint": serving_fingerprint(
                    {**self.cfg, "engine": {**self.cfg["engine"], **speed}}
                ),
                "model": self.cfg["model"],
                "draft": self.cfg["draft"],
                "prompt_sha256": PROMPT_SHA256,
                "code_commit": code_commit,
            },
        }

    def launch(
        self,
        slot: Slot,
        cluster: Cluster,
        chunks: list[str],
        code_commit: str,
        *,
        fp: str | None = None,
        kind: str = "work",
    ) -> str | None:
        a = self.assignment(slot, chunks, code_commit, fp=fp, kind=kind)
        site = slot.site
        try:
            code = g5k.deploy_code(site, code_commit, git_archive(code_commit))
            self.stage(site, a)
            g5k.policy_check(site)
            self.store.write_json(f"assignments/{a['id']}.json", a)  # before oarsub: crash-safe
            command = f"{code}/scripts/node_job.sh {code} {a['id']}"
            args = oarsub_arguments(
                cluster,
                slot.walltime,
                slot.job_type,
                a["name"],
                command=command,
                besteffort=self.settings.besteffort,
            )
            a["job_id"] = g5k.submit(site, args)
            a["state"] = "submitted"
            self.store.write_json(f"assignments/{a['id']}.json", a)
            if not self.starts_soon(site, a["job_id"]):
                g5k.cancel(site, a["job_id"])
                a["state"] = "cancelled_late_start"
                self.store.write_json(f"assignments/{a['id']}.json", a)
                self.back_off(cluster)
                self.log(
                    f"{site}/{cluster.name}: job {a['job_id']} would start late; "
                    "cancelled, backing off"
                )
                return None
            a["submitted_at"] = datetime.now(PARIS).isoformat(timespec="seconds")
            self.store.write_json(f"assignments/{a['id']}.json", a)
            g5k.policy_check(site)
            self.log(
                f"submitted {a['name']} on {site}/{cluster.name} "
                f"({len(chunks)} chunks): job {a['job_id']}"
            )
            return a["job_id"]
        except g5k.RemoteError as exc:
            if BESTEFFORT_ONLY in str(exc):
                self.store.write_json(
                    f"access/{site}_{cluster.name}.json", {"besteffort_only": True}
                )
            a["state"] = "failed_submit"
            a["error"] = str(exc)[-500:]
            self.store.write_json(f"assignments/{a['id']}.json", a)
            self.log(f"{site}: submission failed: {exc}")
            return None

    def stage(self, site: str, a: dict) -> None:
        """Ship the assignment to the site; bulk inputs travel via the bucket if set."""
        remote = self.remote()
        if remote is None:
            self.stage_spool(site, a)
            return
        a["bucket"] = self.settings.bucket
        self.upload_chunks(remote, a["chunks"])
        self.store.write_json(f"assignments/{a['id']}.json", a)
        g5k.ssh(site, f"mkdir -p {g5k.REMOTE_ROOT}/work/assignments {g5k.REMOTE_ROOT}/logs")
        g5k.rsync(
            str(self.store.path(f"assignments/{a['id']}.json")),
            f"{site}:{g5k.REMOTE_ROOT}/work/assignments/",
        )

    def upload_chunks(self, remote: "BucketRemote", chunks: list[str]) -> None:
        """Put chunk inputs in the bucket once, then drop the local copies (scarce SSD)."""
        present = set(remote.ls("chunks/"))
        local = [c for c in chunks if self.store.exists(f"chunks/{c}.parquet")]
        todo = [c for c in local if f"chunks/{c}.parquet" not in present]
        remote.put([(self.store.path(f"chunks/{c}.parquet"), f"chunks/{c}.parquet") for c in todo])
        for c in local:
            self.store.path(f"chunks/{c}.parquet").unlink()

    def stage_spool(self, site: str, a: dict) -> None:
        """Copy the assignment, its chunk inputs and existing parts to the site spool."""
        root = f"{site}:{g5k.REMOTE_ROOT}/work/"
        g5k.ssh(
            site,
            " ".join(
                f"mkdir -p {g5k.REMOTE_ROOT}/{d};"
                for d in ("work/assignments", "work/chunks", "logs")
            ),
        )
        files = [f"chunks/{c}.parquet" for c in a["chunks"]]
        files += [
            str(p.relative_to(self.store.root))
            for c in a["chunks"]
            for p in self.store.part_paths(a["fp"], c)
        ]
        tmp = self.store.path(f"assignments/{a['id']}.json")
        self.store.write_json(f"assignments/{a['id']}.json", a)
        files.append(str(tmp.relative_to(self.store.root)))
        listing = self.store.path(f".stage-{a['id']}")
        listing.write_text("\n".join(files) + "\n")
        try:
            g5k.rsync(f"{self.store.root}/", root, extra=(f"--files-from={listing}",))
        finally:
            listing.unlink(missing_ok=True)

    # --- control ---------------------------------------------------------------------

    def cancel_all(self) -> list[str]:
        cancelled = []
        for site in self.settings.sites:
            for job in g5k.our_jobs(site):
                g5k.cancel(site, job.job_id)
                cancelled.append(f"{site}:{job.job_id}")
        return cancelled


def git_archive(ref: str) -> bytes:
    return subprocess.run(
        ["git", "archive", "--format=tar", ref], capture_output=True, check=True
    ).stdout


def chunk_ids(rows: Iterable[dict]) -> list[str]:
    return [r["chunk_id"] for r in rows]
