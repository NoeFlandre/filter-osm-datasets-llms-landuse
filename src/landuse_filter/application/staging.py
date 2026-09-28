"""Moving assignments and results between the controller and sites (bucket or spool)."""

from collections.abc import Callable

from landuse_filter.adapters import g5k
from landuse_filter.adapters.remote import BucketRemote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.sync import fetch_manifests
from landuse_filter.application.work_progress import WorkProgress


class Transport:
    def __init__(self, store: WorkStore, bucket: str | None, log: Callable[[str], None]) -> None:
        self.store = store
        self.bucket = bucket
        self.log = log

    def remote(self) -> BucketRemote | None:
        return BucketRemote(self.bucket) if self.bucket else None

    # --- controller -> site --------------------------------------------------------

    def stage(self, site: str, a: dict) -> None:
        """Ship the assignment to the site; bulk inputs travel via the bucket if set."""
        remote = self.remote()
        if remote is None:
            self.stage_spool(site, a)
            return
        a["bucket"] = self.bucket
        self.upload_chunks(remote, a["chunks"])
        self.store.write_json(f"assignments/{a['id']}.json", a)
        g5k.ssh(site, f"mkdir -p {g5k.REMOTE_ROOT}/work/assignments {g5k.REMOTE_ROOT}/logs")
        g5k.rsync(
            str(self.store.path(f"assignments/{a['id']}.json")),
            f"{site}:{g5k.REMOTE_ROOT}/work/assignments/",
        )

    def upload_chunks(self, remote: BucketRemote, chunks: list[str]) -> None:
        """Put chunk inputs in the bucket once, then drop the local copies (scarce SSD)."""
        present = set(remote.ls("chunks/"))
        local = [c for c in chunks if self.store.exists(f"chunks/{c}.parquet")]
        todo = [c for c in local if f"chunks/{c}.parquet" not in present]
        remote.put([(self.store.path(f"chunks/{c}.parquet"), f"chunks/{c}.parquet") for c in todo])
        for c in local:
            self.store.path(f"chunks/{c}.parquet").unlink()

    def stage_spool(self, site: str, a: dict) -> None:
        """Copy the assignment, its chunk inputs and existing parts to the site spool."""
        dirs = ("work/assignments", "work/chunks", "logs")
        g5k.ssh(site, " ".join(f"mkdir -p {g5k.REMOTE_ROOT}/{d};" for d in dirs))
        files = [f"chunks/{c}.parquet" for c in a["chunks"]]
        files += [
            str(p.relative_to(self.store.root))
            for c in a["chunks"]
            for p in self.store.part_paths(a["fp"], c)
        ]
        self.store.write_json(f"assignments/{a['id']}.json", a)
        files.append(f"assignments/{a['id']}.json")
        listing = self.store.path(f".stage-{a['id']}")
        listing.write_text("\n".join(files) + "\n")
        try:
            g5k.rsync(
                f"{self.store.root}/",
                f"{site}:{g5k.REMOTE_ROOT}/work/",
                extra=(f"--files-from={listing}",),
            )
        finally:
            listing.unlink(missing_ok=True)

    # --- site -> controller --------------------------------------------------------

    def pull(self, sites: list[str], ledger: list[dict], progress: WorkProgress) -> None:
        remote = self.remote()
        if remote is None:
            self.pull_spools(sites, ledger)
            return
        new = fetch_manifests(remote, self.store, f"parts/{progress.work_fp}/")
        progress.index(progress.work_fp).ingest(self.store.path(p) for p in new)
        fetch_manifests(remote, self.store, "jobs/")

    def pull_spools(self, sites: list[str], ledger: list[dict]) -> None:
        for site in sites:
            if not any(a["site"] == site and a.get("job_id") for a in ledger):
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
