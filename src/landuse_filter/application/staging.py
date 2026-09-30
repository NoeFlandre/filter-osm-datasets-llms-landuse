"""Moving assignments and results between the controller and sites via the bucket.

Chunk inputs and result parts live only in the private HF Bucket (ADR-0009); a site
receives just the small assignment file, and the controller pulls only manifests and
job summaries. There is no other transport.
"""

from collections.abc import Callable

from landuse_filter.adapters import g5k
from landuse_filter.adapters.remote import Remote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.assignment import Assignment
from landuse_filter.application.sync import fetch_manifests
from landuse_filter.application.work_progress import WorkProgress


class Transport:
    def __init__(
        self, store: WorkStore, remote: Remote, bucket_id: str, log: Callable[[str], None]
    ) -> None:
        self.store = store
        self.remote = remote
        self.bucket_id = bucket_id
        self.log = log

    def stage(self, site: str, a: Assignment) -> None:
        """Upload the assignment's chunk inputs once and ship the assignment file."""
        a.bucket = self.bucket_id
        self.upload_chunks(a.chunks)
        self.store.write_json(f"assignments/{a.id}.json", a.to_json())
        g5k.ssh(site, f"mkdir -p {g5k.REMOTE_ROOT}/work/assignments {g5k.REMOTE_ROOT}/logs")
        g5k.rsync(
            str(self.store.path(f"assignments/{a.id}.json")),
            f"{site}:{g5k.REMOTE_ROOT}/work/assignments/",
        )

    def upload_chunks(self, chunks: list[str]) -> None:
        """Put chunk inputs in the bucket once, then drop the local copies (scarce SSD)."""
        present = set(self.remote.ls("chunks/"))
        local = [c for c in chunks if self.store.exists(f"chunks/{c}.parquet")]
        todo = [c for c in local if f"chunks/{c}.parquet" not in present]
        self.remote.put(
            [(self.store.path(f"chunks/{c}.parquet"), f"chunks/{c}.parquet") for c in todo]
        )
        for c in local:
            self.store.path(f"chunks/{c}.parquet").unlink()

    def pull(self, progress: WorkProgress) -> None:
        index = progress.index(progress.work_fp)
        new = fetch_manifests(
            self.remote, self.store, f"parts/{progress.work_fp}/", seen=index.has_seen
        )
        index.ingest(self.store.path(p) for p in new)
        for p in new:  # the hashes are in the index: keep nothing on the scarce SSD
            self.store.path(p).unlink(missing_ok=True)
        fetch_manifests(self.remote, self.store, "jobs/")
