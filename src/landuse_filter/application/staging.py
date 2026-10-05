"""Moving assignments and results between the controller and sites via the bucket.

Chunk inputs and result parts live only in the private HF Bucket (ADR-0009); a site
receives just the small assignment file, and the controller pulls only manifests and
job summaries. There is no other transport.
"""

import threading
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass

from landuse_filter.adapters import g5k
from landuse_filter.adapters.indexes import ProgressIndex
from landuse_filter.adapters.remote import Remote, rate_limit_headers
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.assignment import Assignment
from landuse_filter.application.sync import fetch_manifests
from landuse_filter.application.work_progress import WorkProgress

RECONCILE_INTERVAL = 3600.0  # seconds between full ``parts/<fp>/`` listings (the safety net)
LIVE_INTERVAL = 900.0  # seconds between listings of the chunks of live assignments
PULL_BUDGET = 240.0  # seconds after which a pull starts no new listing (ADR-0031)


@dataclass
class PullReport:
    """What one pull did, for the log line that makes stalls visible."""

    listings: int = 0  # successful listings (full, per chunk, jobs/)
    manifests: int = 0  # new manifests ingested
    failed: int = 0  # listings given up on (rate limit)
    deferred: int = 0  # per-chunk listings postponed by the time budget
    full: bool = False
    limited: bool = False
    seconds: float = 0.0

    def line(self) -> str:
        return (
            f"ingest: {'full' if self.full else 'incremental'} pull, {self.listings} listings, "
            f"{self.manifests} new manifests, {self.failed} rate-limited, "
            f"{self.deferred} deferred, {self.seconds:.0f} s"
        )


class Transport:
    def __init__(
        self,
        store: WorkStore,
        remote: Remote,
        bucket_id: str,
        log: Callable[[str], None],
        *,
        reconcile_interval: float = RECONCILE_INTERVAL,
        live_interval: float = LIVE_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.remote = remote
        self.ingest_remote = remote  # the controller swaps in a short 429 budget (ADR-0031)
        self.bucket_id = bucket_id
        self.log = log
        self._bucket_lock = threading.Lock()  # one bucket upload at a time across sites
        self._state_lock = threading.Lock()  # workers track chunks while the main thread pulls
        self.reconcile_interval = reconcile_interval
        self.live_interval = live_interval
        self.clock = clock
        self._live: set[str] = set()  # chunks of the live assignments at the previous pull
        self._owed: set[str] = set()  # ended chunks whose final listing has not succeeded yet
        self._last_full: float | None = None
        self._last_live: float | None = None
        self._jobs_due = True

    def stage(self, site: str, a: Assignment) -> None:
        """Upload the assignment's chunk inputs once and ship the assignment file."""
        a.bucket = self.bucket_id
        self.track(a.chunks)  # even a job that ends before the next pull gets its final listing
        self.upload_chunks(a.chunks)
        self.store.write_json(f"assignments/{a.id}.json", a.to_json())
        g5k.ssh(site, f"mkdir -p {g5k.REMOTE_ROOT}/work/assignments {g5k.REMOTE_ROOT}/logs")
        g5k.rsync(
            str(self.store.path(f"assignments/{a.id}.json")),
            f"{site}:{g5k.REMOTE_ROOT}/work/assignments/",
        )

    def upload_chunks(self, chunks: list[str]) -> None:
        """Put chunk inputs in the bucket once, then drop the local copies (scarce SSD)."""
        local = [c for c in chunks if self.store.exists(f"chunks/{c}.parquet")]
        if not local:  # planned on a node: already in the bucket, and listing it takes ~a minute
            return
        with self._bucket_lock:
            self._upload(local)

    def _upload(self, local: list[str]) -> None:
        present = set(self.remote.ls("chunks/"))
        todo = [c for c in local if f"chunks/{c}.parquet" not in present]
        self.remote.put(
            [(self.store.path(f"chunks/{c}.parquet"), f"chunks/{c}.parquet") for c in todo]
        )
        for c in local:
            self.store.path(f"chunks/{c}.parquet").unlink()

    def track(self, chunks: Collection[str]) -> None:
        """Remember chunks about to be worked on: their parts are listed once they end."""
        with self._state_lock:
            self._live |= set(chunks)

    def pull(
        self,
        progress: WorkProgress,
        live_chunks: Collection[str] | None = None,
        *,
        budget: float = PULL_BUDGET,
    ) -> PullReport:
        """Ingest new manifests, listing only the chunk prefixes that can have new parts.

        Listing all of ``parts/<fp>/`` costs one Hub API request per ~1000 entries and grows
        with the run (ADR-0029), so a pull lists the chunks of assignments that ended since
        the previous pull (once, or until the listing succeeds) and, every ``live_interval``,
        those of the live ones. A full listing runs on the first pull of a process, every
        ``reconcile_interval``, and whenever ``live_chunks`` is unknown (``None``): nothing
        stays missed, and the progress index keeps ingestion idempotent.

        A pull is bounded (ADR-0031): the first rate-limited listing ends it (what is left stays
        owed or due for the next pull), and no new listing starts after ``budget`` seconds.
        """
        started = self.clock()
        report = PullReport()
        fp = progress.work_fp
        index = progress.index(fp)
        live = None if live_chunks is None else set(live_chunks)
        if live is not None:
            with self._state_lock:
                ended = self._live - live
                self._owed |= ended
                self._jobs_due = self._jobs_due or bool(ended)
                self._live = live
        full = (
            live is None
            or self._last_full is None
            or started - self._last_full >= self.reconcile_interval
        )
        if full:
            report.full = True
            if self._attempt(report, lambda: self._ingest(index, f"parts/{fp}/", report)):
                self._owed.clear()
                self._last_full = self._last_live = started
                self._jobs_due = True
        if not report.limited:
            self._chunks(index, fp, live=live, report=report, started=started, budget=budget)
        jobs = lambda: fetch_manifests(self.remote, self.store, "jobs/")  # noqa: E731
        if self._jobs_due and not report.limited and self._attempt(report, jobs):
            self._jobs_due = False
        report.seconds = self.clock() - started
        return report

    def _chunks(
        self,
        index: ProgressIndex,
        fp: str,
        *,
        live: set[str] | None,
        report: PullReport,
        started: float,
        budget: float,
    ) -> None:
        """Per-chunk listings: the owed chunks, plus the live ones when they are due."""
        todo = set(self._owed)
        live_due = started - (self._last_live or 0.0) >= self.live_interval
        if live is not None and self._last_full is not None and live_due:
            todo |= live
            self._last_live = started
        for chunk in sorted(todo):
            if self.clock() - started >= budget:
                report.deferred += 1
                continue
            if self._attempt(
                report, lambda c=chunk: self._ingest(index, f"parts/{fp}/{c}/", report)
            ):
                self._owed.discard(chunk)
            if report.limited:
                break

    def _attempt(self, report: PullReport, listing: Callable[[], object]) -> bool:
        """Run one listing; a rate-limit that outlasts the short budget ends the pull."""
        try:
            listing()
        except Exception as exc:
            if rate_limit_headers(exc) is None:
                raise
            report.limited = True
            report.failed += 1
            return False
        report.listings += 1
        return True

    def _ingest(self, index: ProgressIndex, prefix: str, report: PullReport) -> None:
        new = fetch_manifests(self.ingest_remote, self.store, prefix, seen=index.has_seen)
        report.manifests += index.ingest(self.store.path(p) for p in new)
        for p in new:  # the hashes are in the index: keep nothing on the scarce SSD
            self.store.path(p).unlink(missing_ok=True)
