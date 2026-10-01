"""What the controller has learned about clusters: back-offs and besteffort-only access."""

import threading
from datetime import datetime, timedelta

from landuse_filter.adapters.store import WorkStore
from landuse_filter.domain.capacity import Cluster
from landuse_filter.domain.scheduling import long_attempt_tripped


class ClusterMemory:
    def __init__(self, store: WorkStore) -> None:
        self.store = store
        self._lock = threading.Lock()  # sites submitted in parallel record concurrently

    def back_off(self, site: str, cluster: str, until: datetime) -> None:
        self.store.write_json(f"backoff/{site}_{cluster}.json", {"until": until.isoformat()})

    def backed_off(self, cluster: Cluster, now: datetime) -> bool:
        path = f"backoff/{cluster.site}_{cluster.name}.json"
        return (
            self.store.exists(path)
            and datetime.fromisoformat(self.store.read_json(path)["until"]) > now
        )

    def remember_besteffort_only(self, site: str, cluster: str) -> None:
        self.store.write_json(f"access/{site}_{cluster}.json", {"besteffort_only": True})

    def besteffort_only(self, cluster: Cluster) -> bool:
        return self.store.exists(f"access/{cluster.site}_{cluster.name}.json")

    def record_long_attempt(
        self, site: str, cluster: str, *, ok: bool, now: datetime, limit: int, pause: timedelta
    ) -> bool:
        """Count a long-walltime attempt; True when this failure trips the throttle.

        A success, or a failure after an earlier pause has lapsed, restarts the count; the
        ``limit``-th consecutive failure pauses long attempts for ``pause``.
        """
        path = f"longwall/{site}_{cluster}.json"
        with self._lock:  # read-modify-write of the cluster's counter
            failures = self.store.read_json(path)["failures"] if self.store.exists(path) else 0
            failures = 0 if ok else failures + 1
            tripped = not ok and long_attempt_tripped(failures, limit)
            until = (now + pause).isoformat() if tripped else None
            if tripped:
                failures = 0
            self.store.write_json(path, {"failures": failures, "until": until})
        return tripped

    def long_throttled(self, cluster: Cluster, now: datetime) -> bool:
        path = f"longwall/{cluster.site}_{cluster.name}.json"
        if not self.store.exists(path):
            return False
        until = self.store.read_json(path)["until"]
        return until is not None and datetime.fromisoformat(until) > now
