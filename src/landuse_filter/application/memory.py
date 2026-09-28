"""What the controller has learned about clusters: back-offs and besteffort-only access."""

from datetime import datetime

from landuse_filter.adapters.store import WorkStore
from landuse_filter.domain.capacity import Cluster


class ClusterMemory:
    def __init__(self, store: WorkStore) -> None:
        self.store = store

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
