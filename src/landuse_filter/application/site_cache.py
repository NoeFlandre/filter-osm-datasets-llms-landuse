"""One view of each site per cycle, shared by several controllers (issue #44)."""

import threading

from landuse_filter.adapters import g5k


class SiteCache:
    """Memoises ``oarstat`` and the status API per site until :meth:`reset`.

    A submission on a site invalidates that site's job list, so per-site caps seen by
    the next controller in the same cycle include the new job.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, list[g5k.Job]] = {}
        self._status: dict[str, dict] = {}
        self._lock = threading.Lock()  # submission workers invalidate sites concurrently

    def reset(self) -> None:
        self._jobs.clear()
        self._status.clear()

    def our_jobs(self, site: str) -> list[g5k.Job]:
        with self._lock:
            known = self._jobs.get(site)
        if known is None:
            known = g5k.our_jobs(site)
            with self._lock:
                self._jobs[site] = known
        return list(known)

    def site_status(self, site: str) -> dict:
        if site not in self._status:
            self._status[site] = g5k.site_status(site)
        return self._status[site]

    def invalidate(self, site: str) -> None:
        with self._lock:
            self._jobs.pop(site, None)
