"""One view of each site per cycle, shared by several controllers (issue #44)."""

from landuse_filter.adapters import g5k


class SiteCache:
    """Memoises ``oarstat`` and the status API per site until :meth:`reset`.

    A submission on a site invalidates that site's job list, so per-site caps seen by
    the next controller in the same cycle include the new job.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, list[g5k.Job]] = {}
        self._status: dict[str, dict] = {}

    def reset(self) -> None:
        self._jobs.clear()
        self._status.clear()

    def our_jobs(self, site: str) -> list[g5k.Job]:
        if site not in self._jobs:
            self._jobs[site] = g5k.our_jobs(site)
        return list(self._jobs[site])

    def site_status(self, site: str) -> dict:
        if site not in self._status:
            self._status[site] = g5k.site_status(site)
        return self._status[site]

    def invalidate(self, site: str) -> None:
        self._jobs.pop(site, None)
