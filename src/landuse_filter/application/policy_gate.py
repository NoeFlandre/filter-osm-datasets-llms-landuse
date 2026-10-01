"""Runs ``usagepolicycheck -t`` around submissions at the cadence of ADR-0019.

``per-job``: before and after every submission (the original behaviour; a failure raises
``RemoteError`` into the submission). ``per-batch``: once per site before its first
submission of a cycle and once after its last; a failed check blocks the site for the rest
of the cycle, and a violation found after the batch is logged loudly.
"""

import threading
from collections.abc import Callable

from landuse_filter.adapters import g5k
from landuse_filter.domain.policy_check import AFTER, BEFORE, PER_BATCH, check_due


class PolicyGate:
    def __init__(self, mode: str, log: Callable[[str], None]) -> None:
        self.mode = mode
        self.log = log
        self._lock = threading.Lock()
        self._checked: set[str] = set()
        self._blocked: set[str] = set()

    def begin_cycle(self) -> None:
        with self._lock:
            self._checked.clear()
            self._blocked.clear()

    def blocked(self, site: str) -> bool:
        with self._lock:
            return site in self._blocked

    def before(self, site: str) -> None:
        """Check ``site`` before an oarsub if due; raises ``RemoteError`` when it fails."""
        if self.blocked(site):
            raise g5k.RemoteError(f"{site}: policy check failed earlier in this cycle")
        with self._lock:
            due = check_due(self.mode, BEFORE, checked_before=site in self._checked)
        if not due:
            return
        try:
            g5k.policy_check(site)
        except g5k.RemoteError:
            if self.mode == PER_BATCH:  # per-job: only this submission fails, as before
                with self._lock:
                    self._blocked.add(site)
            raise
        with self._lock:
            self._checked.add(site)

    def after_job(self, site: str) -> None:
        """Per-job mode only: validate the submission just made (raises on failure)."""
        if self.mode != PER_BATCH:
            g5k.policy_check(site)

    def finish(self, site: str) -> None:
        """Per-batch: validate the site's whole batch; a violation is logged, never hidden."""
        if self.mode != PER_BATCH:
            return
        with self._lock:
            due = check_due(self.mode, AFTER, checked_before=site in self._checked)
            self._checked.discard(site)  # finished: finish_all() must not recheck it
        if not due:
            return
        try:
            g5k.policy_check(site)
        except g5k.RemoteError as exc:
            with self._lock:
                self._blocked.add(site)
            self.log(f"POLICY VIOLATION: {site}: check after this cycle's batch failed: {exc}")

    def finish_all(self) -> None:
        with self._lock:
            sites = sorted(self._checked)
        for site in sites:
            self.finish(site)
