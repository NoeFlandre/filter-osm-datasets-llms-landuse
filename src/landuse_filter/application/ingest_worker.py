"""A background thread that ingests results so that ingest can never delay submission.

The Hub rate limit (ADR-0028) makes any listing able to wait; run in the controller's cycle
it starved the GPUs (ADR-0031). The worker owns its own progress index connection (SQLite
connections belong to one thread), wakes every ``interval`` seconds, runs one bounded pull
and logs a line with its counts and duration. The cycle never waits for it.
"""

import threading
from collections.abc import Callable


class BackgroundIngest:
    def __init__(
        self,
        pull: Callable[[], str],
        *,
        interval: float,
        log: Callable[[str], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self._pull = pull  # runs one pull in the worker thread, returns its log line
        self.interval = interval
        self.log = log
        self.on_error = on_error
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the thread once; later calls are no-ops."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="ingest", daemon=True)
            self._thread.start()

    def kick(self) -> None:
        """Pull again now rather than at the end of the interval."""
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.once()
            self._wake.wait(self.interval)
            self._wake.clear()

    def once(self) -> None:
        try:
            self.log(self._pull())
        except Exception as exc:  # noqa: BLE001 - a failed pull is retried at the next interval
            self.on_error(exc)
