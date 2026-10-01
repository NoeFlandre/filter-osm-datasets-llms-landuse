"""Wall-clock stamps for a job record that cannot jump backwards.

The anchor is one wall-clock reading plus the monotonic clock; every later stamp is
anchor + monotonic elapsed, so an NTP step during the job never produces a negative duration.
"""

import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any


class Timeline:
    def __init__(self) -> None:
        self._wall = datetime.now(UTC)
        self._mono = time.monotonic()

    def at(self, mono: float | None) -> str | None:
        """ISO UTC string of a ``time.monotonic()`` instant (``None`` stays ``None``)."""
        if mono is None:
            return None
        return (self._wall + timedelta(seconds=mono - self._mono)).isoformat(timespec="seconds")

    def now(self) -> str:
        return (self._wall + timedelta(seconds=time.monotonic() - self._mono)).isoformat(
            timespec="seconds"
        )

    @property
    def started_mono(self) -> float:
        return self._mono


def env_ready_seconds(environ: Mapping[str, str]) -> int | None:
    """Seconds node_job.sh spent building the environment (``None`` if unset or garbled)."""
    text = environ.get("LUF_ENV_READY_SECONDS", "")
    return int(text) if text.isdigit() else None


def timeline_record(
    tl: Timeline,
    *,
    engine_ready: float,
    first_result: float | None,
    last_result: float | None,
    walltime_s: int,
    assigned_texts: int,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """The timeline keys of ``jobs/<site>/<job>.json`` (old records simply lack them)."""
    return {
        "started_at": tl.at(tl.started_mono),
        "env_ready_seconds": env_ready_seconds(environ),
        "engine_ready_at": tl.at(engine_ready),
        "first_result_at": tl.at(first_result),
        "last_result_at": tl.at(last_result),
        "ended_at": tl.now(),
        "walltime_s": walltime_s,
        "assigned_texts": assigned_texts,
    }
