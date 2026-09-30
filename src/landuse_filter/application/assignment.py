"""The controller's ledger entry for one job, and its on-disk JSON form.

Files under ``assignments/`` are read by the controller, ``luf status`` and the node
runner, and older files lack keys added later (``kind``, ``window``, ``late_after_s``,
``bucket``...). ``Assignment.from_json`` therefore tolerates every key being absent,
and keeps keys it does not know in ``extra`` so a read-modify-write never drops them.
"""

from dataclasses import dataclass, field, fields
from typing import Any

# Keys omitted from the JSON while unset, so files written before they existed round-trip.
_OPTIONAL = ("window", "late_after_s", "bucket", "submitted_at", "error")


@dataclass
class Assignment:
    id: str = ""
    name: str = ""
    site: str = ""
    cluster: str = ""
    gpu: str = ""
    chunks: list[str] = field(default_factory=list)
    kind: str = "work"
    fp: str = ""
    state: str = ""
    job_id: str | None = None
    window: int | None = None
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    walltime_s: int = 0
    late_after_s: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    bucket: str | None = None
    submitted_at: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Assignment":
        known = {f.name for f in fields(cls)} - {"extra"}
        return cls(
            **{k: v for k, v in data.items() if k in known},
            extra={k: v for k, v in data.items() if k not in known},
        )

    def late_tolerance(self, default: float) -> float:
        """Seconds a waiting job may slip past its predicted start (old files lack it)."""
        return default if self.late_after_s is None else self.late_after_s

    def to_json(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        for key in _OPTIONAL:
            if out[key] is None:
                del out[key]
        return {**self.extra, **out}


@dataclass
class CycleReport:
    """What one controller cycle saw and did (emitted as a JSON line by the loops)."""

    pending_chunks: int
    live_jobs: int
    submitted: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "pending_chunks": self.pending_chunks,
            "live_jobs": self.live_jobs,
            "submitted": self.submitted,
        }
