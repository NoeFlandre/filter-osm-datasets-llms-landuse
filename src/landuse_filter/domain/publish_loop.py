"""Keep publishing until done (ADR-0023, pure).

A publish job ends at its walltime; ``luf g5k publish-loop`` submits the next one whenever none
of the dataset is live on the site, until the job's own status marker in the bucket says
nothing is left.
"""

from dataclasses import dataclass
from typing import Final

SUBMIT: Final = "submit"
WAIT: Final = "wait"
FINISHED: Final = "finished"

NAME_LENGTH = 20  # dataset characters kept in an OAR job name


def job_name(prefix: str, mode: str, dataset: str) -> str:
    """OAR name of a CPU job. Publish jobs get their own name so a planning loop (which looks
    for ``plan``) never mistakes them for planning; every other mode keeps the ``plan`` name."""
    kind = "publish" if mode == "publish" else "plan"
    return f"{prefix}{kind}-{dataset[:NAME_LENGTH]}"


@dataclass(frozen=True, slots=True)
class PublishStatus:
    dataset: str
    revision: str
    files_total: int
    files_complete: int
    files_partial: int
    mirrored: bool
    unscanned: int
    stopped: str | None

    @property
    def done(self) -> bool:
        """Nothing is left: mirrored, planner index fully scanned, every file labelled, and the
        run was not cut short (a cut-short run may still owe viewer tables)."""
        return (
            self.mirrored
            and self.unscanned == 0
            and self.files_total > 0
            and self.files_complete == self.files_total
            and self.stopped is None
        )

    def to_json(self) -> dict:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "files": {
                "total": self.files_total,
                "complete": self.files_complete,
                "partial": self.files_partial,
                "unscanned": self.unscanned,
            },
            "mirrored": self.mirrored,
            "stopped": self.stopped,
            "done": self.done,
        }


def is_done(status: object, revision: str) -> bool:
    """True when the marker of *this* revision says done (a marker of another revision, or a
    malformed one, never ends the loop)."""
    if not isinstance(status, dict):
        return False
    return status.get("done") is True and status.get("revision") == revision


def decide(*, done: bool, live: bool) -> str:
    """What the loop does next: stop, wait for the live job, or submit a new one."""
    if done:
        return FINISHED
    return WAIT if live else SUBMIT


def pause(failures: int, interval: float, cap: float) -> float:
    """Seconds to sleep: ``interval`` while all goes well, doubling per consecutive failure
    (slow frontend, timeouts) up to ``cap``."""
    return min(interval * 2 ** min(max(failures, 0), 30), cap)
