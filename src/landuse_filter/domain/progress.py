"""Pure progress arithmetic: ETA and drift alerts for `luf status`."""

from collections.abc import Sequence
from dataclasses import dataclass

# The benchmark reference fails to parse 2.42 % of items; well above that means
# something changed (truncation, template, GPU) and the gate no longer describes us.
FAILED_RATE_ALERT = 0.05


@dataclass(frozen=True, slots=True)
class Eta:
    remaining: int
    sentences_per_second: float

    @property
    def hours(self) -> float | None:
        if self.remaining == 0:
            return 0.0
        return (
            self.remaining / self.sentences_per_second / 3600
            if self.sentences_per_second > 0
            else None
        )


def eta(remaining: int, live_rates: Sequence[float]) -> Eta:
    """Time to finish at the aggregate rate of the jobs running now (None if none)."""
    return Eta(remaining, sum(live_rates))


def alerts(completed: int, failed: int) -> list[str]:
    if completed and failed / completed > FAILED_RATE_ALERT:
        return [f"failed rate {failed / completed:.1%} > {FAILED_RATE_ALERT:.0%} (benchmark 2.4%)"]
    return []
