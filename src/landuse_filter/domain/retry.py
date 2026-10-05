"""When and how long to wait after a Hub rate limit (HTTP 429, ADR-0028, pure).

The Hub allows 1000 API calls per 5 minutes per account and answers 429 with a
``Retry-After`` header. A caller waits that long (or backs off exponentially with jitter when
the header is missing or unreadable), at most ``MAX_ATTEMPTS`` times and ``MAX_TOTAL_WAIT``
seconds in all, then gives up and re-raises the original error.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

MAX_ATTEMPTS = 10  # calls in all: the first one plus nine retries
MAX_DELAY = 300.0  # one wait never exceeds one rate-limit window
MAX_TOTAL_WAIT = 1200.0  # all waits together: 20 minutes
BASE_DELAY = 2.0
_SECONDS = re.compile(r"(\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class RetryBudget:
    """How patiently one call waits out 429s: attempts, longest wait, total wait."""

    max_attempts: int = MAX_ATTEMPTS
    max_delay: float = MAX_DELAY
    max_total_wait: float = MAX_TOTAL_WAIT


LONG = RetryBudget()  # uploads, node and publish jobs: the work must go through
SHORT = RetryBudget(max_attempts=3, max_delay=60.0, max_total_wait=90.0)  # ingest (ADR-0031)


def parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Seconds named by a ``Retry-After`` header (any case), or ``None`` if absent or unreadable."""
    if not headers:
        return None
    for key, value in headers.items():
        if key.lower() == "retry-after":
            match = _SECONDS.fullmatch(str(value).strip())
            return float(match.group(1)) if match else None
    return None


def next_delay(
    attempt: int,
    retry_after: float | None,
    jitter: float,
    budget: RetryBudget = LONG,
) -> float:
    """Seconds to wait after the failed call number ``attempt`` (1-based).

    ``jitter`` is a uniform draw in [0, 1). The server's value wins (plus 1 s of margin and up to
    1 s of jitter, so that waiting jobs do not return together); otherwise exponential backoff
    with jitter. Always between 0 and ``budget.max_delay``.
    """
    if retry_after is not None:
        delay = max(retry_after, 0.0) + 1.0 + jitter
    else:
        delay = BASE_DELAY * 2 ** (attempt - 1) * (0.5 + jitter / 2)
    return min(delay, budget.max_delay)


def give_up(attempt: int, waited: float, delay: float, budget: RetryBudget = LONG) -> bool:
    """True when failed call number ``attempt`` must not be retried after ``delay`` more seconds."""
    return attempt >= budget.max_attempts or waited + delay > budget.max_total_wait
