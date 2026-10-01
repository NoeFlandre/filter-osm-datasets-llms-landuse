"""When the usage-policy check (``usagepolicycheck -t``) must run (ADR-0019, pure).

``per-job`` checks around every submission. ``per-batch`` checks a site once before the
first submission of a cycle and once after the last one, which validates the same
submissions with two checks instead of two per job.
"""

from enum import StrEnum

PER_JOB = "per-job"
PER_BATCH = "per-batch"
MODES = (PER_JOB, PER_BATCH)


class PolicyCheck(StrEnum):
    """CLI choices of ``--policy-check``."""

    PER_JOB = PER_JOB
    PER_BATCH = PER_BATCH


BEFORE = "before"
AFTER = "after"


def check_due(mode: str, phase: str, *, checked_before: bool) -> bool:
    """Whether the ``phase`` check of a submission on a site must run now.

    ``checked_before``: the site already had its pre-batch check this cycle. Per-batch,
    the pre-check runs once, and the post-check only for a site that was pre-checked.
    """
    if mode == PER_JOB:
        return True
    return (phase == BEFORE) != checked_before
