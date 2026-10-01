import pytest
from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.domain.policy_check import AFTER, BEFORE, MODES, PER_BATCH, PER_JOB, check_due


@given(st.sampled_from([BEFORE, AFTER]), st.booleans())
def test_per_job_always_checks(phase, checked):
    assert check_due(PER_JOB, phase, checked_before=checked)


@pytest.mark.parametrize(
    ("phase", "checked", "due"),
    [
        (BEFORE, False, True),  # first submission to the site this cycle
        (BEFORE, True, False),  # later submissions ride on the same check
        (AFTER, True, True),  # the batch ended: validate it
        (AFTER, False, False),  # nothing was submitted: nothing to validate
    ],
)
def test_per_batch_checks_once_before_and_once_after(phase, checked, due):
    assert check_due(PER_BATCH, phase, checked_before=checked) is due


def test_modes_are_the_two_documented_values():
    assert MODES == ("per-job", "per-batch")
