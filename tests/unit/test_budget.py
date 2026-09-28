import pytest

from landuse_filter.domain.budget import Generated, predicted_under_cap, truncation_rate


def test_generation_within_cap_is_unchanged():
    assert predicted_under_cap(Generated("yes", 1000), 1000) == "yes"


def test_generation_over_cap_becomes_failed():
    assert predicted_under_cap(Generated("yes", 1001), 1000) is None


def test_truncation_rate():
    assert truncation_rate([Generated("yes", 10), Generated(None, 4096)], 2048) == 0.5
    with pytest.raises(ValueError, match="no generations"):
        truncation_rate([], 1)
