"""The walltimes tried for one slot never break the window and never grow."""

from datetime import timedelta

from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.domain.scheduling import walltime_ladder

MINUTES = st.integers(min_value=1, max_value=2000).map(lambda m: timedelta(minutes=m))


@given(window=MINUTES, preferred=MINUTES, fallback=MINUTES, day=MINUTES, night=st.booleans())
def test_walltime_ladder_properties(window, preferred, fallback, day, night):
    ladder = walltime_ladder(
        window_max=window, night=night, day=day, preferred=preferred, fallback=fallback
    )
    assert ladder  # a positive window always allows a job
    assert all(timedelta(0) < w <= window for w in ladder)
    assert list(ladder) == sorted(set(ladder), reverse=True)  # strictly decreasing
    assert night or len(ladder) == 1
    assert ladder[0] == min(preferred if night else day, window)
    assert not night or all(w <= preferred for w in ladder)
