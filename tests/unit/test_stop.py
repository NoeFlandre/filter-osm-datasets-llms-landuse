import json

from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.domain.stop import (
    STOP_MARGIN_SECONDS,
    deadline_from_env,
    deadline_from_oarstat,
    should_stop,
    stop_reason,
)

times = st.floats(min_value=0, max_value=1e10, allow_nan=False)


def test_stops_exactly_at_the_margin():
    assert not should_stop(999, 1000 + STOP_MARGIN_SECONDS)
    assert should_stop(1000, 1000 + STOP_MARGIN_SECONDS)
    assert should_stop(5000, 1000)  # past the deadline


def test_no_deadline_never_stops():
    assert not should_stop(1e12, None)


@given(times, times, times)
def test_stopping_is_monotone_in_time(now, later_by, deadline):
    if should_stop(now, deadline):
        assert should_stop(now + later_by, deadline)


@given(times, times, st.floats(min_value=0, max_value=1e4))
def test_a_larger_margin_stops_earlier(now, deadline, margin):
    if should_stop(now, deadline, margin):
        assert should_stop(now, deadline, margin + 1)


@given(times, st.one_of(st.none(), times), st.one_of(st.none(), st.just("SIGUSR2")))
def test_reason_is_present_iff_a_cause_exists(now, deadline, signalled):
    reason = stop_reason(now, deadline, signalled)
    assert (reason is not None) == (bool(signalled) or should_stop(now, deadline))


def test_signal_wins_over_deadline():
    assert stop_reason(10, 10, "SIGTERM") == "signal SIGTERM"
    assert stop_reason(10, 10, None) == "deadline"
    assert stop_reason(0, 1e9, None) is None


def test_deadline_from_env():
    assert deadline_from_env({"LUF_JOB_DEADLINE_EPOCH": "1700000000"}) == 1700000000
    for bad in ({}, {"LUF_JOB_DEADLINE_EPOCH": ""}, {"LUF_JOB_DEADLINE_EPOCH": "x"}):
        assert deadline_from_env(bad) is None
    assert deadline_from_env({"LUF_JOB_DEADLINE_EPOCH": "0"}) is None
    assert deadline_from_env({"LUF_JOB_DEADLINE_EPOCH": "-5"}) is None


def test_deadline_from_oarstat_uses_start_time_when_running():
    text = json.dumps({"123": {"walltime": 3600, "startTime": 1000}})
    assert deadline_from_oarstat(text, now=2000) == 4600


def test_deadline_from_oarstat_falls_back_to_now_and_flat_shape():
    assert deadline_from_oarstat(json.dumps({"walltime": 60, "startTime": 0}), 500) == 560
    assert deadline_from_oarstat(json.dumps({"1": {"walltime": 60}}), 500) == 560


def test_deadline_from_oarstat_ignores_garbage():
    for bad in ("", "nope", "{}", "[]", json.dumps({"1": {"walltime": 0}}), json.dumps({"1": {}})):
        assert deadline_from_oarstat(bad, 1) is None
