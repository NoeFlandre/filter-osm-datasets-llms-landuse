from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.adapters.g5k import Job
from landuse_filter.application.cpu_guard import CpuJobCancelledError, guard_start, live, sweep
from landuse_filter.domain.cpu_job_guard import (
    cpu_job_verdict,
    crosses_forbidden,
    is_zombie,
    stale_day_job,
)
from landuse_filter.domain.policy import is_working_day

PARIS = ZoneInfo("Europe/Paris")
H = timedelta(hours=1)


def at(d, h, mi=0, m=10, y=2026):
    return datetime(y, m, d, h, mi, tzinfo=PARIS)


def verdict(sub, start, wall=H):
    return cpu_job_verdict(submitted=sub, walltime=wall, expected_start=start).allowed


def test_incident_is_refused():
    # Thu 2026-10-01 16:40, started 18:30 -> ran to 19:25
    assert not verdict(at(1, 16, 40, 10), at(1, 18, 30, 10))


@pytest.mark.parametrize(
    ("sub", "start", "ok"),
    [
        (at(1, 10), at(1, 10, 5), True),
        (at(1, 10), at(1, 10, 11), False),  # late
        (at(1, 16, 59), at(1, 17, 5), True),  # ends 18:05
        (at(1, 16, 59), at(1, 17, 9), True),
        (at(1, 17, 0), at(1, 17, 5), True),
        (at(1, 18, 0), at(1, 18, 5), True),  # evening submit, may cross 19:00
        (at(1, 18, 59), at(1, 18, 59), True),  # ends 19:59, submitted after 17:00
        (at(1, 16, 59), at(1, 18, 30), False),  # late AND crossing
        (at(1, 19, 0), at(1, 19, 0), True),  # night starts 19:00 sharp
        (at(1, 19, 0), at(2, 8, 30), False),  # ends 09:30 on a working day
        (at(1, 8, 59), at(1, 8, 59), False),  # crosses 09:00
        (at(1, 8, 0), at(1, 8, 0), True),
        (at(1, 9, 0), at(1, 9, 0), True),
        (at(3, 10), at(3, 22), True),  # Saturday, no boundary
        (at(2, 21), at(2, 21), True),  # Friday night
    ],
)
def test_boundaries(sub, start, ok):
    assert verdict(sub, start) is ok


def test_night_job_ending_exactly_at_nine_is_ok():
    assert verdict(at(1, 20), at(2, 8, 0))


def test_weekend_and_holiday_have_no_boundary():
    assert verdict(at(3, 8, 30), at(3, 8, 30))  # Saturday
    assert verdict(
        datetime(2026, 5, 1, 18, 30, tzinfo=PARIS), datetime(2026, 5, 1, 18, 30, tzinfo=PARIS)
    )
    assert verdict(
        datetime(2026, 5, 1, 8, 59, tzinfo=PARIS), datetime(2026, 5, 1, 8, 59, tzinfo=PARIS)
    )


def test_friday_night_to_monday_crossing():
    assert verdict(at(2, 20), at(2, 20), timedelta(hours=60))  # until Monday 08:00
    assert not verdict(at(2, 20), at(2, 20), timedelta(hours=62))  # crosses Monday 09:00


def test_unknown_start():
    assert not verdict(at(1, 10), None)
    assert verdict(at(1, 22), None)


def test_stale_day_job_and_watchdog():
    assert not stale_day_job(submitted=at(1, 10), now=at(1, 10, 9))
    assert stale_day_job(submitted=at(1, 10), now=at(1, 10, 11))
    assert not stale_day_job(submitted=at(1, 16, 45), now=at(1, 16, 49))
    assert stale_day_job(submitted=at(1, 16, 45), now=at(1, 16, 50))
    assert not stale_day_job(submitted=at(1, 17, 0), now=at(1, 20))
    assert not stale_day_job(submitted=at(1, 22), now=at(2, 6))  # night job
    assert not stale_day_job(submitted=at(3, 10), now=at(3, 12))  # weekend


def test_zombie():
    assert is_zombie(state="Waiting", submitted=at(1, 10), now=at(1, 12, 1))
    assert not is_zombie(state="Waiting", submitted=at(1, 10), now=at(1, 11, 59))
    assert not is_zombie(state="Running", submitted=at(1, 10), now=at(1, 20))
    assert not is_zombie(state="Waiting", submitted=None, now=at(1, 20))


def test_crosses_forbidden_edges():
    assert not crosses_forbidden(at(1, 10), at(1, 17), at(1, 19))
    assert crosses_forbidden(at(1, 10), at(1, 18, 30), at(1, 19, 25))
    assert not crosses_forbidden(at(1, 17), at(1, 18, 30), at(1, 19, 25))
    assert not crosses_forbidden(at(1, 17), at(1, 19), at(1, 20))


minutes = st.integers(min_value=0, max_value=365 * 24 * 60 - 1)


@given(
    sub_min=minutes,
    delay=st.integers(min_value=0, max_value=4 * 24 * 60),
    wall=st.integers(min_value=1, max_value=70 * 60),
)
def test_accepted_job_never_crosses(sub_min, delay, wall):
    sub = datetime(2026, 1, 1, tzinfo=PARIS) + timedelta(minutes=sub_min)
    start = sub + timedelta(minutes=delay)
    end = start + timedelta(minutes=wall)
    if not cpu_job_verdict(submitted=sub, walltime=end - start, expected_start=start).allowed:
        return
    # independent oracle: every whole minute strictly inside (start, end)
    for k in range(1, wall):
        t = start + timedelta(minutes=k)
        if not is_working_day(t.date()) or t.minute != 0:
            continue
        assert t.hour != 9
        if t.hour == 19:
            assert sub.date() == t.date()
            assert sub.hour >= 17


def test_guard_start_cancels_late_job():
    cancelled = []
    with pytest.raises(CpuJobCancelledError, match="would start late; cancelled"):
        guard_start(
            job_id="5",
            submitted=at(1, 16, 40),
            walltime=H,
            now=lambda: at(1, 16, 41),
            status=lambda j: ("Waiting", int(at(1, 18, 30).timestamp())),
            cancel=cancelled.append,
            sleep=lambda s: None,
        )
    assert cancelled == ["5"]


def test_guard_start_keeps_running_and_soon_jobs():
    for status in (("Running", None), ("Waiting", int(at(1, 10, 3).timestamp()))):
        cancelled = []
        guard_start(
            job_id="5",
            submitted=at(1, 10),
            walltime=H,
            now=lambda: at(1, 10, 1),
            status=lambda j, s=status: s,
            cancel=cancelled.append,
            sleep=lambda s: None,
        )
        assert cancelled == []


def test_guard_start_unknown_start_in_daytime_cancels():
    cancelled = []
    with pytest.raises(CpuJobCancelledError):
        guard_start(
            job_id="5",
            submitted=at(1, 10),
            walltime=H,
            now=lambda: at(1, 10),
            status=lambda j: ("Waiting", None),
            cancel=cancelled.append,
            sleep=lambda s: None,
        )
    assert cancelled == ["5"]


def job(i, name="luf-publish-x", state="Waiting", sub=None):
    return Job("lyon", str(i), name, state, "default", None, sub and int(sub.timestamp()))


def test_sweep_cancels_only_own_stale_waiting_jobs():
    cancelled, logs = [], []
    jobs = [
        job(1, sub=at(1, 10)),  # stale
        job(2, sub=at(1, 10, 55)),  # fresh
        job(3, name="luf-other", sub=at(1, 10)),  # not ours
        job(4, state="Running", sub=at(1, 10)),
        job(5, sub=None),
    ]
    sweep(jobs, "luf-publish-x", now=at(1, 11), cancel=cancelled.append, log=logs.append)
    assert cancelled == ["1"]
    assert len(logs) == 1


def test_live_ignores_zombies():
    now = at(1, 15)
    assert live([job(1, sub=at(1, 14))], "luf-publish-x", now=now)
    assert not live([job(1, sub=at(1, 12))], "luf-publish-x", now=now)
    assert not live([job(1, name="o", sub=at(1, 14))], "luf-publish-x", now=now)
    assert live([job(1, sub=None)], "luf-publish-x", now=now)


def test_run_loop_sweeps_every_cycle():
    from landuse_filter.application.publish_loop import LoopIO, run_loop

    swept = []
    io = LoopIO(
        status=lambda: {"done": True, "revision": "r"},
        live=lambda: False,
        submit=lambda: "1",
        sleep=lambda s: None,
        log=lambda m: None,
        sweep=lambda: swept.append(1),
    )
    assert run_loop(io, "r", interval=1, cap=1, max_cycles=2) == "finished"
    assert swept == [1]


def reason(sub, start, wall=H):
    return cpu_job_verdict(submitted=sub, walltime=wall, expected_start=start).reason


def test_reasons_and_exact_tolerance():
    assert reason(at(1, 10), None) == "would start late: no predicted start in daytime"
    assert reason(at(1, 22), None) == "night job, start left to the night type"
    assert reason(at(1, 10), at(1, 10, 10)) == "ok"
    assert reason(at(1, 10), at(1, 10, 10) + timedelta(seconds=1)) == "would start late"
    assert reason(at(1, 8, 59), at(1, 8, 59)) == "would cross a day/night boundary"
    assert reason(at(1, 20), at(1, 20)) == "ok"
    # a night submission is not subject to the 10 minute start tolerance
    assert verdict(at(1, 20), at(1, 23))
    assert verdict(at(3, 10), at(3, 15))  # weekend daytime hours


def test_exact_edges_do_not_cross():
    assert not crosses_forbidden(at(1, 10), at(1, 8), at(1, 9))  # ends at 09:00
    assert not crosses_forbidden(at(1, 10), at(1, 9), at(1, 10))  # starts at 09:00
    assert crosses_forbidden(at(1, 10), at(1, 8, 59), at(1, 9, 1))
    assert not crosses_forbidden(at(1, 10), at(1, 18), at(1, 19))
    assert not crosses_forbidden(at(1, 10), at(1, 19), at(1, 20))
    assert crosses_forbidden(at(1, 10), at(1, 18, 59), at(1, 19, 1))
    assert crosses_forbidden(at(1, 16, 59), at(1, 18, 59), at(1, 19, 1))
    assert not crosses_forbidden(at(1, 17), at(1, 18, 59), at(1, 19, 1))
    # a 17:00 submission of ANOTHER day does not grant today's crossing
    assert crosses_forbidden(at(30, 17, m=9), at(1, 18, 59), at(1, 19, 1))
    # an evening job may cross 19:00 but never the next 09:00
    assert crosses_forbidden(at(1, 17), at(1, 18), at(2, 10))


def test_zombie_and_stale_exact_limits():
    assert not is_zombie(state="Waiting", submitted=at(1, 10), now=at(1, 12))
    assert is_zombie(state="Waiting", submitted=at(1, 10), now=at(1, 12) + timedelta(seconds=1))
    assert not stale_day_job(submitted=at(1, 10), now=at(1, 10, 10))
    assert stale_day_job(submitted=at(1, 10), now=at(1, 10, 10) + timedelta(seconds=1))
    assert not stale_day_job(submitted=at(1, 16, 45), now=at(1, 16, 49))
    assert stale_day_job(submitted=at(1, 16, 59), now=at(1, 17, 5)) is True
    assert not stale_day_job(submitted=at(1, 17), now=at(1, 17, 30))
    assert stale_day_job(submitted=at(1, 16, 59), now=at(1, 17, 11))
    assert not stale_day_job(submitted=at(1, 8, 55), now=at(1, 9, 5))  # night job submitted early
    assert stale_day_job(submitted=at(1, 16, 40), now=at(2, 9, 0))
