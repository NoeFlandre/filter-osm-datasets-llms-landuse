from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from landuse_filter.domain.policy import allowed_window, easter, french_holidays, is_daytime

PARIS = ZoneInfo("Europe/Paris")


def at(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=PARIS)


def test_easter_and_holidays():
    assert easter(2026) == date(2026, 4, 5)
    assert date(2026, 4, 6) in french_holidays(2026)  # Easter Monday
    assert date(2026, 5, 14) in french_holidays(2026)  # Ascension


def test_daytime_boundaries():
    assert is_daytime(at(2026, 9, 28, 9))  # Monday
    assert not is_daytime(at(2026, 9, 28, 19))
    assert not is_daytime(at(2026, 9, 26, 12))  # Saturday
    assert not is_daytime(at(2026, 11, 11, 12))  # Armistice


def test_daytime_only_immediate_one_hour_jobs():
    assert allowed_window(at(2026, 9, 28, 10), starts_now=False) is None
    w = allowed_window(at(2026, 9, 28, 10), starts_now=True)
    assert w.max_walltime == timedelta(hours=1)
    assert w.job_type is None


def test_daytime_one_hour_jobs_are_always_policy_safe():
    assert allowed_window(at(2026, 9, 28, 16, 30), starts_now=True).max_walltime == timedelta(
        hours=1
    )
    late = allowed_window(at(2026, 9, 28, 18, 30), starts_now=True)
    assert late.max_walltime == timedelta(hours=1)


@pytest.mark.parametrize(
    ("now", "hours"),
    [(at(2026, 9, 28, 20), 13), (at(2026, 9, 29, 3), 6), (at(2026, 10, 2, 20), 61)],
)
def test_night_and_weekend_end_before_next_working_morning(now, hours):
    w = allowed_window(now, starts_now=False)
    assert w.job_type == "night"
    assert w.max_walltime == timedelta(hours=hours)
