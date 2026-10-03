from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.domain.policy import (
    allowed_window,
    easter,
    french_holidays,
    immediate_window,
    is_daytime,
    next_day_start,
)

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


@pytest.mark.parametrize(
    ("year", "month", "day"),
    [
        (1818, 3, 22),
        (1886, 4, 25),
        (1900, 4, 15),
        (1943, 4, 25),
        (1954, 4, 18),
        (1961, 4, 2),
        (2000, 4, 23),
        (2008, 3, 23),
        (2011, 4, 24),
        (2019, 4, 21),
        (2038, 4, 25),
        (2100, 3, 28),
        (2285, 3, 22),
    ],
)
def test_easter_known_dates(year, month, day):
    assert easter(year) == date(year, month, day)


@given(st.integers(1583, 4099))
def test_easter_is_a_sunday_between_march_22_and_april_25(year):
    e = easter(year)
    assert e.weekday() == 6
    assert date(year, 3, 22) <= e <= date(year, 4, 25)


def test_french_holidays_2026_exactly():
    days = [(1, 1), (4, 6), (5, 1), (5, 8), (5, 14), (5, 25), (7, 14), (8, 15), (11, 1)]
    days += [(11, 11), (12, 25)]
    assert french_holidays(2026) == {date(2026, m, d) for m, d in days}


def test_next_day_start_is_strictly_after_the_current_morning():
    assert next_day_start(at(2026, 9, 28, 9)) == at(2026, 9, 29, 9)
    assert next_day_start(at(2026, 9, 28, 8, 59)) == at(2026, 9, 28, 9)
    assert next_day_start(at(2026, 9, 26, 20)) == at(2026, 9, 28, 9)  # Saturday -> Monday


def knuth_easter(year):
    """Independent oracle: Knuth, TAOCP vol. 1, 1.3.2 exercise 14."""
    golden = year % 19 + 1
    century = year // 100 + 1
    x = 3 * century // 4 - 12
    z = (8 * century + 5) // 25 - 5
    sunday = 5 * year // 4 - x - 10
    epact = (11 * golden + 20 + z - x) % 30
    if (epact == 25 and golden > 11) or epact == 24:
        epact += 1
    n = 44 - epact
    if n < 21:
        n += 30
    n += 7 - (sunday + n) % 7
    return date(year, 4, n - 31) if n > 31 else date(year, 3, n)


def test_easter_matches_independent_algorithm():
    assert all(easter(y) == knuth_easter(y) for y in range(1583, 4100))


# --- immediate-start window inside nights and weekends (ADR-0027) ---

PARIS = ZoneInfo("Europe/Paris")


def test_immediate_window_is_absent_by_day_and_without_a_window():
    day = allowed_window(datetime(2026, 9, 29, 10, 0, tzinfo=PARIS), starts_now=True)
    assert immediate_window(day) is None
    assert immediate_window(None) is None


def test_immediate_window_is_one_hour_without_type_at_night_and_weekend():
    for moment in (
        datetime(2026, 9, 28, 22, 0, tzinfo=PARIS),
        datetime(2026, 10, 3, 14, 0, tzinfo=PARIS),
    ):
        window = immediate_window(allowed_window(moment, starts_now=True))
        assert window is not None
        assert (window.max_walltime, window.job_type) == (timedelta(hours=1), None)


def test_immediate_window_ends_before_the_next_working_day_start():
    early = datetime(2026, 9, 29, 8, 30, tzinfo=PARIS)
    window = immediate_window(allowed_window(early, starts_now=True))
    assert window is not None
    assert window.max_walltime == timedelta(minutes=30)
