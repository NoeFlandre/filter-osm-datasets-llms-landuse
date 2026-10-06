"""Grid'5000 default-queue usage policy as pure rules (Europe/Paris local time).

Source: https://www.grid5000.fr/w/Grid5000:UsagePolicy (read 2026-09-28).

- Working days 09:00-19:00 are daytime; jobs must not cross 09:00/19:00 on working
  days (crossing 19:00 is allowed for jobs submitted at/after 17:00).
- Daytime jobs of <= 1 h submitted < 10 min before they start are exempt from the
  daily quota. We only ever submit those during the day, so we never consume quota.
- Nights and weekends allow longer jobs, ending before the next working-day 09:00.

``usagepolicycheck -t`` on the frontend remains the authority; these rules only keep
the controller from proposing anything it would reject.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

DAY_START = time(9, 0)
DAY_END = time(19, 0)
SATURDAY = 5
DAY_EXEMPT_MAX = timedelta(hours=1)


def easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def french_holidays(year: int) -> frozenset[date]:
    e = easter(year)
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    moving = [e + timedelta(days=1), e + timedelta(days=39), e + timedelta(days=50)]
    return frozenset([date(year, m, d) for m, d in fixed] + moving)


def is_working_day(day: date) -> bool:
    return day.weekday() < SATURDAY and day not in french_holidays(day.year)


def is_daytime(moment: datetime) -> bool:
    return is_working_day(moment.date()) and DAY_START <= moment.time() < DAY_END


def next_day_start(moment: datetime) -> datetime:
    """The next working-day 09:00 strictly after ``moment``'s current window."""
    day = moment.date() if moment.time() < DAY_START else moment.date() + timedelta(days=1)
    while not is_working_day(day):
        day += timedelta(days=1)
    return datetime.combine(day, DAY_START, tzinfo=moment.tzinfo)


@dataclass(frozen=True, slots=True)
class Window:
    """Longest walltime we may request for a job starting now, and its OAR type."""

    max_walltime: timedelta
    job_type: str | None  # "night" outside daytime, None for exempt daytime jobs


def allowed_window(now: datetime, *, starts_now: bool) -> Window | None:
    """Policy-compliant window for a job submitted at ``now``, or ``None``.

    During daytime only immediately-starting jobs of <= 1 h are proposed (quota
    exempt). Otherwise the job must end before the next working-day 09:00.
    """
    if is_daytime(now):
        if not starts_now:
            return None
        # A <= 1 h job started before 17:00 ends before 18:00; one submitted at or
        # after 17:00 may cross 19:00. Either way 1 h never breaks the 19:00 rule.
        return Window(DAY_EXEMPT_MAX, None)
    return Window(next_day_start(now) - now, "night")


def immediate_window(window: Window | None) -> Window | None:
    """The quota-free immediate-start window inside a night/weekend ``window`` (ADR-0034).

    A job without the ``night`` type and of <= 1 h that ends before the next working-day
    09:00 breaks no rule outside daytime. ``None`` for a daytime or missing window (the
    daytime window already is immediate-start).
    """
    if window is None or window.job_type is None:
        return None
    return Window(min(window.max_walltime, DAY_EXEMPT_MAX), None)
