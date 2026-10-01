# ADR-0017: Long day jobs with a short fallback and a self-throttle

Status: accepted (2026-10-01)

## Context

By day (09:00-19:00) the usage policy allows only jobs that start immediately and last at most
1 hour. About 6 of every 30 minutes of a job are startup, so a 60-minute day job raises useful
generation time from about 80 % to about 90 %. A long day job is harder to place, though, and
the owner's requirement is that trying it must never slow us down.

## Decision

- `--day-walltime-minutes` (luf.toml `day_walltime_minutes`, `LUF_DAY_WALLTIME_MINUTES`) is the
  preferred day walltime; unset, it equals `--walltime-minutes`, which changes nothing.
- `domain/scheduling.walltime_ladder` takes `day_short`: by day the rungs are
  `min(day, window max)` then `min(day_short, day)`; duplicates collapse, so equal values give
  the single rung used before. `Controller.launch_ladder` is the same fallback code as at
  night: refused or late (`starts_soon` false) retries the slot once with `--walltime-minutes`
  before any back-off.
- Production-queue clusters (no window, no start-time restriction) use the day walltime
  directly by day, with `--walltime-minutes` as the fallback if the submission is refused.
  At night they keep `--walltime-minutes` (unchanged).
- Self-throttle: after `--day-long-max-failures` (default 3) consecutive failed long attempts on
  a cluster, that cluster uses only the short walltime for 1 hour, then long attempts resume.
  A successful long job resets the count. State lives in `ClusterMemory`
  (`longwall/<site>_<cluster>.json`), so it survives controller restarts.
- Controller log lines, one per event, for counting from the production log:
  `long_walltime fallback <site>/<cluster> 60->30` and
  `long_walltime throttle <site>/<cluster> 3 failures, short only until HH:MM`.

## Reverting

Leave `--day-walltime-minutes` unset (or equal to `--walltime-minutes`): the ladder has one rung,
no long attempt is counted, nothing is logged, and submissions and walltimes are identical to
before (tested). Setting `--day-long-max-failures 1` makes the throttle as cautious as possible.

## Consequences

A failed long day attempt costs one extra submission and a 20 s start check; at most three of
those per cluster per hour. A cluster that cannot start long jobs is therefore throttled to
the previous behaviour within three attempts.
