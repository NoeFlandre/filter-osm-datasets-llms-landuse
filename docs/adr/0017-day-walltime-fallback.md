# ADR-0017: Long day jobs with a short fallback and a self-throttle

Status: accepted (2026-10-01)

## Context

By day (09:00-19:00), the usage policy allows only jobs that start immediately and last a maximum of
1 hour. Approximately 6 of each 30 minutes of a job are startup. Thus a 60-minute day job increases the useful
generation time from approximately 80 % to approximately 90 %. But it is harder to place a long day job.
The owner requires that an attempt with a long job must never slow us down.

## Decision

- `--day-walltime-minutes` (luf.toml `day_walltime_minutes`, `LUF_DAY_WALLTIME_MINUTES`) is the
  preferred day walltime. If you do not set it, it equals `--walltime-minutes`. Then nothing changes.
- `domain/scheduling.walltime_ladder` takes `day_short`. By day, the rungs are
  `min(day, window max)` and then `min(day_short, day)`. Duplicates merge. Thus equal values give
  the single rung that the project used before. `Controller.launch_ladder` is the same fallback code as at
  night. If OAR refuses the job or the job is late (`starts_soon` false), the controller retries the slot one time with
  `--walltime-minutes` before any back-off.
- Production-queue clusters (no window, no start-time restriction) use the day walltime
  directly by day. If OAR refuses the submission, they use `--walltime-minutes` as the fallback.
  At night they keep `--walltime-minutes` (no change).
- Self-throttle: after `--day-long-max-failures` (default 3) consecutive failed long attempts on
  a cluster, that cluster uses only the short walltime for 1 hour. Then the long attempts resume.
  A successful long job resets the count. The state is in `ClusterMemory`
  (`longwall/<site>_<cluster>.json`). Thus it survives controller restarts.
- The controller log has one line for each event. You can count them in the production log:
  `long_walltime fallback <site>/<cluster> 60->30` and
  `long_walltime throttle <site>/<cluster> 3 failures, short only until HH:MM`.

## Reverting

Do not set `--day-walltime-minutes` (or set it equal to `--walltime-minutes`). Then the ladder has one rung.
The controller counts no long attempt and logs nothing. The submissions and walltimes are the same as
before (tested). If you set `--day-long-max-failures 1`, the throttle is as cautious as possible.

## Consequences

A failed long day attempt costs one extra submission and a 20 s start check. The maximum is three of
these for each cluster each hour. Thus the throttle returns a cluster that cannot start long jobs to
the previous behavior within three attempts.
