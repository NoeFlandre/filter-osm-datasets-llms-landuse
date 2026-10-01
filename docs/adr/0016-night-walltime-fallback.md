# ADR-0016: Long night jobs with a shorter fallback in the same slot

Status: accepted (2026-10-01)

## Context

About 6 of the 30 minutes of every job go to environment build and SGLang start. At night
(`-t night`, queued from 19:00, deadline 09:00) longer jobs would raise throughput by 15-20 %.
But a long job is harder to place: it may be refused, or OAR may predict a late start, in which
case the controller cancels it and backs the cluster off for 30 minutes.

## Decision

`domain/scheduling.walltime_ladder` returns the walltimes to try for a slot: by day the single
day walltime; at night the preferred (long) walltime then the fallback, each capped by the
window maximum, the fallback never longer than the preferred one. `Slot.fallbacks` carries the
remaining rungs. `Controller.launch_ladder` tries them in order, recomputing the chunk
assignment (capacity = sentences/s x (walltime - setup)) for each; a failed attempt is no longer
live, so its chunks return to the pool. The cluster is backed off only after the last rung fails.

`--night-max-queued-per-site` gives night and weekend windows their own queue depth (default:
the day value).

## Consequences

A late long job costs one extra submission and a 20 s start check before the short job, instead
of a 30 minute back-off. Production clusters (abaca queue) keep their fixed walltime.
