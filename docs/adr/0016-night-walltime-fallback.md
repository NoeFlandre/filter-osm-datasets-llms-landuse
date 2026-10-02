# ADR-0016: Long night jobs with a shorter fallback in the same slot

Status: accepted (2026-10-01)

## Context

Approximately 6 of the 30 minutes of each job are for the environment build and the SGLang start. At night
(`-t night`, queued from 19:00, deadline 09:00), longer jobs can increase the throughput by 15-20 %.
But it is harder to place a long job. OAR can refuse it, or OAR can predict a late start. Then the controller
cancels the job and backs off the cluster for 30 minutes.

## Decision

`domain/scheduling.walltime_ladder` returns the walltimes to try for a slot. By day, it returns the single
day walltime. At night, it returns the preferred (long) walltime and then the fallback. The window maximum caps each of them.
The fallback is never longer than the preferred walltime. `Slot.fallbacks` has the remaining rungs.
`Controller.launch_ladder` tries them in order. For each rung, it calculates the chunk
assignment again (capacity = sentences/s x (walltime - setup)). A failed attempt is no longer
live. Thus its chunks return to the pool. The controller backs off the cluster only after the last rung fails.

`--night-max-queued-per-site` gives the night and weekend windows their own queue depth (default:
the day value).

## Consequences

A late long job costs one extra submission and a 20 s start check before the short job. Before this change,
it cost a 30 minute back-off. Production clusters (abaca queue) keep their fixed walltime.
