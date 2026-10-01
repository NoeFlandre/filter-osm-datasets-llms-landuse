# ADR-0025: Stale besteffort jobs are reaped

Status: accepted (2026-10-02)

## Context

A besteffort job is submitted for GPUs that were free at that moment. When they are taken
meanwhile, OAR keeps the job in state Waiting ("Cannot find enough resources") indefinitely.
At Rennes 17 such jobs, 6-8 hours old, filled `max_jobs_per_site`, so nothing was submitted
there for hours, and each held its chunks, which no other GPU could take.

## Decision

- `domain.scheduling.is_stale_besteffort` is true only for a job in queue `besteffort`, state
  `Waiting`, with a known submission time, at least `STALE_BESTEFFORT_WAIT` (default 20 min)
  old. Running jobs, other queues (night, exotic: they legitimately wait) and jobs without a
  submission time are never stale.
- `Controller.reap_stale_besteffort` runs in `reconcile` after `drop_drifted`: it cancels each
  stale job of ours, marks the assignment `cancelled_stale` (terminal, so its chunks return to
  pending like any cancelled assignment) and backs the cluster off for 10 minutes.
- `--stale-besteffort-minutes` (`luf g5k run`, `run-admission`; default 20) sets the threshold.

## Consequences

Only `submitting` and `submitted` assignments are live, so every reader of live state (status,
admission, published stats, reconcile) treats `cancelled_stale` like `cancelled_late_start`.
