# ADR-0025: Stale besteffort jobs are reaped

Status: accepted (2026-10-02)

## Context

The project submits a besteffort job for GPUs that were free at that time. If other users take the GPUs
meanwhile, OAR keeps the job in state Waiting ("Cannot find enough resources") with no end.
At Rennes, 17 such jobs, 6-8 hours old, filled `max_jobs_per_site`. Thus the controller submitted nothing
there for hours. Each job held its chunks, and no other GPU could take them.

## Decision

- `domain.scheduling.is_stale_besteffort` is true only for a job in queue `besteffort`, state
  `Waiting`, with a known submission time, and at least `STALE_BESTEFFORT_WAIT` (default 20 min)
  old. A job is never stale in these cases: it is running, it is in another queue (night, exotic: they
  wait for a valid reason), or it has no submission time.
- `Controller.reap_stale_besteffort` runs in `reconcile` after `drop_drifted`. It cancels each
  stale job of ours and marks the assignment `cancelled_stale` (terminal, thus its chunks return to
  pending like any cancelled assignment). It backs off the cluster for 10 minutes.
- `--stale-besteffort-minutes` (`luf g5k run`, `run-admission`; default 20) sets the threshold.

## Consequences

Only `submitting` and `submitted` assignments are live. Thus each reader of the live state (status,
admission, published stats, reconcile) treats `cancelled_stale` like `cancelled_late_start`.
