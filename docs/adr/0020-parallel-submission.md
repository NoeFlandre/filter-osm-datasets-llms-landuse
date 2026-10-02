# ADR-0020: Submit to different sites in parallel

Status: accepted (2026-10-01)

## Context

The submissions were strictly sequential. Each job costs ssh round trips, code deployment, a
20 s `starts_soon` wait and (ADR-0019) possibly policy checks of several minutes. A controller launched
approximately 12 jobs each hour. Thus the controller refilled free GPUs slowly.

## Decision

`--submit-workers N` (`luf g5k run` and `run-admission`; `submit_workers` in luf.toml,
`LUF_SUBMIT_WORKERS`; default 1). With N > 1, `Controller.submit` does these steps:

1. **Decides** sequentially in the ranking order (`domain/launch_plan.plan_launches`) which slot
   gets which chunks, within the total caps and the per-site caps. The function is pure and deterministic, with
   Hypothesis properties (no chunk twice, caps respected, same input gives the same plan).
2. **Launches** on a thread pool, one task for each site, a maximum of N sites at the same time. The jobs of a site
   run in plan order on one worker (ssh, oarsub and the policy check for each site stay sequential).
   The controller skips a slot that refuses us for the rest of the cycle, as before.
3. **Collects** the results in plan order, in any order of thread completion. The controller logs a crashing
   worker with its traceback. The crash does not hide the other sites.

Each cycle logs `cycle submitted N jobs in S s (sites K)` (any N). With N = 1, the original
sequential loop runs without change. With `--policy-check per-batch`, each site worker runs its
post-batch check when its own jobs are done.

When an attempt must fall back to a shorter walltime (ADR-0016/0017), its chunks go back to
the shared pool. The controller assigns them again for the new walltime under a lock. This is the only place
where the chunk assignment depends on thread timing (it never produces a chunk two times).

## Thread-safety audit

- Assignment ledger: one file for each assignment id. Only the thread that launches it writes the file. The
  temporary name of the atomic write now includes the thread id.
- `taken` (chunks held by live jobs): the decision step changes it. After that, it changes only under a lock.
- `ClusterMemory`: the back-off, besteffort-only and long-walltime files are for each cluster. A
  cluster belongs to one site and thus to one worker. The read-modify-write of the long-walltime file is also
  under a lock.
- `SiteCache.invalidate` and the cache fills are under a lock.
- `Transport.stage`: one assignment file for each id. A lock serializes the bucket upload (`ls` + `put`).
  Thus concurrent sites never hit the Hub at the same time.
- `ArchiveCache`: the read is under a lock. Thus the project archives a commit one time, not one time for each site.
- Logging goes through a lock. Thus the lines do not interleave. The `PolicyGate` state has its own lock.
- The ladder and fallback code touches only the items above and read-only settings.

## Reverting

Use `--submit-workers 1` (or do not set it).

## Addendum: ingesting while sites submit

In parallel mode, the periodic pull (`PULL_INTERVAL`) of the sequential loop does not run. Thus the
progress view was late by a whole cycle. `_submit_parallel` now waits for the site workers in the
main thread with `concurrent.futures.wait(..., timeout=PULL_INTERVAL)`. Each time the timeout
ends and workers still run, it calls `pull()` and `progress.pending()` (which completes
and forgets finished chunks). This runs only from the main thread. The SQLite progress index
must not leave the thread that created it, and the workers never touch it. The controller logs a failed in-loop ingest
("ingest during submission failed") and the wait continues. `cycle()` also ingests one more time
after the submissions. Thus the status line is current when the cycle ends. `--submit-workers 1`
keeps its own sequential loop without change.
