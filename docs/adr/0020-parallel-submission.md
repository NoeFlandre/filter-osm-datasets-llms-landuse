# ADR-0020: Submit to different sites in parallel

Status: accepted (2026-10-01)

## Context

Submissions were strictly sequential: each job costs ssh round trips, code deployment, a
20 s `starts_soon` wait and (ADR-0019) possibly policy checks of minutes. A controller launched
about 12 jobs per hour, so freed GPUs were refilled slowly.

## Decision

`--submit-workers N` (`luf g5k run` and `run-admission`; `submit_workers` in luf.toml,
`LUF_SUBMIT_WORKERS`; default 1). With N > 1, `Controller.submit`:

1. **Decides** sequentially in the ranking order (`domain/launch_plan.plan_launches`): which slot
   gets which chunks, within the total and per-site caps. It is pure and deterministic, with
   Hypothesis properties (no chunk twice, caps respected, same input same plan).
2. **Launches** on a thread pool, one task per site, at most N sites at a time. A site's jobs
   run in plan order on one worker (ssh, oarsub and the per-site policy check stay sequential);
   a slot that refuses us is skipped for the rest of the cycle, as before.
3. **Collects** results in plan order, whatever the order the threads finished in. A crashing
   worker is logged with its traceback and does not hide the other sites.

Each cycle logs `cycle submitted N jobs in S s (sites K)` (any N). With N = 1 the original
sequential loop runs unchanged. With `--policy-check per-batch`, each site worker runs its
post-batch check when its own jobs are done.

When an attempt has to fall back to a shorter walltime (ADR-0016/0017), its chunks go back to
the shared pool and are re-assigned for the new walltime under a lock; this is the one place
where the chunk assignment depends on thread timing (never producing a chunk twice).

## Thread-safety audit

- Assignment ledger: one file per assignment id, written only by the thread launching it; the
  atomic write's temporary name now includes the thread id.
- `taken` (chunks held by live jobs): mutated in the decision step, then only under a lock.
- `ClusterMemory`: back-off, besteffort-only and long-walltime files are per cluster, and a
  cluster belongs to one site, hence one worker; the long-walltime read-modify-write is also
  under a lock.
- `SiteCache.invalidate` and cache fills are under a lock.
- `Transport.stage`: assignment file per id; the bucket upload (`ls` + `put`) is serialised by a
  lock, so concurrent sites never hit the Hub together.
- `ArchiveCache`: read under a lock, so a commit is archived once, not once per site.
- Logging goes through a lock so lines do not interleave; `PolicyGate` state has its own lock.
- The ladder and fallback code only touches the above and read-only settings.

## Reverting

`--submit-workers 1` (or leave unset).

## Addendum: ingesting while sites submit

In parallel mode the sequential loop's periodic pull (`PULL_INTERVAL`) does not run, so the
progress view lagged by a whole cycle. `_submit_parallel` now waits for the site workers in the
main thread with `concurrent.futures.wait(..., timeout=PULL_INTERVAL)`; each time the timeout
elapses with workers still running it calls `pull()` and `progress.pending()` (which completes
and forgets finished chunks). This is done only from the main thread: the SQLite progress index
must not leave the thread that created it, and workers never touch it. A failed in-loop ingest is
logged ("ingest during submission failed") and the wait goes on. `cycle()` also ingests once more
after the submissions, so the status line is current when the cycle ends. `--submit-workers 1`
keeps its own sequential loop unchanged.
