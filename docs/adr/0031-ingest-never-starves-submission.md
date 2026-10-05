# ADR-0031: Ingest never starves submission

Status: accepted (2026-10-05)

## Context

With the Hub API near its rate limit (1000 requests per 5 minutes per account, shared by about 60
GPU jobs, three controllers and publish jobs), controller ingest calls got 429. `with_retry`
(ADR-0028) waits up to 300 s per wait and 1200 s per call, and the controller made those calls in
its own cycle: a pull at the start of each cycle, and a pull from the main thread every 180 s
while the site workers submitted (`_wait_ingesting`, which only noticed finished workers between
two pulls). A cycle therefore lasted as long as its slowest ingest: on 2026-10-05 the process was
silent for 40 minutes while the GPUs drained from 70 to 10, and `cycle submitted 71 jobs in 1844 s`
(jobs run 30 minutes, so the GPUs idled half the time). The wiki chunk counter froze as well,
because chunks are registered complete only after a pull.

The submission itself is ~14 jobs per site, each with `starts_soon` (a fixed 20 s wait for OAR to
schedule the job, sites in parallel), the code deployment, the staging rsync and the per-site
policy check; that is about 10 minutes, not 30.

## Decision

- Ingest runs in a dedicated daemon thread (`BackgroundIngest`, `Settings.background_ingest`, on
  for `luf g5k run` and `run-many`). Every 180 s (and when the cycle kicks it) it runs one pull and
  logs one line: `ingest: full|incremental pull, N listings, M new manifests, F rate-limited,
  D deferred, S s`. The cycle only hands over the live chunks (reconcile, and every launch) and
  never waits for it; `pending()` reads the index as the latest completed ingest left it.
- Threads: the ingest thread owns its own `WorkProgress`/`ProgressIndex` (a SQLite connection per
  thread, 60 s busy timeout, rollback journal as before); it is the only one that ingests
  manifests. The cycle thread is the only writer of `complete.jsonl` (`pending()`), and keeps
  registering completions every 180 s while it submits.
- Bounded listings: ingest uses its own `BucketRemote` with the short retry budget `SHORT`
  (3 attempts, at most 60 s per wait, 90 s in all) while uploads, nodes and publish jobs keep the
  long one (`LONG`, unchanged). The budget is a `RetryBudget` parameter of `with_retry` and
  `BucketRemote`. The first listing that still fails with a 429 ends the pull (no hammering; the
  chunk stays owed and the full listing stays due), and a pull starts no new listing after 240 s
  (`PULL_BUDGET`, the rest stays owed). A non-429 error propagates and is logged by the worker.
- ADR-0029 guarantees stay: the first pull after a start is a full listing (now in the
  background, so submission starts at once), then ended chunks, live chunks every 900 s and a full
  listing every 3600 s; ingestion is idempotent through the progress index, so a retried or
  overlapping listing duplicates and misses nothing.
- Not changed: `usagepolicycheck -t` per batch, the 20 s `starts_soon` wait, the serialised
  bucket upload (these are work, not ingest).

## Consequences

A cycle's duration no longer depends on the number of ended chunks nor on the Hub rate limit,
except through the uploads of chunk inputs. Under saturation ingest lags (retried every 180 s)
instead of freezing submission. Controllers must be restarted to pick this up.
