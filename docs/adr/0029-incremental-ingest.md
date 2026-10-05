# ADR-0029: Ingest lists only the chunks that can have new parts

Status: accepted (2026-10-05)

## Context

Every controller pull (180 s) listed the whole `parts/<fp>/` prefix, then `jobs/`. At 44k parts
(88k entries with manifests) that is about 90 paged Hub API requests per listing, growing every
hour, for each controller process (the run and the admission controllers) and the publish jobs.
With ~35 GPUs the account limit (1000 requests per 5 minutes) became the production bottleneck:
ingest lagged 5-8 minutes behind 429 waits (ADR-0028).

## Decision

- A part lives at `parts/<fp>/<chunk>/<part>.{parquet,json}`, so a chunk prefix holds only that
  chunk's parts. `Transport.pull(progress, live_chunks)` lists:
  - the prefix of each chunk whose assignment ended since the previous pull, once (kept owed until
    the listing succeeds); `Transport.stage` also tracks the chunks, so a job that starts and ends
    between two pulls is still listed;
  - the prefixes of the live chunks every `LIVE_INTERVAL` (900 s), so progress stays current;
  - the whole `parts/<fp>/` on the first pull of a process (a restart catches up completely),
    every `RECONCILE_INTERVAL` (3600 s, the safety net for parts of a job killed in unusual ways),
    and whenever the live chunks are unknown (`live_chunks=None`);
  - `jobs/` only when an assignment ended or on a full listing.
- Ingestion stays idempotent: the progress index dedups by manifest path and hash. Per-chunk
  listings and the full listing may overlap freely.
- `fetch_manifests` with `seen` now returns a manifest that is local but unseen (interrupted
  ingest) so it is ingested, instead of skipping it forever.
- Nodes, the bucket layout and `luf status` do not change. Publish jobs and other one-shot
  listings (`remote_plan`, `results`, `resolution_sync`) are unchanged: they run rarely.

## Consequences

Requests per controller drop from ~90+ per 3 minutes to ~(live chunks / 15 min) + ended chunks +
~100 per hour. Progress of a running job can lag up to 15 minutes (nothing is decided on it:
live chunks are already excluded from assignment); ended jobs are ingested on the next pull.
Controllers must be restarted to pick up the change.
