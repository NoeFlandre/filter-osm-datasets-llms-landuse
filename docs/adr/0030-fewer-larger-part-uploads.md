# ADR-0030: Fewer, larger part uploads

Status: accepted (2026-10-05)

## Context

At ~35 GPUs the account API limit (1000 requests per 5 minutes) is saturated, mostly by result
uploads. Each `upload_part` made two `remote.put` calls (part, then manifest); each `put` is one
`batch_bucket_files`, i.e. about 2 Hub API requests (xet write-token refresh + `/batch`), so about
4 requests per part. The Runner flushed every 256 results or 120 s: ~8 parts per chunk, ~0.7
parts/s fleet-wide, ~2.8 requests/s, most of the budget (3.3/s).

## Decision

- Part and manifest go in one `put` (one commit, atomic: a manifest still implies its part).
  4 -> ~2 requests per part.
- `FLUSH_EVERY = 2048` results, `FLUSH_SECONDS = 300.0` (constants in `application/node.py`).
  At ~5 sentences/s the time bound dominates (~1500 results per part): ~6x fewer parts.
  Combined: about 12x fewer requests (~0.25 requests/s fleet-wide, ~0.007 per GPU).
- Loss on kill: a graceful stop (SIGTERM / OAR checkpoint) flushes everything; only an abrupt kill
  loses up to 300 s of work per GPU, acceptable.
- Unchanged: content addressing, dedup by hash, resume, ingest. Old small parts and new large
  ones mix freely in a chunk since progress is tracked by manifest hashes.
