# ADR-0026: Publish jobs extend a resolution snapshot instead of re-reading every part

Status: accepted (2026-10-02)

## Context

A wiki publish job downloaded every result part of the bucket (44,336 parts, 11.7 GB, 5,564
chunks), then parsed ~11 M rows serially to build the resolution index, before the first input
file was looked at. That cost the whole 1 hour day walltime (OAR 2072170, 3128739: no commit) and
about 50 minutes of the 8 hour night job. A second pass over every local part rebuilt the
generations that ship with complete files.

## Decision

- The resolution index (text hash -> verdict, the smallest part key that produced it, the set of
  parts already read, the parser version) is checkpointed to `index/resolve-<fp>.sqlite` in the
  bucket. `application/resolution_sync.py` downloads it, lists `parts/<fp>/` once and reads only
  the unseen parts: batches of 1,000 downloaded in 4 streams while the previous batch is parsed
  by a process pool; each part is deleted once indexed. The snapshot is uploaded every 5 minutes
  and at the end, so a stopped job still advances the next one (the first job after this change
  catches up the backlog the same way, resumably).
- The smallest part key still wins for a hash (`add_part` upserts only a smaller key), so the
  result is independent of arrival order; canonical rows are unchanged. A snapshot made by a
  different `PARSER_VERSION` is discarded and rebuilt.
- `publish(resolved=, fetch_parts=)`: the generations shipping with complete files are built from
  the parts the index points to, fetched on demand, instead of a pass over every part.
- Partial files that are ready are committed at least every 10 minutes (`FLUSH_SECONDS`), not
  only per 100, so a stop or a slow input scan never ends a job with an empty commit list.
- A stop request during the catch-up ends it early (snapshot saved); publication continues
  with the verdicts known, which can only delay files, never publish a wrong one.

## Consequences

Steady-state restore is the snapshot download (hundreds of MB) plus one bucket listing and the
parts that appeared since, instead of 44 k downloads and ~11 M parses. The bucket gains one
sqlite file per fingerprint. Chunk ids sort bytewise like the part ordering used for the
canonical choice.
