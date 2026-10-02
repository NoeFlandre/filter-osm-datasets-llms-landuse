# ADR-0026: Publish jobs extend a resolution snapshot instead of re-reading every part

Status: accepted (2026-10-02)

## Context

A wiki publish job downloaded each result part of the bucket (44,336 parts, 11.7 GB, 5,564
chunks). Then it parsed approximately 11 M rows serially to build the resolution index. This occurred before the job looked at the first input
file. It used the whole 1 hour day walltime (OAR 2072170, 3128739: no commit) and
approximately 50 minutes of the 8 hour night job. A second pass over each local part rebuilt the
generations that ship with complete files.

## Decision

- The project saves a checkpoint of the resolution index to `index/resolve-<fp>.sqlite` in the
  bucket. The index has the text hash -> verdict, the smallest part key that produced it, the set of
  parts already read, and the parser version. `application/resolution_sync.py` downloads the index and lists `parts/<fp>/` one time.
  It reads only the unseen parts. It downloads batches of 1,000 in 4 streams while a process pool parses the previous batch.
  It deletes each part after indexing. It uploads the snapshot every 5 minutes
  and at the end. Thus a stopped job still advances the next one (the first job after this change
  catches up the backlog the same way, and you can resume it).
- The smallest part key still wins for a hash (`add_part` upserts only a smaller key). Thus the
  result does not depend on the arrival order. The canonical rows do not change. The project discards a snapshot that a
  different `PARSER_VERSION` made and rebuilds it.
- `publish(resolved=, fetch_parts=)`: the project builds the generations that ship with complete files from
  the parts that the index points to. It fetches them on demand. It does not pass over each part.
- The job commits ready partial files at least every 10 minutes (`FLUSH_SECONDS`), not
  only for each 100. Thus a stop or a slow input scan never ends a job with an empty commit list.
- A stop request during the catch-up ends it early (the snapshot is saved). The publication continues
  with the known verdicts. This can only delay files. It never publishes a wrong one.

## Consequences

The steady-state restore is the snapshot download (hundreds of MB), one bucket listing and the
parts that are new since then. Before this change, it was 44 k downloads and approximately 11 M parses. The bucket gets one
sqlite file for each fingerprint. Chunk ids sort bytewise like the part ordering that the canonical choice uses.
