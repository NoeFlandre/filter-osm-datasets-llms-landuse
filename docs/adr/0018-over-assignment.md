# ADR-0018: Over-assign chunks so jobs run until the checkpoint signal

Status: accepted (2026-10-01)

## Context

A job receives chunks sized to profile speed x (walltime - setup) x 1.2. The measured median
useful fraction (generation time / walltime) is only about 0.5: jobs finish early, exit, and the
GPU has to be re-acquired. Work is saved per text in manifests, and OAR sends SIGUSR2
(`--checkpoint 300`) 5 minutes before the walltime ends.

## Verified behaviour (tests)

- Node: `node_main.Stop` handles SIGTERM, SIGUSR2 and SIGINT; `Runner` then stops admitting,
  cancels in-flight requests, flushes completed texts as a part, `on_part` uploads the part and
  its manifest, the job summary is written and the process exits 0. A chunk is listed in
  `chunks_done` only if every text has a part (`test_node.py`, with real signals). No node
  change was needed.
- Controller: a chunk is complete only when `done_count >= size`; a chunk stopped part way stays
  in `pending()` and is assigned again (`test_controller_parts.py`).

## Decision

`--chunk-overflow` (luf.toml `chunk_overflow`, `LUF_CHUNK_OVERFLOW`, validated >= 1.0, default
1.2) is the `overflow` argument of `assign_chunks`. Production value: 1.6.

## Reverting

Leave the option unset: 1.2, the previous behaviour.

## Consequences

Jobs hold more chunks than they finish, so a chunk can be reserved for a whole job although only
partly processed; the job's stop returns it to the pool. Progress per job becomes walltime-bound
instead of estimate-bound.
