# ADR-0018: Over-assign chunks so jobs run until the checkpoint signal

Status: accepted (2026-10-01)

## Context

A job receives chunks with a size of profile speed x (walltime - setup) x 1.2. The measured median
useful fraction (generation time / walltime) is only approximately 0.5. Jobs finish early and exit.
Then the project must get the GPU again. The manifests save the work for each text. OAR sends SIGUSR2
(`--checkpoint 300`) 5 minutes before the walltime ends.

## Verified behavior (tests)

- Node: `node_main.Stop` handles SIGTERM, SIGUSR2 and SIGINT. Then `Runner` stops the admission of new work,
  cancels the requests in progress and flushes the completed texts as a part. `on_part` uploads the part and
  its manifest. The node writes the job summary and the process exits with code 0. A chunk is in
  `chunks_done` only if each text has a part (`test_node.py`, with real signals). The node needed no change.
- Controller: a chunk is complete only when `done_count >= size`. A chunk that stopped part way stays
  in `pending()` and the controller assigns it again (`test_controller_parts.py`).

## Decision

`--chunk-overflow` (luf.toml `chunk_overflow`, `LUF_CHUNK_OVERFLOW`, validated >= 1.0, default
1.2) is the `overflow` argument of `assign_chunks`. Production value: 1.6.

## Reverting

Do not set the option. The value is then 1.2, the previous behavior.

## Consequences

Jobs hold more chunks than they finish. Thus a job can reserve a chunk for the whole job although it processes the chunk only in part.
When the job stops, the chunk returns to the pool. The progress of each job depends on the walltime,
not on the estimate.
