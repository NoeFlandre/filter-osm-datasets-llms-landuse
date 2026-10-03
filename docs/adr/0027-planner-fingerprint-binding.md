# ADR-0027: Bind each planner index to one generation fingerprint

Status: accepted (2026-10-03, owner decision)

## Context

Planner SQLite state is stored at `index/<dataset>.sqlite`, while plan lines are stored under
`plans/<dataset>/<config fingerprint>/`. The index remembers which texts were assigned to chunks.
Recovering a plan file from that index is safe only when those chunk ids were made for the current
fingerprint. Otherwise recovery can make a new plan point to an old pretokenized chunk.

## Decision

Persist the generation fingerprint in a `planner_meta` table in the SQLite index. A bound index
can only be opened with its recorded fingerprint. For a legacy index without metadata, validate
that each stored chunk id recomputes from the current fingerprint and its indexed text hashes
before binding it. Reject an unbound index if that validation fails or if it contains completed
texts with no chunk id, since their original fingerprint cannot be established.

## Consequences

* Same-fingerprint restarts keep their existing chunk assignments and can recover lost plan lines.
* A different fingerprint must use a separate work store; existing chunk files and index records
  remain intact, and old token ids are never silently assigned to a new plan.
* A legacy index with unverifiable state fails closed. The owner can resume it with its original
  fingerprint or choose a fresh work store for a new generation configuration.
