# ADR-0002 — Mirror inputs; labels live in their own tables

**Status:** accepted · 2026-09-28

**Context.** Downstream analysis needs both yes and no sentences, and the inputs must stay untouched.
**Decision.** Each `-landuse` repo mirrors its input byte-for-byte and adds `labels/<input path>` (one row per sentence position, natural join keys + `label_id`) and `generations/<fp>/` (one row per unique text, keyed by `generation_id`).
**Consequences.** No polygon or sentence is dropped; raw outputs are stored once per unique text, not per row.
