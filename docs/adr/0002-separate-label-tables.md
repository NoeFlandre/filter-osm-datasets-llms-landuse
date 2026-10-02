# ADR-0002 — Mirror inputs; labels live in their own tables

**Status:** accepted · 2026-09-28

**Context.** Downstream analysis needs both yes and no sentences. The inputs must stay unchanged.
**Decision.** Each `-landuse` repo copies its input byte for byte. It adds two tables:
`labels/<input path>` (one row for each sentence position, natural join keys + `label_id`) and
`generations/<fp>/` (one row for each unique text, with the key `generation_id`).
**Consequences.** The project drops no polygon and no sentence. The project stores the raw outputs one time for each unique text, not for each row.
