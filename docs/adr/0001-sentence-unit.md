# ADR-0001 — The unit is one pre-segmented sentence

**Status:** accepted · 2026-09-28

**Context.** The benchmark classifies single sentences; the inputs ship sentences segmented upstream (SaT).
**Decision.** Classify every upstream sentence as-is; never re-segment.
**Consequences.** Results match the benchmark's task. Texts left unsplit upstream are handled by ADR-0003.
