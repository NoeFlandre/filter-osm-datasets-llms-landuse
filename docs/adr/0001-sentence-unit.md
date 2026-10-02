# ADR-0001 — The unit is one pre-segmented sentence

**Status:** accepted · 2026-09-28

**Context.** The benchmark classifies single sentences. The inputs contain sentences that an upstream tool (SaT) segmented.
**Decision.** Classify each upstream sentence as it is. Never segment again.
**Consequences.** The results match the task of the benchmark. ADR-0003 explains how the project handles texts that the upstream tool did not split.
