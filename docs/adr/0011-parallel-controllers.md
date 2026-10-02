# ADR-0011 — Several controllers, one namespace each

**Status:** accepted · 2026-09-28

**Context.** GPU admission runs one benchmark for each GPU type. If the benchmarks run one after the other, free capacity is wasted.
**Decision.** A controller reconciles only the assignments of its own namespace on its own sites (#33). Thus several controllers can safely share the work tree. Each controller stays the single writer for its namespace (ADR-0007). The per-site caps count each `luf-` job on a site. Issue #44 tracks the merge of the controllers into one multi-namespace controller.
