# ADR-0011 — Several controllers, one namespace each

**Status:** accepted · 2026-09-28

**Context.** GPU admission runs one benchmark per GPU type; running them one after another wastes free capacity.
**Decision.** A controller only reconciles the assignments of its own namespace on its own sites (#33), so several controllers can share the work tree safely, and each stays the single writer for its namespace (ADR-0007). Per-site caps count every `luf-` job on a site. Consolidating them into one multi-namespace controller is tracked in #44.
