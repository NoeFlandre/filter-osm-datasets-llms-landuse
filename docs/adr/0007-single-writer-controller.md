# ADR-0007 — The laptop controller is the single writer of assignments

**Status:** accepted · 2026-09-28

**Context.** The frontends must run only light tasks. Distributed locking across sites is fragile.
**Decision.** Only the controller assigns chunks. The controller writes the assignments *before* `oarsub`. After a crash, it adopts them by job name. Ended jobs release the unfinished chunks. The nodes write only content-addressed parts to their site spool. The controller pulls and verifies them.
**Consequences.** If you kill the controller, a job or a site, the project loses a maximum of the requests in progress. Grid'5000 needs no credential (the models are public).
