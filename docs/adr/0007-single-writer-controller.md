# ADR-0007 — The laptop controller is the single writer of assignments

**Status:** accepted · 2026-09-28

**Context.** Frontends must only run light tasks; distributed locking across sites is fragile.
**Decision.** Only the controller assigns chunks. Assignments are written *before* `oarsub` and adopted by job name after a crash; ended jobs release unfinished chunks. Nodes only write content-addressed parts to their site spool; the controller pulls and verifies them.
**Consequences.** Killing the controller, a job or a site loses at most in-flight requests; no credential is needed on Grid'5000 (the models are public).
