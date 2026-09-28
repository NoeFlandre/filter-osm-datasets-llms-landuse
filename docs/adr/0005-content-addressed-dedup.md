# ADR-0005 — Content-addressed dedup per generation fingerprint

**Status:** accepted · 2026-09-28

**Decision.** A unique sentence text (sha256 of its exact UTF-8 bytes) is generated once per `config_fingerprint` (model, draft, revisions, prompt, template kwargs, sampling, output-affecting engine args). Speed-only engine args are excluded so per-GPU profiles share results; the gate decides whether they are truly neutral (ADR-0006).
**Consequences.** One canonical generation per text (smallest part id); a row's decision can come from a different GPU than a duplicate's would have (documented in debt).
