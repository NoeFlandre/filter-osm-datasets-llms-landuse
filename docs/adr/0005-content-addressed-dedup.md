# ADR-0005 — Content-addressed dedup per generation fingerprint

**Status:** accepted · 2026-09-28

**Decision.** The project generates a unique sentence text (sha256 of its exact UTF-8 bytes) one time for each `config_fingerprint` (model, draft, revisions, prompt, template kwargs, sampling, engine arguments that change the output). The fingerprint excludes the engine arguments that change only the speed. Thus the GPU profiles share results. The gate decides if these arguments are really neutral (ADR-0006).
**Consequences.** Each text has one canonical generation (the smallest part id). The decision of a row can come from a different GPU than the GPU that a duplicate would use. The [known weaknesses](../debt.md) page documents this.
