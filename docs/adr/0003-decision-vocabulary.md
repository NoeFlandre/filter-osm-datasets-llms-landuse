# ADR-0003 — Decisions: yes, no, failed, skipped_unsplit

**Status:** accepted · 2026-09-28

**Decision.** `failed` (truncated, unclosed think, empty, ambiguous, non-English token, no label) is kept with its reason and never coerced. `skipped_unsplit` marks texts the upstream segmenter did not split (unsupported/undetected language); they get no LLM call.
**Consequences.** Every in-scope sentence has exactly one decision; failures are auditable and re-parsable offline.
