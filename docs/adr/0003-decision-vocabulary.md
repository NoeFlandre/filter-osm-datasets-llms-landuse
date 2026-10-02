# ADR-0003 — Decisions: yes, no, failed, skipped_unsplit

**Status:** accepted · 2026-09-28

**Decision.** The project keeps `failed` (truncated, unclosed think, empty, ambiguous, non-English token, no label) with its reason. It never forces a value. `skipped_unsplit` marks the texts that the upstream segmenter did not split (unsupported or undetected language). These texts get no LLM call.
**Consequences.** Each in-scope sentence has exactly one decision. You can audit the failures and parse them again offline.
