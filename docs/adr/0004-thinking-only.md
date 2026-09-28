# ADR-0004 — Thinking-mode generation only

**Status:** accepted · 2026-09-28

**Context.** The benchmark's logprob variant scores MCC 0.02; reasoning is what makes LFM2.5-2.6B accurate.
**Decision.** Only greedy thinking-mode generation. The chat template is applied with the benchmark's kwargs (`enable_thinking=False`, which the LFM2.5-2.6B template ignores: it always opens `<think>`), so token ids match the benchmark exactly. Parsing reads only the text after the last `</think>`.
**Consequences.** ~1,140 generated tokens per sentence: speed comes from serving and scheduling, not from shortcuts.
