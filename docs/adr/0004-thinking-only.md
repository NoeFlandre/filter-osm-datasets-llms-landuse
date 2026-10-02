# ADR-0004 — Thinking-mode generation only

**Status:** accepted · 2026-09-28

**Context.** The logprob variant of the benchmark has an MCC of 0.02. Reasoning makes LFM2.5-2.6B accurate.
**Decision.** Use only greedy thinking-mode generation. Apply the chat template with the kwargs of the benchmark (`enable_thinking=False`). The LFM2.5-2.6B template ignores this kwarg and always opens `<think>`. Thus the token ids match the benchmark exactly. The parser reads only the text after the last `</think>`.
**Consequences.** Each sentence needs approximately 1,140 generated tokens. The speed comes from serving and scheduling, not from shortcuts.
