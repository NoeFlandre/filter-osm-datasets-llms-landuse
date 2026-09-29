# ADR-0012 — Production concurrency window is 96

**Status:** accepted · 2026-09-29

**Context.** The reference configuration serves 16 concurrent requests, which leaves the GPU
mostly idle: the model generates about 1,000 thinking tokens per sentence, and each decoding
step is almost free to widen. Greedy decoding is not batch-invariant, so a different window
changes a few outputs and must pass the non-inferiority gate (ADR-0006).

**Decision.** Run production at `max_running_requests = 96` on the A100-SXM4-40GB, A40 and L40S
profiles. Windows 64 and 96 passed the full-benchmark gate; 128 failed the per-language guard
by 0.00002 and is not used. The gate is not loosened for it. The four other admitted types (H100 NVL, A100-PCIe-40GB, RTX A5000, L4) passed the same
gate in a second full-benchmark run at 96. A GPU type admitted later stays at 16 until a
gated run covers it.

**Consequences.** About 50 % more sentences/s per GPU (L40S: 3.7 → 5.5). Profiles are data
(`profiles/<gpu>.json`), so the window can be raised per type after its own gated run.
Details and the sweep are in [tuning](../tuning.md).
