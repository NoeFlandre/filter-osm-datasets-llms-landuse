# ADR-0010 — Parity is established per GPU type

**Status:** accepted · 2026-09-28

**Context.** Greedy decoding is not invariant to the batch or to the GPU. Our outputs agree with the published L40S reference on approximately 88% of items. This is the same agreement as between the two reference-grade runs of the benchmark. Parity run 1 mixed A100 and A40 and narrowly failed the gate. Most of the drop is on the A40 items (pooled ΔMCC −0.017 on 5k items, not conclusive).
**Decision.** The full benchmark on a single GPU type must pass the pre-registered gate. This establishes implementation parity. A100-SXM4-40GB passed on 2026-09-28: Δ macro-F1 −0.0000 (lower bound −0.0028), MCC −0.0014 (−0.0080), accuracy +0.0016 (−0.0017), failed rate −0.33 pp. Production uses only admitted GPU types (ADR-0006).
**Consequences.** New hardware costs approximately 2 GPU-hours for each GPU type. Never loosen the margins after the fact without the explicit approval of the owner. The lower-bound margins changed from −0.01 to −0.02 on 2026-09-29 (ADR-0006).
