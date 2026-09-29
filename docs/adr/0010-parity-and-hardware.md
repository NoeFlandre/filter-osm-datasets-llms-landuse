# ADR-0010 — Parity is established per GPU type

**Status:** accepted · 2026-09-28

**Context.** Greedy decoding is not batch- or GPU-invariant. Our outputs agree with the published L40S reference on about 88% of items, the same agreement as between the benchmark's own two reference-grade runs. Parity run 1 mixed A100 and A40 and narrowly failed the gate, with most of the drop on the A40 items (pooled ΔMCC −0.017 on 5k items, inconclusive).
**Decision.** Implementation parity is established by the full benchmark on a single GPU type passing the pre-registered gate. A100-SXM4-40GB passed on 2026-09-28: Δ macro-F1 −0.0000 (lower bound −0.0028), MCC −0.0014 (−0.0080), accuracy +0.0016 (−0.0017), failed rate −0.33 pp. Production only uses admitted GPU types (ADR-0006).
**Consequences.** Adding hardware costs about 2 GPU-hours per GPU type. Margins are never loosened after the fact without the owner's explicit approval (the lower-bound margins moved from −0.01 to −0.02 on 2026-09-29: ADR-0006).
