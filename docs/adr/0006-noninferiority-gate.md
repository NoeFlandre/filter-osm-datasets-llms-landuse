# ADR-0006 — Serving changes pass a non-inferiority gate

**Status:** accepted · 2026-09-28 (margins approved by the owner)

**Decision.** Paired, language-stratified bootstrap (10,000 resamples, seed 0) of candidate − reference on the 25,500-item benchmark. Pass iff the one-sided 95% lower bounds of Δ macro-F1, Δ macro-MCC and Δ macro-accuracy all exceed −0.01, the failed-rate increase upper bound is below +0.5 pp, and no language loses more than 0.05 F1.
**Why accuracy too.** The benchmark's F1/MCC exclude failed items; failing on hard items *raises* them (seen in the budget simulation). Accuracy and the failed-rate guard close that loophole.
**Per-GPU-type admission (revised 2026-09-28, owner decision).** Each GPU type is admitted once by running the **full** 25,500-item benchmark in its own namespace (`<fp>-gpu-<key>`) through this same gate (`luf bench admit`). A 1,700-item subset was rejected: its confidence intervals are about 5× wider, so a good GPU would fail the −0.01 margins on sampling noise alone.
**Outcome so far.** Parity run 1 (mixed A100/A40) narrowly failed (MCC lower bound −0.0134). The benchmark's two reference-grade runs pass the gate against each other (lower bound −0.0073, the noise floor). A100-SXM4-40GB alone passes (MCC lower bound −0.0080): see ADR-0010.

**Margin revised 2026-09-29 (owner decision).** The lower-bound margins on Δ macro-F1, Δ macro-MCC and Δ macro-accuracy move from −0.01 to **−0.02**. Rationale: the benchmark's own two reference-grade runs sit at −0.0073 against each other, so −0.01 left almost no room for GPU-to-GPU numeric noise (L40S: point ΔMCC −0.006 but lower bound −0.0123; A40 −0.016) while quality is unchanged. The failed-rate guard (+0.5 pp) and per-language guard (0.05 F1) are unchanged. Already-admitted types are unaffected; rejected types are re-scored with `luf bench admit`.
