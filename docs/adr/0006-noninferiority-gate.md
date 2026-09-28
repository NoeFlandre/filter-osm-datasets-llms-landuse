# ADR-0006 — Serving changes pass a non-inferiority gate

**Status:** accepted · 2026-09-28 (margins approved by the owner)

**Decision.** Paired, language-stratified bootstrap (10,000 resamples, seed 0) of candidate − reference on the 25,500-item benchmark. Pass iff the one-sided 95% lower bounds of Δ macro-F1, Δ macro-MCC and Δ macro-accuracy all exceed −0.01, the failed-rate increase upper bound is below +0.5 pp, and no language loses more than 0.05 F1.
**Why accuracy too.** The benchmark's F1/MCC exclude failed items; failing on hard items *raises* them (seen in the budget simulation). Accuracy and the failed-rate guard close that loophole.
**Per-GPU admission.** A one-time 1,700-item smoke run per GPU model (issue #18), not per job.
