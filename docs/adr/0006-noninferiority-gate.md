# ADR-0006 — Serving changes pass a non-inferiority gate

**Status:** accepted · 2026-09-28 (the owner approved the margins)

**Decision.** Use a paired, language-stratified bootstrap (10,000 resamples, seed 0) of candidate − reference on the 25,500-item benchmark. A candidate passes if all these conditions are true:

- The one-sided 95% lower bounds of Δ macro-F1, Δ macro-MCC and Δ macro-accuracy are all above −0.01.
- The upper bound of the failed-rate increase is below +0.5 pp.
- No language loses more than 0.05 F1.

**Why accuracy too.** The F1 and MCC of the benchmark exclude failed items. If the model fails on hard items, F1 and MCC *increase* (the budget simulation shows this). The accuracy and the failed-rate guard close this loophole.
**Per-GPU-type admission (revised 2026-09-28, owner decision).** The project admits each GPU type one time. It runs the **full** 25,500-item benchmark in its own namespace (`<fp>-gpu-<key>`) through this same gate (`luf bench admit`). The project rejected a subset of 1,700 items. Its confidence intervals are approximately 5× wider. Thus a good GPU can fail the −0.01 margins only because of sampling noise.
**Outcome so far.** Parity run 1 (mixed A100/A40) narrowly failed (MCC lower bound −0.0134). The two reference-grade runs of the benchmark pass the gate against each other (lower bound −0.0073, the noise floor). A100-SXM4-40GB alone passes (MCC lower bound −0.0080). See ADR-0010.

**Margin revised 2026-09-29 (owner decision).** The lower-bound margins on Δ macro-F1, Δ macro-MCC and Δ macro-accuracy change from −0.01 to **−0.02**. Reason: the two reference-grade runs of the benchmark are at −0.0073 against each other. Thus −0.01 left almost no room for GPU-to-GPU numeric noise (L40S: point ΔMCC −0.006 but lower bound −0.0123; A40 −0.016). The quality does not change. The failed-rate guard (+0.5 pp) and the per-language guard (0.05 F1) do not change. The change does not affect the types that the project already admitted. Score the rejected types again with `luf bench admit`.
