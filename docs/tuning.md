# Speed tuning

Goal: maximise sentences/s per GPU without degrading quality. Thinking stays on for every
candidate. Every candidate configuration runs the full 25,500-item benchmark (namespace
`<fp>-w<N>`) and is gated with `luf bench compare --namespace w<N>` against the published
reference (margins: ADR-0006).

## Concurrency (`max_running_requests`, "window")

Calibration sweep on one A40 (Rennes abacus25, 300 sentences per level):

| window | sentences/s |
|---:|---:|
| 16 (reference) | 2.20 |
| 32 | 2.68 |
| 64 | 3.28 |
| 128 | 3.57 |

Gate results on the full benchmark (mixed A100-SXM4-40GB / A40 / L40S):

| candidate | ΔF1 | ΔMCC (95 % lower) | Δaccuracy | failed rate | worst language | gate |
|---|---:|---:|---:|---:|---|:--:|
| w64 | −0.0022 | −0.0070 (−0.0137) | −0.0026 | −0.14 pp | kn −0.043 | pass |
| w96 | −0.0020 | −0.0069 (−0.0136) | −0.0025 | −0.16 pp | ar −0.039 | pass |
| w128 | −0.0017 | −0.0047 (−0.0113) | −0.0021 | −0.09 pp | kn −0.050018 | **fail** |
| w96 on H100 NVL / A100-PCIe / RTX A5000 / L4 (`w96b`) | −0.0019 | −0.0059 (−0.0126) | −0.0010 | −0.27 pp | tg −0.037 | pass |

w128 fails only the per-language guard (limit 0.05), by 0.00002 on 300 items in one
language. The guard is not loosened. **Production uses window 96** for the A100-SXM4-40GB,
A40 and L40S profiles (measured: L40S 5.5 sentences/s at 96 against 3.7 at 16).
Other GPU types stay at 16 until a gated run covers them.

## Other candidates

| candidate | status |
|---|---|
| Radix cache ON | not run |
| `mem_fraction_static`, `cuda_graph_max_bs`, `chunked_prefill_size` | not run |
| Length-aware ordering | not run |
| `max_new_tokens` below 4096 | offline pre-screen: 3840 and 4000 pass, saving under 1 %; budget stays 4096 ([benchmark](benchmark.md)) |
| Multi-engine per node / data parallel | not run |
| SGLang / FlashInfer bumps | not run (would need re-gating) |

## Reproduce

```bash
luf g5k run --datasets benchmark --namespace w96 --window 96 \
    --gpu-models a100_sxm4_40gb,a40,l40s
luf bench compare --namespace w96 --label w96
```
