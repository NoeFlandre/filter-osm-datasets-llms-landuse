# Benchmark parity

Reference: `LiquidAI/LFM2.5-2.6B+DSpark-throughput-b16` on the 25,500-item
multilingual benchmark (85 languages × 300). Our metrics code reproduces its
published macro scores exactly: F1 0.8104, MCC 0.5644, accuracy 0.7623, failed 2.42 %.

## Token budget (offline simulation, 2026-09-28)

Greedy decoding is prefix-deterministic. Thus you can simulate a lower `max_new_tokens` cap
from the reference generations. In the simulation, the longer outputs become `failed`.

| cap | truncated | Δ macro-F1 | Δ macro-MCC | Δ macro-acc (95 % low) | Δ failed (95 % high) | gate |
|---:|---:|---:|---:|---:|---:|:--:|
| 1024 | 37.7 % | +0.045 | +0.089 | −0.249 | +35.8 pp | ✗ |
| 2048 | 11.4 % | +0.017 | +0.038 | −0.056 | +9.3 pp | ✗ |
| 3072 | 4.8 % | +0.005 | +0.010 | −0.015 | +2.5 pp | ✗ |
| 3584 | 3.3 % | +0.002 | +0.003 | −0.006 | +0.97 pp | ✗ |
| 3840 | 2.8 % | +0.001 | +0.002 | −0.003 | +0.44 pp | ✓ |
| 4000 | 2.5 % | +0.000 | +0.001 | −0.001 | +0.13 pp | ✓ |

F1 and MCC *increase* because the benchmark leaves the failed items out of their cells.
The accuracy and the failed-rate guard show the real loss. The caps that pass save
less than 1 % of the generated tokens. Thus **the budget stays at 4096**.

To reproduce, run `luf bench budget`.
