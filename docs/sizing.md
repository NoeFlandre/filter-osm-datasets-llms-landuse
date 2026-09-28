# Workload sizing

Measured with `luf plan` (resumable scan of every in-scope input file at a pinned revision).

| Dataset | Revision | Files | Sentence positions | `skipped_unsplit` | Unique texts (LLM calls) | Chunks (2,000) |
|---|---|---:|---:|---:|---:|---:|
| osm-polygon-description-tag | `b4706eb` | 386 | 1,057,002 | 78,229 (7.4 %) | 461,463 (2.3× dedup) | 231 |
| osm-polygon-wikidata-and-wikipedia | — | — | — | — | — | — |
| osm-polygon-website-tag | — | — | 45.8 M (card) | — | — | — |

At the reference throughput (1.4 sentences/s per L40S at 16 concurrent requests),
description-tag needs ≈ 92 GPU-hours; tuned concurrency is measured in [tuning](benchmark.md).
