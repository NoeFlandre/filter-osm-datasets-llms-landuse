# Workload sizing

Measured with `luf plan` (resumable scan of every in-scope input file at a pinned revision).

| Dataset | Revision | Files | Sentences | + unsplit texts (`skipped_unsplit`) | Unique sentences (LLM calls) | Chunks (2,000) |
|---|---|---:|---:|---:|---:|---:|
| osm-polygon-description-tag | `b4706eb` | 386 | 978,773 | 78,229 | 461,463 (2.1× dedup) | 231 |
| osm-polygon-wikidata-and-wikipedia | — | — | — | — | — | — |
| osm-polygon-website-tag | — | — | 45.8 M (card) | — | — | — |

At the reference throughput (1.4 sentences/s per L40S at 16 concurrent requests),
description-tag needs ≈ 92 GPU-hours; tuned concurrency is measured in [tuning](benchmark.md).

Label rows = sentences + unsplit texts (description: 978,773 + 78,229 = 1,057,002). Each unique sentence is generated once and its label is copied to every identical sentence.
