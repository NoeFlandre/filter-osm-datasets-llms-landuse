# Workload sizing

Measured with `luf plan` (resumable scan of every in-scope input file at a pinned revision).

| Dataset | Revision | Files | Sentences | + unsplit texts (`skipped_unsplit`) | Unique sentences (LLM calls) | Chunks (2,000) |
|---|---|---:|---:|---:|---:|---:|
| osm-polygon-description-tag | `b4706eb` | 386 | 978,773 | 78,229 | 461,463 (2.1× dedup) | 231 |
| osm-polygon-wikidata-and-wikipedia | `d6fc058` | 761 | 74,791,929 | 1,469,177 | 52,776,641 (1.42× dedup) | 26,389 |
| osm-polygon-website-tag | `6503db8` | 386 | 45,860,920 | 37,473 | 21,421,305 (2.14× dedup) | 10,711 |

At the reference throughput (1.4 sentences/s per L40S at 16 concurrent requests),
description-tag needs ≈ 92 GPU-hours; tuned concurrency is measured in [tuning](benchmark.md).

Label rows = sentences + unsplit texts (description: 978,773 + 78,229 = 1,057,002). Each unique sentence is generated once and its label is copied to every identical sentence.

Measured throughput (sentences/s per GPU, thinking mode, benchmark prompts):

| GPU | window 16 (reference) | window 96 |
|---|---:|---:|
| A100-SXM4-40GB | 3.9 | not yet measured |
| A40 | 2.2 | 3.6 (window 128: 3.57) |
| L40S | 3.7 | 5.5 |

Total unique sentences across the three datasets: about 74.7 M, i.e. roughly 4,000 GPU-hours
at 5 sentences/s per GPU. Website and wiki are therefore multi-day jobs even with 25+ GPUs.
