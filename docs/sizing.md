# Workload sizing

The data below comes from `luf plan`. It is a resumable scan of each in-scope input file at a pinned revision.

| Dataset | Revision | Files | Sentences | + unsplit texts (`skipped_unsplit`) | Unique sentences (LLM calls) | Chunks (2,000) |
|---|---|---:|---:|---:|---:|---:|
| osm-polygon-description-tag | `b4706eb` | 386 | 978,773 | 78,229 | 461,463 (2.1× dedup) | 231 |
| osm-polygon-wikidata-and-wikipedia | `d6fc058` | 761 | 74,791,929 | 1,469,177 | 52,776,641 (1.42× dedup) | 26,389 |
| osm-polygon-website-tag | `6503db8` | 386 | 45,860,920 | 37,473 | 21,421,305 (2.14× dedup) | 10,711 |

The reference throughput is 1.4 sentences/s for each L40S at 16 concurrent requests.
At this throughput, description-tag needs approximately 92 GPU-hours.
The [tuning](tuning.md) page gives the tuned concurrency.

Label rows = sentences + unsplit texts (description: 978,773 + 78,229 = 1,057,002).
The project generates each unique sentence one time. It copies the label to each identical sentence.

Measured throughput (sentences/s for each GPU, thinking mode, benchmark prompts):

| GPU | window 16 (reference) | window 96 |
|---|---:|---:|
| A100-SXM4-40GB | 3.9 | not yet measured |
| A40 | 2.2 | 3.6 (window 128: 3.57) |
| L40S | 3.7 | 5.5 |

The total of unique sentences in the three datasets is approximately 74.7 M.
At 5 sentences/s for each GPU, this is approximately 4,000 GPU-hours.
Thus the website and wiki datasets need several days, even with more than 25 GPUs.
