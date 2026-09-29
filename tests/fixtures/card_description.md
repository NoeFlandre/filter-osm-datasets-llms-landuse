---
license: odbl
pretty_name: osm-polygon-description-tag (land-use labels)
language: [multilingual]
task_categories: [text-classification]
tags: [openstreetmap, land-use, land-cover, geospatial]
dataset_status: in_progress
configs:
- config_name: sentences
  default: true
  data_files: "viewer/**/*.parquet"
- config_name: labels
  data_files: "labels/**/*.parquet"
- config_name: generations
  data_files: "generations/**/*.parquet"
---
# osm-polygon-description-tag-landuse

A land-use / land-cover relevance label for every sentence of [`NoeFlandre/osm-polygon-description-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag) (revision `b4706eb`). The input is mirrored here unchanged; labels and model outputs are separate tables that join back to it.

**In progress: 3 of 386 input files labelled, 20 rows, 12 unique texts sent to the model.**

| `decision` | Rows | Share | Meaning |
|---|---:|---:|---|
| `yes` | 10 | 50.0% | relevant to land use / land cover |
| `no` | 5 | 25.0% | not relevant |
| `failed` | 3 | 15.0% | the model output could not be parsed into yes/no |
| `skipped_unsplit` | 2 | 10.0% | text not segmented upstream; not sent to the model |
| **total** | **20** | | |

### Failed rows

3 rows (15.0% of all rows) have no
usable answer. Reasons:

| `failure_reason` | Rows | Share | Meaning |
|---|---:|---:|---|
| `truncated` | 2 | 66.7% | hit the token limit before answering |
| `no_label` | 1 | 33.3% | no yes/no in the answer |

## Tables

* `viewer/<input path>.parquet` (the default view): `sentence`, `label`, `language` and `region` (the input file), nothing else.
* `labels/<input path>.parquet`: one row per sentence. Join keys (description_identity, tag_key, sentence_index), `text_sha256`, `decision`, `parse_mode`, `failure_reason`, `generation_id`.
* `generations/<fingerprint>/*.parquet`: one row per **unique** text: raw output including the reasoning, token counts, finish reason, speculative-decoding statistics, GPU, site, job and code commit. Identical texts are generated once and share a `generation_id`.

```sql
SELECT i.*, l.decision, g.raw_output
FROM 'language-v1/data/*.parquet' i
JOIN 'labels/language-v1/data/*.parquet' l USING (description_identity, tag_key)
LEFT JOIN 'generations/*/*.parquet' g USING (generation_id);
```

## Method

`LiquidAI/LFM2.5-2.6B` (revision `654f`) with the speculative-decoding draft `LiquidAI/LFM2.5-2.6B-DSpark` (revision `458c`), served with SGLang (BF16), greedy decoding, thinking mode, at most 4,096 new tokens. The prompt is the one of [`NoeFlandre/benchmark-llms-landuse-relevance`](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance).

## Quality

Only GPU types that passed a pre-registered non-inferiority gate on the full 25,500-item benchmark generated labels. Differences are against the published reference run (one-sided 95 % lower bounds must stay above -0.02; failed-rate increase under +0.5 pp).

| GPU type | Generations | ΔF1 | ΔMCC (95 % low) | Δaccuracy | Δfailed |
|---|---:|---:|---:|---:|---:|
| `a100_sxm4_40gb` | 9 (75.0%) | -0.0000 | -0.0014 (-0.0080) | +0.0016 | -0.33 pp |
| `l40s` | 3 (25.0%) | -0.0018 | -0.0060 (-0.0123) | -0.0018 | -0.16 pp |

Greedy decoding is not batch- or GPU-invariant: about 12 % of individual decisions would flip between two runs of the reference setup. The gate guarantees aggregate quality, not per-row reproducibility.

## License and citation

Labels: ODbL, like the OpenStreetMap-derived input.

Code and configuration (fingerprint `71dd8471f52321ab`): https://github.com/NoeFlandre/filter-osm-datasets-llms-landuse
