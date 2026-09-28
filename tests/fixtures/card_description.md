---
license: odbl
pretty_name: osm-polygon-description-tag (land-use labels)
tags: [openstreetmap, land-use, land-cover, remote-sensing, geospatial]
dataset_status: in_progress
configs:
- config_name: labels
  data_files: "labels/**/*.parquet"
- config_name: generations
  data_files: "generations/**/*.parquet"
---
# osm-polygon-description-tag-landuse

Every in-scope sentence of [`NoeFlandre/osm-polygon-description-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag)
(revision `b4706eb`, mirrored here unchanged) labelled for land-use / land-cover
relevance by **LiquidAI/LFM2.5-2.6B@654f** with the DSpark draft **LiquidAI/LFM2.5-2.6B-DSpark@458c** (SGLang, BF16, greedy,
thinking mode, `max_new_tokens=4096`).

**Status: in_progress**: 3 / 386 input files labelled.

| Decision | Sentence positions |
|---|---:|
| `failed` | 1 |
| `no` | 5 |
| `skipped_unsplit` | 2 |
| `yes` | 10 |

## Files

* `labels/<input path>.parquet`: one row per sentence position. Columns: `label_id` and the
  join keys (description_identity, tag_key, sentence_index), `text_sha256`, `decision` (`yes`, `no`, `failed`, `skipped_unsplit`),
  `parse_mode`, `failure_reason` and `generation_id`.
* `generations/<fingerprint>/*.parquet`: one row per **unique** sentence text, with the
  raw output (including reasoning), token counts, finish reason, DSpark acceptance,
  GPU, site, job and code commit. Duplicate sentences share one generation.

| failure_reason | meaning |
|---|---|
| `truncated` | hit max_new_tokens before answering |
| `unclosed_think` | never closed its reasoning block |
| `empty` | nothing after the reasoning |
| `ambiguous` | answer mentions both labels |
| `non_english_token` | answered in another language (e.g. oui/ja), not mapped |
| `no_label` | no yes/no in the answer |
| `unsplit_upstream` | text not segmented upstream (skipped_unsplit, no LLM call) |

## Join recipe (DuckDB)

```sql
SELECT i.*, l.decision, g.raw_output
FROM 'language-v1/data/*.parquet' i
JOIN 'labels/language-v1/data/*.parquet' l USING (description_identity, tag_key)
LEFT JOIN 'generations/*/*.parquet' g USING (generation_id);
```

## Quality

Prompt and serving configuration are those of the benchmark
[`NoeFlandre/benchmark-llms-landuse-relevance`](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance).
Only GPU types that passed a pre-registered non-inferiority gate on the full
25,500-item benchmark were used. The table shows the Δ macro scores against the
published reference; margins are -0.02, failed rate +0.5 pp.

| GPU type | ΔF1 | ΔMCC (95 % low) | Δaccuracy | Δfailed |
|---|---:|---:|---:|---:|
| `a100_sxm4_40gb` | -0.0000 | -0.0014 (-0.0080) | +0.0016 | -0.33 pp |

Caveats: greedy decoding is not batch- or GPU-invariant, so about 12 % of individual
decisions would flip between two runs of the reference setup; aggregate quality is
what the gate guarantees.

## License and citation

Labels: ODbL, like the OpenStreetMap-derived input.
Code: https://github.com/NoeFlandre/filter-osm-datasets-llms-landuse (config
fingerprint `71dd8471f52321ab`).
