# Output schema

## `labels/<input path>.parquet`

One row per in-scope sentence position of the input file with the same path.

| Column | Meaning |
|---|---|
| `label_id` | sha256 of dataset + join keys (primary key) |
| join keys | description: `description_identity`, `tag_key`, `sentence_index` · wiki: `sentence_id` · website: `polygon_id`, `field` (`website`/`contact_website`), `sentence_index` |
| `text_sha256` | sha256 of the exact sentence text |
| `decision` | `yes`, `no`, `failed`, `skipped_unsplit` |
| `parse_mode` / `failure_reason` | how the verdict was read, or why it failed |
| `generation_id` | → `generations/` (null for `skipped_unsplit`) |
| `language`, `config_fingerprint`, `input_revision` | provenance |

## `generations/<config fingerprint>/part-*.parquet`

One row per unique text: `generation_id`, `raw_output` (full, including reasoning),
`prompt_tokens`, `generated_tokens`, `finish_reason`, `truncated`, DSpark acceptance
counts, latency, model and draft revisions, prompt sha256, SGLang version, GPU, site,
OAR job id, code commit, timestamp.

```python
import duckdb

duckdb.sql("""
  SELECT l.*, g.raw_output
  FROM 'labels/polygons/*.parquet' l
  LEFT JOIN 'generations/*/*.parquet' g USING (generation_id)
  WHERE l.decision = 'yes'
""")
```


## Dataset card and world map

The card of each `-landuse` repo is rendered from counts made over the **published** tables
(never from a single run): per-file stats are cached in `published/<dataset>.stats.jsonl` and
backfilled from the Hub. Where the input has coordinates (`osm-polygon-description-tag`), the
card also embeds `assets/yes_share_map.png`: every `yes`/`no` sentence is placed at the centre of
its polygon's bounding box, binned into an H3 cell (resolution 3) and coloured by the share of
`yes` in the cell, centred on the dataset-wide share. Cells with fewer than 10 sentences are
grey. The land outline is Natural Earth 110m (public domain), vendored in
`src/landuse_filter/adapters/data/land_110m.json`. H3 and matplotlib come from the `map` extra,
installed by publish jobs only.
