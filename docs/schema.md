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
backfilled from the Hub. Every dataset's card embeds `assets/yes_share_map.png` (ADR-0015):
every `yes`/`no` sentence is placed (description: the centre of its polygon's bounding box;
website: the polygon's `lat`/`lon`; wiki: the first polygon, by smallest `polygon_id` with
coordinates, linked to the sentence's document, as in the geographic plan), binned into an H3
cell (resolution 3) and coloured by the share of `yes` in the cell, centred on the dataset-wide
share. Cells with fewer than 10 sentences are
grey. The land outline is Natural Earth 110m (public domain), vendored in
`src/landuse_filter/adapters/data/land_110m.json`. H3 and matplotlib come from the `map` extra,
installed by publish jobs only.


## Viewer table

`viewer/<input path>.parquet` mirrors `labels/` row for row but carries only what a reader of the
dataset viewer needs: `sentence` (the text; the whole text for `skipped_unsplit`), `label`
(`yes`, `no`, `failed`, `skipped_unsplit`), `language` and `region` (the input file's name). The
card lists it as the first, default config (`sentences`), so the Hub viewer opens on it. Files
labelled before this table existed get theirs on the next publish.


## Partial publication

Until ADR-0014 chunks were cut in input-file order and, because repeated texts are generated
once, a dataset at 20 % of its chunks had almost no input file with every sentence labelled. To publish early, a file with some but not all
texts generated goes to the Hub as a *partial* file: `labels/` and `viewer/` hold every sentence,
and a sentence whose text has no answer yet has `decision = pending` (no `generation_id`, no
`failure_reason`). A partial file is refreshed when it gained 1 % of its sentences since its last
upload, and replaced by the final tables once it is complete. Its `generations/` rows ship with
the complete file (the first file where a text appears owns the text's generation), so a partial
file's labels can reference generations that are not published yet. Partial files are recorded in
`published/<dataset>.partial.jsonl`, never in the main ledger, so they stay open; the card counts
them (`N more partially`) and shows a `pending` row in the decision table.
