# Output schema

## `labels/<input path>.parquet`

This table has one row for each in-scope sentence position of the input file with the same path.

| Column | Meaning |
|---|---|
| `label_id` | sha256 of dataset + join keys (primary key) |
| join keys | description: `description_identity`, `tag_key`, `sentence_index` · wiki: `sentence_id` · website: `polygon_id`, `field` (`website`/`contact_website`), `sentence_index` |
| `text_sha256` | sha256 of the exact sentence text |
| `decision` | `yes`, `no`, `failed`, `skipped_unsplit` |
| `parse_mode` / `failure_reason` | how the project read the verdict, or why it failed |
| `generation_id` | → `generations/` (null for `skipped_unsplit`) |
| `language`, `config_fingerprint`, `input_revision` | provenance |

## `generations/<config fingerprint>/part-*.parquet`

This table has one row for each unique text: `generation_id`, `raw_output` (full, with reasoning),
`prompt_tokens`, `generated_tokens`, `finish_reason`, `truncated`, DSpark acceptance
counts, latency, model and draft revisions, prompt sha256, SGLang version, GPU, site,
OAR job id, code commit and timestamp.

The value `finish_reason = 'rule:no_letters'` shows that a rule decided the row, not the model (ADR-0024).
A sentence with no letter gets `raw_output` `</think>no`, zero generated tokens and `decision = no`.

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

`luf node publish --card-only` renders the same card from the bucket ledgers only (ADR-0021).

The project renders the card of each `-landuse` repo from counts of the **published** tables.
It never uses the counts of a single run. The project keeps the statistics for each file in
`published/<dataset>.stats.jsonl` and backfills them from the Hub. The card of each dataset has the
image `assets/yes_share_map.png` (ADR-0015). The project builds the image in these steps:

1. Place each `yes` or `no` sentence on the map. For description, use the center of the bounding box of its polygon. For website, use the `lat` and `lon` of the polygon. For wiki, use the first polygon with coordinates, the one with the smallest `polygon_id`, linked to the document of the sentence (as in the geographic plan).
2. Put the sentence in an H3 cell (resolution 3).
3. Color the cell by the share of `yes` in the cell. The center of the color scale is the share of `yes` in the whole dataset.

Cells with fewer than 10 sentences are grey. The land outline is Natural Earth 110m (public domain).
The project vendors it in `src/landuse_filter/adapters/data/land_110m.json`. H3 and matplotlib come
from the `map` extra. Only publish jobs install them.


## Viewer table

`viewer/<input path>.parquet` has only the data that a reader of the dataset viewer needs:
`sentence` (the text; the whole text for `skipped_unsplit`), `label` (`yes`, `no`, `failed`,
`skipped_unsplit`), `language` and `region` (the name of the input file).

The table follows these rules:

- It has only the sentences with a model answer or a deliberate skip. It never has `pending` sentences.
- It has only the sentences with at least 2 letters. Digits and punctuation do not count. Debris such as `-`, `|`, `...` or `7` is noise in the Hub viewer.
- Those rows stay in `labels/`, which keeps each row. Thus the viewer is always a subset of `labels/`.
- A file with no such row gets no viewer file, because the Hub viewer fails on parquet files with zero rows. The viewer cannot become stale, because resolved sentences never return to `pending`.

The card lists the table as the first, default config (`sentences`). Thus the Hub viewer opens on it.
Files that the project labelled before this table existed get their table on the next publish.


## Partial publication

Before ADR-0014, the project cut chunks in input-file order. The project generates repeated texts one time.
Thus a dataset at 20 % of its chunks had almost no input file with all sentences labelled.

To publish early, the project sends a file with some texts generated, but not all, to the Hub as a *partial* file.
In a partial file, `labels/` has each sentence and `viewer/` has only the answered sentences.
A sentence whose text has no answer has `decision = pending` (no `generation_id`, no `failure_reason`).

The project publishes and refreshes files as follows:

- It publishes a file that it did not publish before when one of its sentences is resolved.
- It refreshes the file when the file gains 1 % of its sentences since its last upload.
- It replaces the file with the final tables when the file is complete.
- Its `generations/` rows go with the complete file (the first file where a text occurs owns the generation of the text). Thus the labels of a partial file can reference generations that are not published yet.
- It records partial files in `published/<dataset>.partial.jsonl`, never in the main ledger. Thus they stay open. The card counts them (`N more partially`) and shows a `pending` row in the decision table.

The project refreshes the card after each partial commit (each 100 files). It uses the ledger of statistics
`published/<dataset>.stats.jsonl` and the cached map cells. Thus a publish job that stops at its walltime
leaves a card that matches the Hub. The project logs a failed refresh and tries it again at the next refresh and at the final one.
The project records the input mirror in `published/<dataset>.mirror.jsonl` (one line for each mirrored path and revision).
Thus a restart resumes where the job stopped.

A publish job also stops before its walltime (ADR-0022). It stops after a SIGUSR2 or SIGTERM,
or 6 minutes before the deadline. Then it does not start new files. It flushes, refreshes the card and saves the ledgers.
The `stopped` field of the JSON report gives the reason (`null` when the run finished).

`published/<dataset>.status.json` is the machine-readable end-of-run marker (ADR-0023). It has these fields:
`dataset`, `revision`, `files` (`total`, `complete`, `partial`, `unscanned`), `mirrored`,
`stopped` (reason or `null`) and `done` (true when nothing is left to publish).
