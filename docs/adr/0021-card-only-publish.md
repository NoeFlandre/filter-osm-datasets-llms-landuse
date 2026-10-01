# ADR-0021: Card-only publish

Status: accepted (2026-10-01, owner decision)

## Context

`luf node publish` restored the planner index (9 GB website, 26 GB wiki), downloaded every result
part and built the resolution index before touching the card. On the 1-hour day walltime the job was
killed during that setup, so the Hub card stayed stale after four jobs. The card needs none of it:
it is a function of the bucket ledgers, the per-file stats, the admission gates and the config.

## Decision

* `luf node publish --card-only` (job mode `luf g5k cpu-job card ...`) restores only the ledgers and
  gates from the bucket, counts any missing per-file stats from the Hub, renders the card and map with
  the same `_refresh_card` code and marker hash as a full run, uploads it, and saves the ledgers back
  after every counted batch, so a stopped run resumes.
* Coverage: `total_files` is the input repo listing filtered by the dataset's patterns;
  `labelled` is the ledger's `labels/` entries among them; partial files are the partial ledger
  minus the main ledger. A full run takes `total_files` from the scanned planner index, so the two
  differ only while planning has not scanned every input file (card-only is then the truer number).
* A dataset with only a mirror (no labels, no partial files) keeps its card untouched.
* Counting is parallel (`COUNT_WORKERS` = 6 threads); ledger appends stay in the main thread. A file
  that fails to count is logged and skipped, and the next run counts it again.

## Consequences

* The card can be refreshed in minutes while generation or full publishes run.
* Card-only never uploads labels or generations; it cannot complete a dataset.
