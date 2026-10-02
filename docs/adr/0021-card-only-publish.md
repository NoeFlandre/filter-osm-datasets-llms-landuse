# ADR-0021: Card-only publish

Status: accepted (2026-10-01, owner decision)

## Context

`luf node publish` restored the planner index (9 GB website, 26 GB wiki), downloaded each result
part and built the resolution index before it touched the card. On the 1-hour day walltime, the job
stopped during that setup. Thus the Hub card stayed stale after four jobs. The card needs none of this
data. It is a function of the bucket ledgers, the per-file stats, the admission gates and the config.

## Decision

* `luf node publish --card-only` (job mode `luf g5k cpu-job card ...`) does these steps:
    * It restores only the ledgers and the gates from the bucket.
    * It counts the missing per-file stats from the Hub.
    * It renders the card and the map with the same `_refresh_card` code and marker hash as a full run, and uploads them.
    * It saves the ledgers back after each counted batch. Thus a stopped run resumes.
* Coverage: `total_files` is the listing of the input repo, filtered by the patterns of the dataset.
  `labelled` is the `labels/` entries of the ledger among them. Partial files are the partial ledger
  minus the main ledger. A full run takes `total_files` from the scanned planner index. Thus the two
  differ only while planning has not scanned each input file (then card-only gives the truer number).
* A dataset with only a mirror (no labels, no partial files) keeps its card without change.
* The count is parallel (`COUNT_WORKERS` = 6 threads). The ledger appends stay in the main thread. If a file
  fails to count, the job logs it and skips it. The next run counts it again.

## Consequences

* You can refresh the card in minutes while generation or full publishes run.
* Card-only never uploads labels or generations. It cannot complete a dataset.
