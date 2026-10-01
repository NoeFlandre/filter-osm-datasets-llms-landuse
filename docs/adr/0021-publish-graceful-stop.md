# ADR-0021: Publish jobs stop gracefully before their walltime

Status: accepted (2026-10-01)

## Context

OAR kills a job at its walltime. `luf node plan` handled the checkpoint signal, but `luf node
publish` did not: the website job ran out of time while downloading input files and left the card
stale for hours, and the wiki job died inside the input mirror three times without recording
progress.

## Decision

- `luf node publish` handles SIGTERM, SIGUSR2 and SIGINT, and reads `LUF_JOB_DEADLINE_EPOCH`
  (exported by `scripts/node_job.sh` from `oarstat`) for jobs that never get a signal in time.
- The decision is the pure `domain/stop.py`: `should_stop(now, deadline, margin)` with a 6 minute
  margin, `stop_reason` (a signal wins over the deadline) and parsers for the deadline.
- `publish` checks a latching stop request between input files (building, missing viewers) and
  between mirror batches. A stop finishes the unit in hand, flushes the partial sink, ships the
  generations of the completed files, refreshes the card from the ledgers, and
  `run_publish` saves every ledger (also if the run crashes). The command exits 0 and the report
  carries `stopped`.
- A stop in the mirror leaves the mirror marker unwritten and the ledger equal to the commits.

## Consequences

A job ends with the Hub, the card and the bucket ledgers consistent and the next job resumes
without redoing published files. If the deadline cannot be determined only the checkpoint signal
protects the job. The final card recount still runs after a stop, so the margin must cover it.
