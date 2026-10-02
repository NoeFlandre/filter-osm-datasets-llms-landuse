# ADR-0022: Publish jobs stop gracefully before their walltime

Status: accepted (2026-10-01)

## Context

OAR kills a job at its walltime. `luf node plan` handled the checkpoint signal, but `luf node
publish` did not. The website job ran out of time during the download of input files and left the card
stale for hours. The wiki job died three times inside the input mirror without a record of progress.

## Decision

- `luf node publish` handles SIGTERM, SIGUSR2 and SIGINT. It reads `LUF_JOB_DEADLINE_EPOCH`
  (exported by `scripts/node_job.sh` from `oarstat`) for jobs that get no signal in time.
- The pure `domain/stop.py` makes the decision: `should_stop(now, deadline, margin)` with a 6 minute
  margin, `stop_reason` (a signal has priority over the deadline) and parsers for the deadline.
- `publish` checks a latching stop request between input files (building, missing viewers) and
  between mirror batches. When a stop occurs, the job finishes the unit in progress, flushes the partial sink, ships the
  generations of the completed files and refreshes the card from the ledgers. Then
  `run_publish` saves each ledger (also if the run crashes). The command exits with code 0 and the report
  has `stopped`.
- A stop in the mirror leaves the mirror marker unwritten. The ledger equals the commits.

## Consequences

A job ends with the Hub, the card and the bucket ledgers consistent. The next job resumes
and does not redo published files. If the job cannot find the deadline, only the checkpoint signal
protects the job. The final card recount still runs after a stop. Thus the margin must cover it.
