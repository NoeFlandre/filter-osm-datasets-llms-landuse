# ADR-0023: Keep publishing until the bucket says done

Status: accepted (2026-10-01)

## Context

Even with a graceful stop (ADR-0022) a publish job covers only what fits in one walltime (1 hour by
day), and someone had to resubmit it by hand, often hours late, which left the Hub stale.

## Decision

- Every publish run writes `published/<dataset>.status.json` to the bucket (file totals, mirrored,
  stopped, `done`). `done` is the pure property `PublishStatus.done` in `domain/publish_loop.py`.
- `luf g5k publish-loop` submits `cpu-job publish` whenever no job of that dataset is live on the
  site and the marker of the same revision is not done. The decision is the pure
  `decide(done, live)`; the sleep is `pause(failures, interval, cap)`, a bounded exponential backoff.
- Publish jobs are named `luf-publish-<dataset>`; other CPU modes keep `luf-plan-<dataset>`, so
  neither loop blocks on the other.
- Slow `usagepolicycheck`, ssh timeouts and bucket errors are logged and retried, never fatal.
  Other exceptions (bugs) stop the loop.

## Consequences

The marker only says what the last run saw: a run racing a planner that keeps scanning reports
`unscanned > 0` and the loop continues. Revisions never mix: a marker of another revision is ignored.
