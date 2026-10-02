# ADR-0023: Keep publishing until the bucket says done

Status: accepted (2026-10-01)

## Context

Even with a graceful stop (ADR-0022), a publish job covers only the work that fits in one walltime (1 hour by
day). A person had to submit it again by hand, often hours late. Thus the Hub stayed stale.

## Decision

- Each publish run writes `published/<dataset>.status.json` to the bucket (file totals, mirrored,
  stopped, `done`). `done` is the pure property `PublishStatus.done` in `domain/publish_loop.py`.
- `luf g5k publish-loop` submits `cpu-job publish` when no job of that dataset is live on the
  site and the marker of the same revision is not done. The pure `decide(done, live)` makes the decision.
  The sleep is `pause(failures, interval, cap)`, a bounded exponential backoff.
- Publish jobs have the name `luf-publish-<dataset>`. Other CPU modes keep `luf-plan-<dataset>`. Thus
  neither loop blocks the other.
- The loop logs a slow `usagepolicycheck`, ssh timeouts and bucket errors and tries again. They are never fatal.
  Other exceptions (bugs) stop the loop.

## Consequences

The marker says only what the last run saw. A run that races a planner that continues to scan reports
`unscanned > 0` and the loop continues. Revisions never mix: the loop ignores a marker of another revision.
