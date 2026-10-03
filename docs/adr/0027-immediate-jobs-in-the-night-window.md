# ADR-0027: Immediate-start jobs also in the night and weekend window

Status: accepted (2026-10-03)

## Context

By day (working days 09:00-19:00) the controller submits only immediate-start jobs of at most
1 h. They start where GPUs are free now, and 28-44 GPUs run. At night and on weekends it submitted
only queued `-t night` jobs. These wait behind the work of other users, and only 5-28 GPUs ran.
Short jobs submitted through the immediate path got 14-39 starts per hour at night.

## Decision

- `domain.policy.immediate_window` gives, for a night or weekend window, a window of at most 1 h
  (and not past the next working-day 09:00) without the `night` type. It gives nothing by day.
- In that window the controller first emits an immediate-start slot for each cluster: no
  `-t night`, the day walltime ladder (`--day-walltime-minutes`, then `--walltime-minutes`),
  and only where GPUs are free now. It never queues. The start check (`LATE_START`) is unchanged.
- The queued night slot stays. When the immediate slot takes the free GPUs, the night slot can only
  queue, up to `--night-max-queued-per-site`. Thus the same GPUs are never requested twice.
- `--immediate-in-night / --no-immediate-in-night` (`luf g5k run`, default on) controls this.
  Off gives the earlier behavior. Weekday behavior does not change.
- `max_jobs_per_site`, chunk reservation, the stale-besteffort reaper, the circuit breaker, backoffs
  and `usagepolicycheck -t` (per job or per batch, ADR-0019) apply to these jobs as to all others.

## Consequences

Free GPUs fill at once at night and on weekends. Jobs are short (30-60 min), so the setup cost
is paid more often. The policy has no daytime limit in this window, and the job ends before 09:00.
