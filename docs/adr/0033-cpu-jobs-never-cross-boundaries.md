# ADR-0033: CPU jobs never cross 09:00/19:00

Status: accepted (2026-10-06)

## Context

Grid'5000 warned us: wiki replan job 2071820 (submitted Thu 13:42) and wiki publish job 2071846
(submitted 16:40), default queue, walltime 1 h, waited in the queue, started at 18:30 and ran to
19:25, crossing 19:00. The GPU controller cancels any job that would start late, but the CPU path
(`_submit_cpu_job`, used by `cpu-job` and `publish-loop`) only clamped the walltime to the
submission-time window and never checked the real start.

## Decision

- `domain/cpu_job_guard.py` (pure): `cpu_job_verdict(submitted, walltime, expected_start)`. A job
  submitted in working-day daytime must start within 10 minutes (unknown start counts as late).
  Whatever the time, `[start, start + walltime]` must not strictly contain a working-day 09:00,
  nor a working-day 19:00 unless the job was submitted that day at/after 17:00. A night job with no
  predicted start is left to OAR's `-t night`, which confines it to the night.
- `application/cpu_guard.py`: after `oarsub`, `_submit_cpu_job` polls `oarstat -J` every 5 s
  (up to 120 s, issue #211: OAR may not have assigned a start yet) until the job is started or has a
  predicted start, then judges it, and on a refusal runs `oardel` and raises `CpuJobCancelledError`
  ("would start late; cancelled"). `cpu-job` exits 1; `publish-loop` logs it and retries on its next
  interval. Every CPU mode (plan, replan, publish, card, repair) goes through this one function.
- Watchdog (`stale_day_job`): at each cycle, `publish-loop` deletes its own Waiting job when it was
  submitted in daytime before 17:00 and waited over 10 minutes, or whenever it is 16:50 or later
  the same day. Only jobs of the loop's own name are touched.
- Zombies (`is_zombie`): a job listed Waiting for over 2 hours is a dead OAR record and no longer
  counts as live, so the loop submits a fresh one.
- No OAR deadline type is used: none is documented as safe for the default queue.
- Night and weekend behaviour is unchanged (`-t night`, walltime clamped to the next working-day
  09:00); the same verdict also rejects a night job predicted to start after that clamp.

## Consequences

A daytime CPU submission now costs one 20 s wait and one `oarstat`. A busy site cancels and retries
instead of queueing. A night job legitimately waiting more than 2 hours stops counting as live in the
loop, so a duplicate may be submitted; the next check keeps the policy intact.
