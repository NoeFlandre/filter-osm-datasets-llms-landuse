# ADR-0019: Run the usage-policy check once per site batch

Status: accepted (2026-10-01)

## Context

`usagepolicycheck -t` (run over ssh by `g5k.policy_check`) now takes about 2.5 minutes per call
on every site. The controller ran it twice per job (before and after `oarsub`) and submitted
jobs one after another, so a controller launched about 12 jobs per hour, freed GPUs were
refilled slowly, and we held 2-6 GPUs instead of 15-20.

## Decision

`--policy-check per-batch` (`luf g5k run` and `run-admission`; `policy_check` in luf.toml,
`LUF_POLICY_CHECK`; default `per-job`):

- per site and cycle, the check runs once before the first submission to the site and once
  after the last one (`domain/policy_check.check_due`, `application/policy_gate.PolicyGate`);
- a site that gets no submission attempt in a cycle is never checked;
- a failing pre-check fails that submission as before and blocks the site for the rest of the
  cycle (no further oarsub, no walltime fallback); the next cycle checks again;
- a violation found by the post-check is logged as `POLICY VIOLATION: <site>: ...` (the jobs
  already exist, exactly as when the per-job post-check failed) and blocks the site; the next
  cycle's pre-check re-runs the check and refuses to submit while it still fails.

`per-job` is unchanged: a check before and after every submission.

## Compliance argument

The check validates the state of our submissions on a site, not an individual oarsub. In
per-batch mode every submission is still preceded by a passing check of the site in the same
cycle and every batch is followed by a check of the whole batch, so a violation is detected at
the end of the cycle that caused it and nothing further is submitted until the check passes
again. The window in which an unchecked submission can exist is bounded by one batch.

## Reverting

Unset the flag (or set `policy_check = "per-job"` / `LUF_POLICY_CHECK=per-job`).
