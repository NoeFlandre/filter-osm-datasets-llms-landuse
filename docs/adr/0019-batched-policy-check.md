# ADR-0019: Run the usage-policy check once per site batch

Status: accepted (2026-10-01)

## Context

`usagepolicycheck -t` (run over ssh by `g5k.policy_check`) now takes approximately 2.5 minutes for each call
on each site. The controller ran it two times for each job (before and after `oarsub`) and submitted
jobs one after the other. Thus a controller launched approximately 12 jobs each hour, the controller refilled free GPUs
slowly, and we held 2-6 GPUs instead of 15-20.

## Decision

`--policy-check per-batch` (`luf g5k run` and `run-admission`; `policy_check` in luf.toml,
`LUF_POLICY_CHECK`; default `per-job`):

- For each site and cycle, the check runs one time before the first submission to the site and one time
  after the last one (`domain/policy_check.check_due`, `application/policy_gate.PolicyGate`).
- A site that gets no submission attempt in a cycle has no check.
- If a pre-check fails, that submission fails as before. The failure blocks the site for the rest of the
  cycle (no more oarsub, no walltime fallback). The next cycle checks again.
- The controller logs a violation that the post-check finds as `POLICY VIOLATION: <site>: ...`
  (the jobs already exist, exactly as when the per-job post-check failed). The violation blocks the site.
  The pre-check of the next cycle runs the check again. It refuses to submit while the check still fails.

`per-job` does not change: a check before and after each submission.

## Compliance argument

The check validates the state of our submissions on a site, not an individual oarsub. In
`per-batch` mode, a passing check of the site in the same cycle still precedes each submission.
A check of the whole batch follows each batch. Thus the controller detects a violation at
the end of the cycle that caused it. The controller submits nothing more until the check passes
again. One batch bounds the window in which an unchecked submission can exist.

## Reverting

Unset the flag (or set `policy_check = "per-job"` / `LUF_POLICY_CHECK=per-job`).
