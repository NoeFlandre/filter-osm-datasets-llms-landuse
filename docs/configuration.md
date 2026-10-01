# Configuration

Every setting has one documented source order, lowest to highest precedence:

1. **Defaults** in the code (`config.py`: model, revisions, generation parameters,
   `/tmp/luf-scratch` and `work` directories; `data/prompt.txt` is the prompt and is
   pinned by its SHA-256).
2. **`luf.toml`** at the repository root (or the file named by `$LUF_CONFIG`):
   namespace, bucket, Grid'5000 sites, walltimes, job caps, interval, CUDA module.
3. **Environment variables**: `LUF_<FIELD>` for each `luf.toml` field
   (`LUF_NAMESPACE`, `LUF_BUCKET`, `LUF_SITES` comma separated, `LUF_WALLTIME_MINUTES`,
   `LUF_NIGHT_WALLTIME_MINUTES`, `LUF_NIGHT_FALLBACK_WALLTIME_MINUTES`, `LUF_MAX_JOBS`, `LUF_MAX_JOBS_PER_SITE`, `LUF_POLICY_CHECK`, `LUF_SUBMIT_WORKERS`,
   `LUF_INTERVAL_SECONDS`, `LUF_CUDA_MODULE`), plus `LUF_WORK` (local work tree,
   default `work`) and `LUF_SCRATCH` (node-local scratch, default `/tmp/luf-scratch`).
4. **CLI options** (for example `--bucket`, `--max-jobs`, `--work`), which win over everything.

The generation fingerprint depends only on the serving configuration in `config.py`,
never on `luf.toml` or the environment. See the [CLI reference](cli.md) for options.

## Night jobs: long walltime with a fallback

Each GPU job spends about 6 minutes on environment build and SGLang start, so longer night
jobs waste a smaller share of their time. At night and on weekends the controller therefore
tries `night_walltime_minutes` (the preferred, long value, `--night-walltime-minutes`, default
120) first. If that job is refused, or OAR predicts it would not start in time (start drifted
or late), the same slot is retried once with `night_fallback_walltime_minutes`
(`--night-fallback-walltime-minutes`, default 30), with chunks re-assigned for the shorter
walltime. Only when the fallback also fails is the cluster backed off for 30 minutes. Both
values are capped by the window (the next working-day 09:00) and the fallback never exceeds the
preferred one. By day jobs keep `walltime_minutes` (at most 1 h) unless
`--day-walltime-minutes` is set (next section).

`--night-max-queued-per-site` (`luf g5k run` and `run-admission`) is the number of waiting jobs
allowed per site at night and on weekends. It defaults to `--max-queued-per-site`, so nothing
changes unless it is set. Example for production:
`--night-walltime-minutes 120 --night-fallback-walltime-minutes 30 --night-max-queued-per-site 15`.
See [ADR-0016](adr/0016-night-walltime-fallback.md).

## Day jobs: long walltime with a fallback and a self-throttle

`--day-walltime-minutes` (`luf g5k run` and `run-admission`; `day_walltime_minutes` in luf.toml,
`LUF_DAY_WALLTIME_MINUTES`) is the preferred walltime from 09:00 to 19:00. Unset, it equals
`--walltime-minutes` and behaviour is exactly as before. When set (for example 60 with
`--walltime-minutes 30`), a default-queue job first tries `min(day walltime, window maximum)`; if
that is refused or would not start immediately, the same slot is retried once with
`--walltime-minutes` before the cluster is backed off. Production-queue clusters (`abaca`) have
no start-time restriction and use the day value directly, falling back to `--walltime-minutes`
on refusal; at night they keep `--walltime-minutes`.

Self-throttle: after `--day-long-max-failures` (`day_long_max_failures`, default 3) consecutive
failed long attempts on a cluster, only the short walltime is used there for one hour, then long
attempts resume. The controller log has one line per event:
`long_walltime fallback <site>/<cluster> 60->30` and `long_walltime throttle <site>/<cluster> ...`.

To revert, unset `--day-walltime-minutes` (or set it equal to `--walltime-minutes`). Production:
`--walltime-minutes 30 --day-walltime-minutes 60`. See
[ADR-0017](adr/0017-day-walltime-fallback.md).

## Over-assignment (`chunk_overflow`)

`--chunk-overflow` (`luf g5k run` and `run-admission`; `chunk_overflow` in luf.toml,
`LUF_CHUNK_OVERFLOW`; float, at least 1.0, default 1.2) sets how much work a job receives:
profile `sentences_per_second` x (walltime - setup) x this factor. Jobs usually finish their
chunks early (median useful fraction about 0.5), so a larger factor keeps the GPU busy until the
OAR checkpoint signal (5 minutes before the walltime ends), when the node flushes finished texts
and exits; unfinished chunks return to the pool. Production: `--chunk-overflow 1.6`.
To revert, leave the option unset (1.2). See [ADR-0018](adr/0018-over-assignment.md).

## Usage-policy check cadence (`policy_check`)

`--policy-check` (`luf g5k run` and `run-admission`; `policy_check` in luf.toml,
`LUF_POLICY_CHECK`; `per-job` or `per-batch`, default `per-job`) sets when
`usagepolicycheck -t` runs. `per-job` checks before and after every submission. `per-batch`
checks each site once before its first submission of a cycle and once after its last (no
check for a site without submissions); a failing check stops further submissions to that site
for the cycle, and a violation found after the batch is logged as `POLICY VIOLATION`. The check
takes minutes per call, so per-batch lets a controller submit many more jobs per hour.
Production: `--policy-check per-batch`. To revert, leave the option unset. See
[ADR-0019](adr/0019-batched-policy-check.md).

## Parallel submission (`submit_workers`)

`--submit-workers N` (`luf g5k run` and `run-admission`; `submit_workers` in luf.toml,
`LUF_SUBMIT_WORKERS`; integer, at least 1, default 1) submits to up to N sites at the same
time. The controller first decides, in order and deterministically, which slot gets which
chunks; then one worker per site launches that site's jobs in turn, and results are collected in
plan order. While the workers run, the main thread ingests results every `PULL_INTERVAL`
(180 s) so progress and chunk counts stay current; the cycle ingests again at its end. The cycle logs `cycle submitted N jobs in S s (sites K)`. Production:
`--submit-workers 8`. To revert, use `--submit-workers 1`. See
[ADR-0020](adr/0020-parallel-submission.md).
