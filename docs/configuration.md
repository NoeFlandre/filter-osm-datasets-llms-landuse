# Configuration

Each setting has one documented source order. The list below goes from the lowest to the highest precedence:

1. **Defaults** in the code (`config.py`: model, revisions, generation parameters,
   the `/tmp/luf-scratch` and `work` directories). `data/prompt.txt` is the prompt.
   Its SHA-256 pins it.
2. **`luf.toml`** at the repository root (or the file that `$LUF_CONFIG` names):
   namespace, bucket, Grid'5000 sites, walltimes, job caps, interval, CUDA module.
3. **Environment variables**: `LUF_<FIELD>` for each `luf.toml` field
   (`LUF_NAMESPACE`, `LUF_BUCKET`, `LUF_SITES` comma separated, `LUF_WALLTIME_MINUTES`,
   `LUF_NIGHT_WALLTIME_MINUTES`, `LUF_NIGHT_FALLBACK_WALLTIME_MINUTES`, `LUF_MAX_JOBS`, `LUF_MAX_JOBS_PER_SITE`, `LUF_POLICY_CHECK`, `LUF_SUBMIT_WORKERS`,
   `LUF_INTERVAL_SECONDS`, `LUF_CUDA_MODULE`). Also `LUF_WORK` (local work tree,
   default `work`) and `LUF_SCRATCH` (node-local scratch, default `/tmp/luf-scratch`).
4. **CLI options** (for example `--bucket`, `--max-jobs`, `--work`). They override all other sources.

The generation fingerprint depends only on the serving configuration in `config.py`.
It never depends on `luf.toml` or the environment. For the options, see the [CLI reference](cli.md).

## Night jobs: long walltime with a fallback

Each GPU job uses approximately 6 minutes for the environment build and the SGLang start.
Thus a long night job wastes a smaller share of its time. At night and on weekends, the controller
tries `night_walltime_minutes` first. This is the preferred long value (`--night-walltime-minutes`,
default 120). The controller retries the same slot one time with
`night_fallback_walltime_minutes` (`--night-fallback-walltime-minutes`, default 30) in these cases:

- OAR refuses the job.
- OAR predicts that the job does not start in time (the start is late).

The controller assigns the chunks again for the shorter walltime. If the fallback also fails,
the controller backs off the cluster for 30 minutes. The window (the next working-day 09:00) caps both
values. The fallback never exceeds the preferred value. By day, jobs keep `walltime_minutes`
(maximum 1 h), unless you set `--day-walltime-minutes` (next section).

`--night-max-queued-per-site` (`luf g5k run` and `run-admission`) is the number of waiting jobs
that each site can have at night and on weekends. It defaults to `--max-queued-per-site`.
Thus nothing changes unless you set it. Example for production:
`--night-walltime-minutes 120 --night-fallback-walltime-minutes 30 --night-max-queued-per-site 15`.
See [ADR-0016](adr/0016-night-walltime-fallback.md).

## Day jobs: long walltime with a fallback and a self-throttle

`--day-walltime-minutes` (`luf g5k run` and `run-admission`; `day_walltime_minutes` in luf.toml,
`LUF_DAY_WALLTIME_MINUTES`) is the preferred walltime from 09:00 to 19:00. If you do not set it, it equals
`--walltime-minutes` and the behavior does not change. When you set it (for example 60 with
`--walltime-minutes 30`), a default-queue job first tries `min(day walltime, window maximum)`.
If OAR refuses that job, or the job does not start immediately, the controller retries the same slot
one time with `--walltime-minutes` before it backs off the cluster. Production-queue clusters (`abaca`)
have no start-time restriction. They use the day value directly and fall back to `--walltime-minutes`
if OAR refuses. At night they keep `--walltime-minutes`.

Self-throttle: after `--day-long-max-failures` (`day_long_max_failures`, default 3) consecutive
failed long attempts on a cluster, the controller uses only the short walltime there for one hour.
Then the long attempts resume. The controller log has one line for each event:
`long_walltime fallback <site>/<cluster> 60->30` and `long_walltime throttle <site>/<cluster> ...`.

To revert, unset `--day-walltime-minutes` (or set it equal to `--walltime-minutes`). Production:
`--walltime-minutes 30 --day-walltime-minutes 60`. See
[ADR-0017](adr/0017-day-walltime-fallback.md).

## Over-assignment (`chunk_overflow`)

`--chunk-overflow` (`luf g5k run` and `run-admission`; `chunk_overflow` in luf.toml,
`LUF_CHUNK_OVERFLOW`; float, at least 1.0, default 1.2) sets the quantity of work that a job gets:
profile `sentences_per_second` x (walltime - setup) x this factor. Jobs usually finish their
chunks early (median useful fraction approximately 0.5). A larger factor keeps the GPU busy until the
OAR checkpoint signal (5 minutes before the walltime ends). Then the node flushes the finished texts
and exits. The unfinished chunks return to the pool. Production: `--chunk-overflow 1.6`.
To revert, do not set the option (1.2). See [ADR-0018](adr/0018-over-assignment.md).

## Stale besteffort jobs

`--stale-besteffort-minutes` (`luf g5k run` and `run-admission`; default 20) cancels our
besteffort jobs that are still Waiting that long after submission (other users took their GPUs).
The controller marks the assignment `cancelled_stale`, releases its chunks and backs off the cluster
for 10 minutes. The controller never cancels jobs in other queues (night, exotic). See
[ADR-0025](adr/0025-stale-besteffort-reaper.md).

## Usage-policy check cadence (`policy_check`)

`--policy-check` (`luf g5k run` and `run-admission`; `policy_check` in luf.toml,
`LUF_POLICY_CHECK`; `per-job` or `per-batch`, default `per-job`) sets when
`usagepolicycheck -t` runs.

- `per-job` checks before and after each submission.
- `per-batch` checks each site one time before its first submission of a cycle and one time after its last. A site without submissions has no check. If a check fails, the controller stops more submissions to that site for the cycle. The controller logs a violation that it finds after the batch as `POLICY VIOLATION`.

The check takes minutes for each call. Thus `per-batch` lets a controller submit many more jobs each hour.
Production: `--policy-check per-batch`. To revert, do not set the option. See
[ADR-0019](adr/0019-batched-policy-check.md).

## Parallel submission (`submit_workers`)

`--submit-workers N` (`luf g5k run` and `run-admission`; `submit_workers` in luf.toml,
`LUF_SUBMIT_WORKERS`; integer, at least 1, default 1) submits to a maximum of N sites at the same
time. The controller first decides, in order and deterministically, which slot gets which
chunks. Then one worker for each site launches the jobs of that site in turn. The controller collects the results in
plan order. While the workers run, the main thread ingests results each `PULL_INTERVAL`
(180 s). Thus the progress and the chunk counts stay current. The cycle ingests again at its end.
The cycle logs `cycle submitted N jobs in S s (sites K)`. Production:
`--submit-workers 8`. To revert, use `--submit-workers 1`. See
[ADR-0020](adr/0020-parallel-submission.md).
