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

`--immediate-in-night / --no-immediate-in-night` (`luf g5k run`; default on) makes the controller
also submit immediate-start jobs (no `-t night`, at most 1 h, only where GPUs are free now) at night
and on weekends, in addition to the queued night jobs. See
[ADR-0034](adr/0034-immediate-jobs-in-the-night-window.md).

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

## Part flush thresholds

A node uploads one part (parquet + manifest, one commit, about 2 Hub API requests) per 2048
results or 300 s, whichever comes first (`FLUSH_EVERY`, `FLUSH_SECONDS` in
`application/node.py`). A graceful stop flushes everything; an abrupt kill loses at most 300 s.
Old small parts mix freely with new ones. See [ADR-0030](adr/0030-fewer-larger-part-uploads.md).

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

## Incremental ingest

The controller no longer lists all of `parts/<fp>/` on every pull. It lists the chunk prefix of
each assignment that ended since the previous pull (and `jobs/` then), the chunks of live
assignments every 900 s (`LIVE_INTERVAL`), and everything on the first pull after a start and
every 3600 s (`RECONCILE_INTERVAL`). Both constants are in `application/staging.py`. Restart the
controller to apply. See [ADR-0029](adr/0029-incremental-ingest.md).

## Background ingest

Ingest runs in its own thread, never in the cycle: a rate-limited Hub cannot delay submission.
Every 180 s it runs one bounded pull (short 429 budget: 3 attempts, at most 60 s per wait; the
first listing that stays rate-limited ends the pull, no new listing after 240 s) and logs
`ingest: full|incremental pull, N listings, M new manifests, F rate-limited, D deferred, S s`.
A long gap between these lines means ingest is stalled. The first pull after a start is the full
catch-up, in the background. See [ADR-0031](adr/0031-ingest-never-starves-submission.md).

## Hub rate limit (HTTP 429)

The Hub allows 1000 API calls per 5 minutes per account. The bucket adapter
(`adapters/remote.py`, used by the controllers, node jobs and the CLI) retries `ls`, `put`, `get`,
`delete` and `ensure` on HTTP 429. It waits the `Retry-After` seconds of the response (plus 1 s and
up to 1 s of jitter), or backs off exponentially with jitter when the header is missing. Limits
(constants in `domain/retry.py`, not configurable at run time): 10 calls in all, one wait at most
300 s, all waits together at most 1200 s. It logs one line per wait. Then it re-raises the original
error. A 429 on a later page of a listing retries that page only. Other errors are not retried.
See [ADR-0028](adr/0028-hub-rate-limit-retry.md).

## Job progress (publish, card)

Publish and card jobs print one flushed line per phase to their OAR stdout
(`luf: [<s>s] <phase> k=v ...`: planner index and resolution snapshot downloads with bytes and
seconds, parts read / total and rows/s, files built, each Hub commit with its file count, short id
and seconds, the stop reason). The same state is written to `published/<dataset>.progress.json`
in the bucket at most once per 60 s (`PUT_SECONDS`, `TICK_SECONDS` in
`application/job_progress.py`). Read it from the laptop with one bucket get:

```
luf g5k publish-status --dataset <dataset>
```

See [ADR-0032](adr/0032-observable-publish-jobs.md).

## CPU-job boundary guard (fixed)

`cpu-job` and `publish-loop` take no option for it. The constants live in
`domain/cpu_job_guard.py`: start tolerance 10 minutes, watchdog cutoff 16:50, evening submission
from 17:00, zombie age 2 hours. After `oarsub`, a daytime job that OAR predicts to start late (or
to cross 09:00/19:00) is deleted with `oardel` and reported as `would start late; cancelled`.
See [ADR-0033](adr/0033-cpu-jobs-never-cross-boundaries.md).
