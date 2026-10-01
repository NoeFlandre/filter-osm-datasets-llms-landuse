# Configuration

Every setting has one documented source order, lowest to highest precedence:

1. **Defaults** in the code (`config.py`: model, revisions, generation parameters,
   `/tmp/luf-scratch` and `work` directories; `data/prompt.txt` is the prompt and is
   pinned by its SHA-256).
2. **`luf.toml`** at the repository root (or the file named by `$LUF_CONFIG`):
   namespace, bucket, Grid'5000 sites, walltimes, job caps, interval, CUDA module.
3. **Environment variables**: `LUF_<FIELD>` for each `luf.toml` field
   (`LUF_NAMESPACE`, `LUF_BUCKET`, `LUF_SITES` comma separated, `LUF_WALLTIME_MINUTES`,
   `LUF_NIGHT_WALLTIME_MINUTES`, `LUF_NIGHT_FALLBACK_WALLTIME_MINUTES`, `LUF_MAX_JOBS`, `LUF_MAX_JOBS_PER_SITE`,
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
preferred one. By day nothing changes: jobs keep `walltime_minutes` (at most 1 h).

`--night-max-queued-per-site` (`luf g5k run` and `run-admission`) is the number of waiting jobs
allowed per site at night and on weekends. It defaults to `--max-queued-per-site`, so nothing
changes unless it is set. Example for production:
`--night-walltime-minutes 120 --night-fallback-walltime-minutes 30 --night-max-queued-per-site 15`.
See [ADR-0016](adr/0016-night-walltime-fallback.md).
