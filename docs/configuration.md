# Configuration

Every setting has one documented source order, lowest to highest precedence:

1. **Defaults** in the code (`config.py`: model, revisions, generation parameters,
   `/tmp/luf-scratch` and `work` directories; `data/prompt.txt` is the prompt and is
   pinned by its SHA-256).
2. **`luf.toml`** at the repository root (or the file named by `$LUF_CONFIG`):
   namespace, bucket, Grid'5000 sites, walltimes, job caps, interval, CUDA module.
3. **Environment variables**: `LUF_<FIELD>` for each `luf.toml` field
   (`LUF_NAMESPACE`, `LUF_BUCKET`, `LUF_SITES` comma separated, `LUF_WALLTIME_MINUTES`,
   `LUF_NIGHT_WALLTIME_MINUTES`, `LUF_MAX_JOBS`, `LUF_MAX_JOBS_PER_SITE`,
   `LUF_INTERVAL_SECONDS`, `LUF_CUDA_MODULE`), plus `LUF_WORK` (local work tree,
   default `work`) and `LUF_SCRATCH` (node-local scratch, default `/tmp/luf-scratch`).
4. **CLI options** (for example `--bucket`, `--max-jobs`, `--work`), which win over everything.

The generation fingerprint depends only on the serving configuration in `config.py`,
never on `luf.toml` or the environment. See the [CLI reference](cli.md) for options.
