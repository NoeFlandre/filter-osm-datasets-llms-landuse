# ADR-0008 — Node environments: reuse a finished site env, else build on node-local disk

**Status:** accepted · 2026-09-28 (revised two times, see Consequences)

**Context.** Users cannot use Docker on G5K nodes. The NFS home performance differs much between sites.
**Decision.** `scripts/node_job.sh` reuses `~/luf/cache/venv-<lock sha>` only if it is finished (`.ready`). Otherwise it builds a private env with `uv sync --frozen --no-install-project --extra gpu` on node-local `/tmp`. The system removes this env when the job exits. The job always runs **its own deployed code**. The script resolves `CODE` to an absolute path and puts it on `PYTHONPATH`. The job never imports the project from an installed copy. The script loads the CUDA toolkit with `module load` for SGLang/DeepEP. The `bin` directory of the venv is on `PATH` for the ninja of FlashInfer. The model weights and the FlashInfer JIT caches are in `~/luf/cache/hf` and `~/luf/cache/flashinfer`.
**Consequences.**

- Grenoble used a whole job to copy an 8 GB venv to NFS (job 3122433). Node-local builds take 1–2 minutes and keep the home quotas free.
- A shared site venv kept the package from the commit that built it (Lyon jobs 2070636 and 2070637, `No such command 'plan'`). Thus the code now runs from PYTHONPATH with an absolute path (#34, #48).
- A Dockerfile with the same lock serves CI and local reproduction. Apptainer is the fallback if node-local builds become too slow.
