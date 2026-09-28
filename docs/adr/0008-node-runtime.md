# ADR-0008 — Node environments: reuse a finished site env, else build on node-local disk

**Status:** accepted · 2026-09-28 (revised twice, see Consequences)

**Context.** Docker is unavailable to users on G5K nodes, and NFS home performance varies a lot between sites.
**Decision.** `scripts/node_job.sh` reuses `~/luf/cache/venv-<lock sha>` only if it is already finished (`.ready`). Otherwise it builds a private env with `uv sync --frozen --no-install-project --extra gpu` on node-local `/tmp`, which is removed when the job exits. The job always runs **its own deployed code**: `CODE` is resolved to an absolute path and put on `PYTHONPATH`, and the project is never imported from an installed copy. The CUDA toolkit is loaded with `module load` for SGLang/DeepEP, the venv's `bin` is on `PATH` for FlashInfer's ninja, and model weights and FlashInfer JIT caches live under `~/luf/cache/hf` and `~/luf/cache/flashinfer`.
**Consequences.**
- Grenoble spent a whole job copying an 8 GB venv to NFS (job 3122433). Node-local builds take 1–2 minutes and keep home quotas free.
- A shared site venv kept the package from the commit that built it (Lyon jobs 2070636 and 2070637, `No such command 'plan'`), which is why the code now runs from PYTHONPATH with an absolute path (#34, #48).
- A Dockerfile with the same lock serves CI and local reproduction; Apptainer is the fallback if node-local builds become too slow.
