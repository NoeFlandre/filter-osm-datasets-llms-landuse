# ADR-0008 — uv environment per site and lockfile on NFS

**Status:** accepted · 2026-09-28

**Decision.** Docker is unavailable to users on G5K nodes. The first job on a site builds `~/luf/cache/venv-<lock sha>` (flock-serialised) with `uv sync --frozen --extra gpu`; later jobs reuse it. Model weights and FlashInfer JIT caches also live under `~/luf/cache`. A Dockerfile with the same lock serves CI and local reproduction.
**Fallback.** Apptainer image built from the Dockerfile if NFS venvs prove too slow.
