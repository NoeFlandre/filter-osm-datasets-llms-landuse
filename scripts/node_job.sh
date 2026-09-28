#!/usr/bin/env bash
# Runs on a reserved Grid'5000 node (never on a frontend). Usage:
#   node_job.sh <code-dir> <assignment-id>
# The Python environment is built once per site and lockfile into ~/luf/cache (NFS),
# serialised with flock, then reused read-only by every later job. Models are public,
# so no credential ever reaches Grid'5000; results go to the site spool ~/luf/work.
set -euo pipefail
CODE=$1
ASSIGNMENT=$2
CACHE="$HOME/luf/cache"
export LUF_WORK="$HOME/luf/work"
export HF_HOME="$CACHE/hf"
export HF_HUB_DISABLE_TELEMETRY=1
export FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer"
export UV_CACHE_DIR="/tmp/$USER-uv-${OAR_JOB_ID:-local}"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$CACHE" "$LUF_WORK"
trap 'rm -rf "$UV_CACHE_DIR"' EXIT

cd "$CODE"
lock_sha=$(sha256sum uv.lock | cut -c1-12)
venv="$CACHE/venv-$lock_sha"
t0=$(date +%s)
if [[ ! -f "$venv/.ready" ]]; then
  (
    flock 9
    if [[ ! -f "$venv/.ready" ]]; then
      rm -rf "$venv"
      UV_PROJECT_ENVIRONMENT="$venv" uv sync --frozen --no-dev --extra gpu --python 3.12
      touch "$venv/.ready"
    fi
  ) 9>"$CACHE/venv-$lock_sha.lock"
fi
echo "luf: env ready in $(( $(date +%s) - t0 ))s ($venv)"
exec "$venv/bin/luf" node run --assignment "$ASSIGNMENT"
