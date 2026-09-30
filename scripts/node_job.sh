#!/usr/bin/env bash
# Runs on a reserved Grid'5000 node (never on a frontend). Usage:
#   node_job.sh <code-dir> <assignment-id>                 GPU generation job
#   node_job.sh <code-dir> plan <dataset> <revision>       CPU planning job
#   node_job.sh <code-dir> replan <dataset> <revision>     CPU job: plan the rest in a geographic order
#   node_job.sh <code-dir> calibrate <chunk-id>             GPU concurrency sweep
# Each job builds a private Python environment on node-local /tmp (never on NFS). Bulk data lives on
# node-local scratch and in the private HF Bucket; the NFS spool only carries small files.
set -euo pipefail
CODE=$1
shift
CODE="$(cd -- "$CODE" && pwd)"
if [[ "${1:-}" == "plan" || "${1:-}" == "publish" || "${1:-}" == "calibrate" || "${1:-}" == "repair" || "${1:-}" == "replan" ]]; then
  MODE=$1
  shift
else
  MODE=run
  ASSIGNMENT=$1
fi
CACHE="$HOME/luf/cache"
export LUF_WORK="$HOME/luf/work"
export HF_HOME="$CACHE/hf"
export HF_HUB_DISABLE_TELEMETRY=1
export FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer"
export UV_CACHE_DIR="/tmp/$USER-uv-${OAR_JOB_ID:-local}"
# Node-local scratch for chunk inputs and parts (bucket mode); removed on exit.
export LUF_SCRATCH="/tmp/$USER-luf-${OAR_JOB_ID:-local}"
# Fine-grained HF token placed by the owner (mode 600). Read into the job's
# environment only; never echoed, never on a command line.
if [[ -r "$HOME/luf/hf_token" ]]; then
  HF_TOKEN="$(<"$HOME/luf/hf_token")"
  export HF_TOKEN
fi
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$CACHE" "$LUF_WORK"
trap 'rm -rf "$UV_CACHE_DIR" "$LUF_SCRATCH"' EXIT

if [[ "$MODE" == "run" || "$MODE" == "calibrate" ]]; then
# SGLang's DeepEP import needs CUDA_HOME to JIT its kernels; OAR starts a non-login
# shell, so load the site's CUDA toolkit explicitly (regression: job 4165500, Rennes).
# shellcheck disable=SC1091
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load "${LUF_CUDA_MODULE:-cuda-toolkit/12.9.1}" 2>/dev/null || module load cuda-toolkit 2>/dev/null || true
if ! command -v nvcc >/dev/null; then
  echo "luf: no CUDA toolkit (nvcc) on this node; SGLang needs CUDA_HOME" >&2
  exit 6
fi
cuda_root="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_HOME="$cuda_root"
export LD_LIBRARY_PATH="$cuda_root/lib64:$cuda_root/lib:${LD_LIBRARY_PATH:-}"
# LFM2.5 declares 131072 context tokens; SGLang's derived default is 128000.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
fi

cd "$CODE"
t0=$(date +%s)
# Always a private env on node-local disk (1-2 min): shared NFS venvs cost ~8 GB of
# home quota per site and can be slow to create (regressions: Grenoble job 3122433
# copying for an hour; sites over quota). Removed on exit.
venv="/tmp/$USER-venv-${OAR_JOB_ID:-local}"
# Only GPU jobs need SGLang/torch/CUDA (~15 GB); CPU jobs (plan, publish) get the light
# tokeniser stack (regression: Lyon planning jobs filled /tmp and died).
case "$MODE" in
  run | calibrate) EXTRAS=(--extra gpu) ;;
  publish | repair | replan) EXTRAS=(--extra tokenize --extra map) ;;  # map: H3 cells (h3, matplotlib)
  *) EXTRAS=(--extra tokenize) ;;
esac
UV_PROJECT_ENVIRONMENT="$venv" UV_LINK_MODE=copy uv sync --frozen --no-dev --no-install-project "${EXTRAS[@]}" --python 3.12
trap 'rm -rf "$UV_CACHE_DIR" "$LUF_SCRATCH" "$venv"' EXIT
# FlashInfer JIT-compiles with the venv's ninja: the venv's bin must be on PATH
# (regression: job 4165509, "No such file or directory: 'ninja'").
export PATH="$venv/bin:$PATH"
# Always run this job's code, never a copy installed in a shared environment: a site
# venv keeps the package from whenever it was built (regression: Lyon job 2070636
# ran stale code without `node plan`).
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
luf() { "$venv/bin/python" -c 'import sys; from landuse_filter.cli import app; sys.exit(app())' "$@"; }
echo "luf: env ready in $(( $(date +%s) - t0 ))s ($venv)"
if [[ "$MODE" == "calibrate" ]]; then
  luf node calibrate --chunk "$1"
  exit $?
fi
if [[ "$MODE" == "repair" ]]; then
  export HF_HOME="$LUF_SCRATCH/hf"
  luf node repair --dataset "$1"
  exit $?
fi
if [[ "$MODE" != "run" ]]; then
  # Input shards and parts are large: keep them on node-local scratch, never on NFS.
  export HF_HOME="$LUF_SCRATCH/hf"
  luf node "$MODE" --dataset "$1" --revision "$2"
  exit $?
fi
luf node run --assignment "$ASSIGNMENT"
