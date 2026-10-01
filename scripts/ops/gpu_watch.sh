#!/usr/bin/env bash
# Compact periodic summary: running/waiting GPUs, per-dataset chunks, admission pending, last error.
# Needs LUF_WORK (the controllers' work tree) and runs from the project's .venv.
# GPU_WATCH_ONCE=1 prints one summary line and exits (used by tests/unit/test_gpu_watch.py).
: "${LUF_WORK:?set LUF_WORK to the controllers work tree}"
W=$LUF_WORK
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="$PROJECT/.venv/bin:$PATH" LUF_WORK=$W
while true; do
  gpus=$(python3 - <<'PY'
import json, glob, os, subprocess, collections
A = {}
for f in glob.glob(os.environ['LUF_WORK'] + '/assignments/*.json'):
    try:
        a = json.load(open(f))
    except Exception:
        continue
    if a.get('job_id'):
        A[(a['site'], str(a['job_id']))] = a
run = collections.Counter(); wait = 0; down = []
for s in "grenoble lille lyon nancy rennes sophia toulouse luxembourg".split():
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", s, "oarstat -u | grep ' luf-'"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 255:  # ssh itself failed (grep finding no job exits 1, not 255)
            raise OSError("ssh failed")
        out = r.stdout
    except Exception:
        down.append(s)  # an unreachable site must not look like an empty one
        continue
    for l in out.splitlines():
        p = l.split()
        if len(p) < 6:
            continue
        a = A.get((s, p[0]))
        if p[5] == 'R':
            kind = 'cpu' if not a else ('adm' if 'gpu-' in a.get('fp', '') else 'prod')
            run[kind + ':' + (a['gpu'] if a else '-')] += 1
        else:
            wait += 1
print(f"running={sum(v for k, v in run.items() if not k.startswith('cpu'))} waiting={wait} " + ' '.join(f"{k}={v}" for k, v in sorted(run.items())) + (f" UNREACHABLE={','.join(down)}" if down else ""))
PY
)
  ds=$(luf status --datasets osm-polygon-description-tag,osm-polygon-website-tag,osm-polygon-wikidata-and-wikipedia --work $W 2>&1 | python3 -c "
import sys, ast
out = []
for line in sys.stdin:
    if line.startswith('osm'):
        name, rest = line.split(':', 1)
        d = ast.literal_eval(rest.strip())
        out.append(f\"{name.split('-')[2][:4]}={d['chunks_complete']}/{d['chunks']}\")
print(' '.join(out))")
  adm=$(tail -n 400 $W/controller-admission.log 2>/dev/null | python3 -c "
import sys, json
last = {}
for line in sys.stdin:
    if line.startswith('{'):
        try: r = json.loads(line); last[r['namespace'].removeprefix('gpu-')] = r['pending_chunks']
        except Exception: pass
print(' '.join(f'{k}={v}' for k, v in sorted(last.items()) if v))")
  err=$(tail -n 60 $W/controller-production.log $W/controller-admission.log 2>/dev/null | grep -hE "cycle failed|submission failed" | grep -v "besteffort. Reserve" | tail -1 | cut -c1-140)
  # A controller that submits but no longer ingests results (seen 2026-10-01: wedged for an hour)
  # shows up as a growing age of the progress index.
  last=$(python3 -c "import glob, os, sys; f = glob.glob(sys.argv[1] + '/index/progress-*.sqlite'); print(int(max(map(os.path.getmtime, f))) if f else 0)" "$W")
  age=$(( ( $(date +%s) - ${last:-0} ) / 60 ))
  echo "$(date +%H:%M) GPUS $gpus | chunks $ds | ingest ${age}m ago | admission pending: ${adm:-none} ${err:+| error: $err}"
  [ -n "${GPU_WATCH_ONCE:-}" ] && exit 0
  sleep 600
done
