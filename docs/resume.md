# Resume the land-use labelling from any machine

State at the stop (2026-10-01 ~15:20 Paris): everything is synced remotely.

* **Code:** github.com/NoeFlandre/filter-osm-datasets-llms-landuse, branch `main` (all merged work). Two feature PRs are open and unmerged: #159 (publish jobs stop gracefully at the walltime) and #161 (`luf g5k publish-loop`); their mutation gate was still red.
* **Work data:** the HF bucket `NoeFlandre/landuse-filter-work` (now PUBLIC): chunk inputs, result parts, planner indexes, gates, plans, ledgers.
* **Controller state:** `controller-state/latest.tar.gz` in the bucket (assignments, completion logs, plans, profiles, gates, back-offs, progress indexes, watchdog script).
* **Datasets on the Hub:** `NoeFlandre/osm-polygon-{description-tag,website-tag,wikidata-and-wikipedia}-landuse`.
* **Remote jobs:** running GPU jobs on Grid'5000 were NOT touched; they keep uploading parts to the bucket until their walltime.

## Restore and restart

```bash
git clone https://github.com/NoeFlandre/filter-osm-datasets-llms-landuse && cd filter-osm-datasets-llms-landuse
uv sync --extra tokenize --extra map          # CPU environment
export LUF_WORK=$HOME/luf-hot && mkdir -p $LUF_WORK
python - <<'PY'                               # needs a Hugging Face login with write access
import os, tarfile, tempfile, pathlib
from huggingface_hub import HfApi
api = HfApi(); tmp = pathlib.Path(tempfile.mkdtemp())
api.download_bucket_files("NoeFlandre/landuse-filter-work", [("controller-state/latest.tar.gz", str(tmp / "s.tgz"))])
with tarfile.open(tmp / "s.tgz") as t:
    t.extractall(tmp)
os.system(f"cp -R {tmp}/luf-hot/. {os.environ['LUF_WORK']}/")
PY
# Grid'5000 access: ssh config for the sites, and ~/luf/hf_token on each site's home (owner-placed).
nohup $LUF_WORK/keep_running.sh >> $LUF_WORK/keep_running.log 2>&1 &   # edit the paths inside first
```

`keep_running.sh` relaunches the production controller (datasets in priority order: description, wiki, website; day 60 / night 120 min walltimes with 30 min fallback; `--policy-check per-batch --submit-workers 8 --chunk-overflow 1.6`), the admission controller and the H200 window-96 gate run. The controller reconciles with OAR, so jobs still running are adopted.

## What is left to do
1. Hub updates: website card and wiki (input mirror at 4,001 files, then partial labels, viewer, map, card) via `luf g5k cpu-job publish --site <site> --dataset <d> --revision <rev> --walltime-minutes 60` (revisions: website 6503db8c4d15f79503cf0b191f7b058693614842, wiki d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801); resubmit until done. PR #159/#161 make this robust.
2. H200 window 96: when the `w96h200` benchmark run completes, `luf bench compare --namespace w96h200 --label w96h200`; if it passes set `profiles/h200_nvl.json` `max_running_requests` to 96.
3. Decide how much of wiki to label (full run is weeks); the plan is geographic (equal share per H3 cell), so any prefix is a uniform sample.
4. Issues open: #1 epic, #17, #18, #26, #28, #29, #144 (resumable replan job).
