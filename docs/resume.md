# Resume the land-use labelling from any machine

State at the stop (2026-10-01 approximately 15:20 Paris): the project synced everything remotely.

* **Code:** github.com/NoeFlandre/filter-osm-datasets-llms-landuse, branch `main` (all merged work). Two feature PRs are open and not merged: #159 (publish jobs stop at the walltime without error) and #161 (`luf g5k publish-loop`). Their mutation gate was still red.
* **Work data:** the HF bucket `NoeFlandre/landuse-filter-work` (now PUBLIC). It holds chunk inputs, result parts, planner indexes, gates, plans and ledgers.
* **Controller state:** `controller-state/latest.tar.gz` in the bucket (assignments, completion logs, plans, profiles, gates, back-offs, progress indexes, watchdog script).
* **Datasets on the Hub:** `NoeFlandre/osm-polygon-{description-tag,website-tag,wikidata-and-wikipedia}-landuse`.
* **Remote jobs:** the project did NOT touch the GPU jobs that run on Grid'5000. They continue to upload parts to the bucket until their walltime ends.

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

Before you start `keep_running.sh`, edit the paths in the script.
The script starts these processes again:

- The production controller. The datasets are in priority order: description, wiki, website. The walltimes are 60 minutes (day) and 120 minutes (night), with a 30-minute fallback. The options are `--policy-check per-batch --submit-workers 8 --chunk-overflow 1.6`.
- The admission controller.
- The H200 window-96 gate run.

The controller reconciles with OAR. Thus it adopts the jobs that still run.

## What is left to do
1. Update the Hub: the website card and the wiki (input mirror at 4,001 files, then partial labels, viewer, map, card). Run `luf g5k cpu-job publish --site <site> --dataset <d> --revision <rev> --walltime-minutes 60` (revisions: website 6503db8c4d15f79503cf0b191f7b058693614842, wiki d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801). Submit it again until it is done. PR #159 and #161 make this more robust.
2. H200 window 96: when the `w96h200` benchmark run is complete, run `luf bench compare --namespace w96h200 --label w96h200`. If it passes, set `max_running_requests` to 96 in `profiles/h200_nvl.json`.
3. Decide how much of wiki to label. A full run needs weeks. The plan is geographic (equal share for each H3 cell). Thus any prefix is a uniform sample.
4. Open issues: #1 epic, #17, #18, #26, #28, #29, #144 (resumable replan job).
