# ADR-0032: Observable publish, plan and card jobs

Status: accepted (2026-10-05)

## Context

A wiki publish job spends 40 to 60 minutes (downloading the 26 GB planner index and the 7.6 GB
resolution snapshot, listing and reading new parts) before its first Hub commit, and its OAR
stdout only said `luf: env ready in Ns`: nothing was printed or flushed per phase. Several jobs
ended (day walltime, a node-disk SQLite error) with no commit and no clue why, and the operator
had to ssh to guess what a job was doing.

## Decision

- `application/job_progress.py`: a small `Progress` helper (injectable clock, wall clock, output and
  marker upload). `event(phase, **counters)` prints one flushed line `luf: [<elapsed>s] <phase>
  k=v ...`; `tick(...)` does the same at most every 60 s for per-item progress; `finish(reason)`
  prints the last line and always uploads the marker.
- Phases: `publish_start`, `restore_index_start/done` (bytes, seconds),
  `snapshot_download_start/done` (bytes, seconds), `parts_listed` (seen, new), `parts_read` (read,
  total, rows/s), `snapshot_upload` (bytes, seconds), `resolution_ready`, `mirror_start`,
  `build_start`, `building_files` (file, total, complete), `fetch_generation_parts`, `hub_commit`
  (kind, files, short commit id, seconds), `final_upload`, `card_start`, `finished`
  (`stop_reason`, labelled, total; `error: ...` when the job raised).
- The state is mirrored to `published/<dataset>.progress.json` (dataset, job id, phase, counters,
  elapsed seconds, update epoch) through the existing retrying bucket put, at most once per 60 s
  (the final line always goes up). It is a sibling of `status.json`, which the publish loop keeps
  reading unchanged. A failing marker upload is printed and never fails the job.
- `luf g5k publish-status --dataset X` prints it with ONE bucket get (no listing).
- `scripts/node_job.sh` exports `PYTHONUNBUFFERED=1` for all modes; `run_forwarding` is untouched.
- `Hub.upload` now returns the last commit id (or `None`); nothing else about what is published
  changes.

## Consequences

The operator reads the live phase from the laptop (`luf g5k publish-status`) or from
`~/luf/logs/<job>.out`. Cost: at most one small put per minute per job and no extra listing.
Publish jobs deploy the current main commit at their next submission.
