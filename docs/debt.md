# Known weaknesses

| Weakness | Why it exists | Cleanup path |
|---|---|---|
| Canonical generation per text (dedup) ignores tiny GPU-numerics differences between duplicates | Greedy decoding is deterministic only per batch composition and GPU | Acceptable by design (ADR-0005); the per-GPU gate bounds the effect |
| `pending_chunks` re-reads every incomplete chunk's parts each cycle | Simplest correct implementation | Cache per-chunk part lists keyed by directory mtime once plans exceed ~10k chunks |
| Resolution map (`resolve_all`) is an in-memory dict | Fine up to a few million unique texts | Move to the planner's SQLite index before the website run |
| Controller only submits where GPUs are free *now* | Grid'5000 is busy; queued jobs wait long | Add a bounded queue per site (`max_queued_per_site`) driven by Gantt predictions |
| Day-time job extensions (`oarwalltime`, quota exempt) are not used | Needs the controller to act in a job's last 10 minutes | Add an extension pass to the cycle |
| Local dev on an external drive is slow | Hardware | Keep the venv and work tree on the internal disk (`UV_PROJECT_ENVIRONMENT`, `LUF_WORK`) |
