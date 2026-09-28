# Known weaknesses

| Weakness | Why it exists | Cleanup path |
|---|---|---|
| Canonical generation per text (dedup) ignores tiny GPU-numerics differences between duplicates | Greedy decoding is deterministic only per batch composition and GPU | Acceptable by design (ADR-0005); the per-GPU gate bounds the effect |
| Day-time job extensions (`oarwalltime`, quota exempt) are not used | Each job's work is sized to its walltime (plus 20 % overflow), so an extended job would have nothing left to do | Only worth it if nodes learn to fetch more chunks mid-job; then extend in the job's last 10 minutes while GPUs stay free |
| Local dev on an external drive is slow | Hardware | Keep the venv and work tree on the internal disk (`UV_PROJECT_ENVIRONMENT`, `LUF_WORK`) |
