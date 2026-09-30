# Known weaknesses

| Weakness | Why it exists | Cleanup path |
|---|---|---|
| Canonical generation per text (dedup) ignores tiny GPU-numerics differences between duplicates | Greedy decoding is deterministic only per batch composition and GPU | Acceptable by design (ADR-0005); the per-GPU gate bounds the effect |
| Day-time job extensions (`oarwalltime`, quota exempt) are not used | Each job's work is sized to its walltime (plus 20 % overflow), so an extended job would have nothing left to do | Only worth it if nodes learn to fetch more chunks mid-job; then extend in the job's last 10 minutes while GPUs stay free |
| Local dev on an external drive is slow | Hardware | Keep the venv and work tree on the internal disk (`UV_PROJECT_ENVIRONMENT`, `LUF_WORK`) |
| Mutation testing and CRAP cover `domain/` only | `application/` mixes I/O and logic; mutmut runs only on the pure layer | Move decisions into pure functions (as `slot_for`), then bring the pure application modules under mutmut and CRAP (issue #117) |
| `cli/*`, `node_main.py`, `adapters/engine.py`, the tokenizer and `adapters/frontend/` are omitted from coverage | They need a GPU, the real tokenizer, a node or Grid'5000 | Typer `CliRunner` tests with stubbed use cases for `cli/*` (issue #116) |
