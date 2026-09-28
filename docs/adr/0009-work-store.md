# ADR-0009 — Work store: private HF Bucket as the transport, laptop holds only a ledger

**Status:** accepted · 2026-09-28 (revised: bucket mode replaces site spools)

**Context.** The laptop SSD is scarce and site homes are quota-limited and not backed up; generations for the wiki and website datasets reach tens of GB.
**Decision.** The private bucket `NoeFlandre/landuse-filter-work` holds chunk inputs, result parts (named by the sha256 of their bytes) plus a small manifest per part (its text hashes), planner index checkpoints, plan lines, job summaries, gates and profiles. Nodes download inputs to node-local scratch and upload each part and manifest as they flush, using a fine-grained token the owner placed at `~/luf/hf_token` (mode 600). The controller pulls only manifests and summaries; its local tree holds the ledger and small metadata. Planning and publishing run as CPU jobs on node scratch.
**Consequences.** Nothing bulky lives on the laptop or in site homes; corruption is detected by re-hashing; duplicates are harmless. Site spools (`~/luf/work`) remain only for assignments, and for the legacy spool mode still used by the admission runs; removal is tracked in #43.
