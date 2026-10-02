# ADR-0009 — Work store: private HF Bucket as the transport, laptop holds only a ledger

**Status:** accepted · 2026-09-28 (revised: bucket mode replaces site spools)

**Context.** The laptop SSD is scarce. The site homes have quotas and no backup. The generations for the wiki and website datasets reach tens of GB.
**Decision.** The private bucket `NoeFlandre/landuse-filter-work` holds these items:

- chunk inputs
- result parts (named by the sha256 of their bytes) and a small manifest for each part (its text hashes)
- planner index checkpoints
- plan lines
- job summaries
- gates and profiles

The nodes download the inputs to node-local scratch. They upload each part and manifest when they flush. They use a fine-grained token that the owner put at `~/luf/hf_token` (mode 600). The controller pulls only the manifests and summaries. Its local tree holds the ledger and small metadata. Planning and publishing run as CPU jobs on node scratch.
**Consequences.** No bulky data stays on the laptop or in the site homes. Re-hashing finds corruption. Duplicates do no harm. The site spools (`~/luf/work`) carry only assignment files and job summaries. The project removed the older spool transport (rsync of chunks and parts through site homes) in #43. The gates download the parts of a namespace to a temporary directory and delete it afterward.
