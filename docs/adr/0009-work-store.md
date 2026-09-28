# ADR-0009 — Work store: local tree + site spools + private HF Bucket

**Status:** accepted · 2026-09-28

**Decision.** The authoritative work tree is local to the controller; each site has a spool (`~/luf/work`) used only for transport; the tree is mirrored to a private Hugging Face Bucket for durability. Parts are named by the sha256 of their bytes, so corruption is detected by re-hashing and duplicates are harmless.
