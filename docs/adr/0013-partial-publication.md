# ADR-0013: Publish partly labelled files

Status: accepted (2026-09-30, owner decision)

## Context

The planner orders chunks globally by text frequency, so generation spreads over every input
file at once. A dataset 20 % through its chunks has almost no file with every sentence labelled,
and the publisher only uploaded complete files: nothing could be published for the website and
Wikipedia datasets until late in each run.

## Decision

Publish partial files. Sentences without a generation get `decision = pending`; a partial file is
re-uploaded when it gained 10 % of its sentences (`PARTIAL_STEP`) and replaced when complete.
Generations are still uploaded once, with the complete file that owns the text. Partial files live
in their own ledger; the card counts pending rows and partial files.

## Consequences

* Labels exist for every dataset early and improve over time; consumers must treat `pending` as
  "no answer yet", not as a label.
* The "every sentence has a decision" guarantee holds only when `dataset_status: complete` and no
  `pending` row remains; the final verification checks exactly that.
* Hub commit history grows with each refresh (bounded by the 10 % step per file).
* Joins from partial labels to `generations/` may miss rows until the owning file completes.
