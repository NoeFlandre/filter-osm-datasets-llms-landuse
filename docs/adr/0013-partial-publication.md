# ADR-0013: Publish partly labelled files

Status: accepted (2026-09-30, owner decision)

## Context

Chunks are cut in input-file order and texts repeat across files (each is generated once),
so generation spreads thinly over many files. A dataset 20 % through its chunks has almost no file with every sentence labelled,
and the publisher only uploaded complete files: nothing could be published for the website and
Wikipedia datasets until late in each run.

## Decision

Publish partial files. Sentences without a generation get `decision = pending`; a partial file is
published as soon as one of its sentences is resolved (so a slowly progressing dataset such as the 761-file wiki shows every file at once, with the geographically uniform order of ADR-0014) and then re-uploaded when it gained 1 % of its sentences (`PARTIAL_STEP`; 10 % at first, lowered once the
geographic order of ADR-0014 spread progress thinly over every file) and replaced when complete.
The dataset viewer table holds only answered or skipped sentences, never `pending` ones.
Generations are still uploaded once, with the complete file that owns the text. Partial files live
in their own ledger; the card counts pending rows and partial files.

## Consequences

* Labels exist for every dataset early and improve over time; consumers must treat `pending` as
  "no answer yet", not as a label.
* The "every sentence has a decision" guarantee holds only when `dataset_status: complete` and no
  `pending` row remains; the final verification checks exactly that.
* Hub commit history grows with each refresh (bounded by the 1 % step per file).
* Joins from partial labels to `generations/` may miss rows until the owning file completes.
