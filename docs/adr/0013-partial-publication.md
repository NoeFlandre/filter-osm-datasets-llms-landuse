# ADR-0013: Publish partly labelled files

Status: accepted (2026-09-30, owner decision)

## Context

The project cuts chunks in input-file order. Texts repeat across files (the project generates each text one time).
Thus the generation spreads thinly over many files. A dataset at 20 % of its chunks has almost no file with all sentences labelled.
The publisher uploaded only complete files. Thus the project could publish nothing for the website and
Wikipedia datasets until late in each run.

## Decision

Publish partial files. A sentence without a generation gets `decision = pending`. The rules are:

- Publish a partial file when one of its sentences is resolved. Thus a slow dataset such as the 761-file wiki shows each file at once, with the geographically uniform order of ADR-0014.
- Upload the file again when it gains 1 % of its sentences (`PARTIAL_STEP`). The step was 10 % at first. The project lowered it when the geographic order of ADR-0014 spread the progress thinly over each file.
- Replace the file when it is complete.

The table of the dataset viewer holds only answered or skipped sentences, never `pending` ones.
The project still uploads generations one time, with the complete file that owns the text. Partial files are in
their own ledger. The card counts the pending rows and the partial files.

## Consequences

* Labels exist early for each dataset and improve with time. Consumers must treat `pending` as
  "no answer yet", not as a label.
* The guarantee "each sentence has a decision" is true only when `dataset_status: complete` and no
  `pending` row remains. The final verification checks this.
* The Hub commit history grows with each refresh (the 1 % step for each file limits this).
* Joins from partial labels to `generations/` can miss rows until the owning file is complete.
