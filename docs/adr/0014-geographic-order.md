# ADR-0014: Plan the work in a geographically uniform order

Status: accepted (2026-09-30, owner decision)

## Context

The project cut chunks from unique texts in input-file order. Thus a run that stopped early covered whole
regions (the files are regional extracts in alphabetical order) and left other regions untouched. The owner can
decide not to label everything. Then the owner wants a geographically uniform sample.

## Decision

`luf node replan` (a CPU job, `luf g5k cpu-job replan`) plans again each text that has no generation:

* **Location.** Website: the `lat` and `lon` of the polygon (input file). Wiki: the first polygon
  (smallest `polygon_id` with coordinates) linked to the document of the sentence through
  `polygon_document_links/<region>.parquet`. The coordinates come from `polygons/<region>.parquet`.
  The project puts each location in an H3 cell (resolution 3, as in the card map). A text that
  occurs first in several places takes its first location.
* **Order.** Round `j` holds the `j`-th text (by hash) of each cell. Cells with the same rank use a
  hash order. Thus any prefix of the plan takes an equal share from each cell that still
  has texts. A text without a location gets a round from its hash. This spreads these texts evenly.
* **Kept work.** The project never plans again the texts that have a generation (from the manifests of the bucket).
  Finished chunks keep their plan line and place. Open chunks leave the plan (their chunk files
  stay in the bucket, not used) and the project cuts their unfinished texts into new chunks. Chunk ids stay a
  function of their texts.
* The line order of the plan file is the assignment order. Thus the controllers need no change, other than to read
  the new plan.

## Consequences

* Dense cells finish last. Sparse regions are covered early ("equal share for each cell", not a
  proportional sample).
* Jobs that still run old chunks can generate a few texts two times (no harm: the result is canonical).
* Planning tokenizes the remaining texts again (hours of CPU on one node).
