# ADR-0014: Plan the work in a geographically uniform order

Status: accepted (2026-09-30, owner decision)

## Context

Chunks were cut from unique texts in input-file order, so a run stopped early covered whole
regions (files are regional extracts, alphabetical) and left others untouched. The owner may
decide not to label everything and then wants a geographically uniform sample.

## Decision

`luf node replan` (a CPU job, `luf g5k cpu-job replan`) plans every not-yet-generated text again:

* **Location.** Website: the polygon's `lat`/`lon` (input file). Wiki: the first polygon
  (smallest `polygon_id` with coordinates) linked to the sentence's document through
  `polygon_document_links/<region>.parquet`, coordinates from `polygons/<region>.parquet`. Each
  is binned into an H3 cell (resolution 3, as in the card map). A text first seen in several
  places takes its first location.
* **Order.** Round `j` holds the `j`-th text (by hash) of every cell; cells are tied in a
  hash order. Any prefix of the plan therefore takes an equal share from each cell that still
  has texts. Texts without a location get a hash-derived round, spreading them evenly.
* **Kept work.** Texts with a generation (from the bucket's manifests) are never planned again;
  finished chunks keep their plan line and place; open chunks leave the plan (their chunk files
  stay in the bucket, unused) and their unfinished texts are re-chunked. Chunk ids stay a
  function of their texts.
* The plan file's line order is the assignment order, so the controllers need no change beyond
  reading the new plan.

## Consequences

* Dense cells finish last; sparse regions are covered early ("equal share per cell", not a
  proportional sample).
* Jobs still running old chunks may generate a few texts twice (harmless: canonical result).
* Planning tokenises the remaining texts again (hours of CPU on one node).
