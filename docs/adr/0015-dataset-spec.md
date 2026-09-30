# ADR-0015: One DatasetSpec per input dataset; the wiki card gets a map

Status: accepted (2026-09-30)

## Context

Per-dataset behaviour was spread over four tables and branches: join keys
(`assemble.JOIN_KEYS`), the card's join recipe and licence text (`card.JOINS`,
`card.TEXT_LICENSE`), the planning locator (`locate.py`) and the published-map locator
(`publish._locator`). Adding or changing a dataset meant editing all of them, and they had
drifted: the wiki input has coordinates (used for the geographic plan, ADR-0014) but its card
had no map.

## Decision

`application/datasets.py` holds `SPECS[dataset]`, a `DatasetSpec` with the `Source` (reader,
repo, shards), the join keys, the card's join condition, key wording, map placement sentence and
licence text, and two optional location capabilities: a planning locator and a map locator
(implemented in `application/locators.py`). It lives in `application`, not next to `SOURCES`,
because the locators use `application.geo` and the layering is domain <- adapters <- application.

The wiki card also draws the yes-share map. A wiki labels row is placed exactly as the planner
places its text: the first polygon (smallest `polygon_id` with coordinates) linked to the
sentence's document (`polygon_document_links/<region>.parquet`, `polygons/<region>.parquet`,
sentences carry `document_id`). Both paths share one function in `geo.py`, so the plan and the
map cannot disagree.

## Consequences

* A new dataset is one `SPECS` entry plus its reader.
* Wiki statistics records gain `cells`/`labelled`/`located`; existing records without `cells`
  are counted again on the next publish (the existing locator behaviour), which reads the wiki
  polygon tables from the Hub once per labels file.
