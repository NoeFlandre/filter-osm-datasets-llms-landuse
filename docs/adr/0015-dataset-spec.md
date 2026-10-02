# ADR-0015: One DatasetSpec per input dataset; the wiki card gets a map

Status: accepted (2026-09-30)

## Context

The behavior for each dataset was in four tables and branches: the join keys
(`assemble.JOIN_KEYS`), the join recipe and the licence text of the card (`card.JOINS`,
`card.TEXT_LICENSE`), the planning locator (`locate.py`) and the locator of the published map
(`publish._locator`). To add or change a dataset, you edited all of them. They drifted apart:
the wiki input has coordinates (used for the geographic plan, ADR-0014) but its card
had no map.

## Decision

`application/datasets.py` holds `SPECS[dataset]`. It is a `DatasetSpec` with these items:

- The `Source` (reader, repo, shards).
- The join keys.
- The join condition of the card, the key wording, the map placement sentence and the licence text.
- Two optional location capabilities: a planning locator and a map locator (implemented in `application/locators.py`).

It lives in `application`, not next to `SOURCES`. The locators use `application.geo` and the layering is
domain <- adapters <- application.

The wiki card also draws the yes-share map. The project places a wiki labels row exactly as the planner
places its text: the first polygon (smallest `polygon_id` with coordinates) linked to the
document of the sentence (`polygon_document_links/<region>.parquet`, `polygons/<region>.parquet`,
the sentences have `document_id`). Both paths use one function in `geo.py`. Thus the plan and the
map cannot disagree.

## Consequences

* A new dataset needs one `SPECS` entry and its reader.
* Wiki statistics records get `cells`, `labelled` and `located`. The next publish counts again the existing
  records without `cells` (the existing locator behavior). This reads the wiki
  polygon tables from the Hub one time for each labels file.
