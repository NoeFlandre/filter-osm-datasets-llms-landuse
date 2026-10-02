# Land-use relevance labels for OSM polygon datasets

The project labels every sentence of three OpenStreetMap polygon datasets.
The label is `yes` or `no` for land-use or land-cover relevance.
The model is **LiquidAI LFM2.5-2.6B + DSpark** (thinking mode, greedy, SGLang).
The model runs in many short jobs on Grid'5000. You can resume each job.

| Input | Output |
|---|---|
| `NoeFlandre/osm-polygon-description-tag` | `NoeFlandre/osm-polygon-description-tag-landuse` |
| `NoeFlandre/osm-polygon-wikidata-and-wikipedia` | `NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse` |
| `NoeFlandre/osm-polygon-website-tag` | `NoeFlandre/osm-polygon-website-tag-landuse` |

The project removes no data. Each output copies its input and adds the `labels/` and
`generations/` tables ([schema](schema.md)). A pre-registered non-inferiority gate links the quality
to the published [benchmark](benchmark.md). For the project terms, see the [glossary](glossary.md).
