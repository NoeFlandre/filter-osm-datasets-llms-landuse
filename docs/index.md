# Land-use relevance labels for OSM polygon datasets

Every sentence of three OpenStreetMap polygon datasets is labelled `yes` / `no` for
land-use / land-cover relevance by **LiquidAI LFM2.5-2.6B + DSpark** (thinking mode,
greedy, SGLang) running as many short, resumable jobs across Grid'5000.

| Input | Output |
|---|---|
| `NoeFlandre/osm-polygon-description-tag` | `NoeFlandre/osm-polygon-description-tag-landuse` |
| `NoeFlandre/osm-polygon-wikidata-and-wikipedia` | `NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse` |
| `NoeFlandre/osm-polygon-website-tag` | `NoeFlandre/osm-polygon-website-tag-landuse` |

Nothing is removed: each output mirrors its input and adds `labels/` and
`generations/` tables ([schema](schema.md)). Quality is tied to the published
[benchmark](benchmark.md) by a pre-registered non-inferiority gate.
