# filter-osm-datasets-llms-landuse

Labels every sentence of three OSM polygon datasets for land-use relevance with
LiquidAI LFM2.5-2.6B + DSpark (thinking mode, SGLang), as short resumable jobs across
Grid'5000, and publishes `<input>-landuse` datasets on the Hugging Face Hub.

* Plan & tracking: [epic #1](https://github.com/NoeFlandre/filter-osm-datasets-llms-landuse/issues/1)
* Docs: `docs/` (MkDocs) — schema, benchmark parity, Grid'5000 operations, ADRs.

```bash
uv sync --extra tokenize
make gauntlet
luf --help
```
