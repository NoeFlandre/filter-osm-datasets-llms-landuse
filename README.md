# filter-osm-datasets-llms-landuse

This project labels **every sentence** of three OpenStreetMap polygon datasets. Each label is
land-use relevant (`yes`) or not (`no`). The model is **LiquidAI LFM2.5-2.6B + its DSpark draft**
(SGLang, BF16, greedy, thinking mode). The project runs short jobs that you can resume.
The jobs run on Grid'5000. The project publishes the results on the Hugging Face Hub.

| Input | Output |
|---|---|
| [`NoeFlandre/osm-polygon-description-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag) | [`…-description-tag-landuse`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag-landuse) |
| [`NoeFlandre/osm-polygon-wikidata-and-wikipedia`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia) | [`…-wikidata-and-wikipedia-landuse`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse) |
| [`NoeFlandre/osm-polygon-website-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag) | [`…-website-tag-landuse`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag-landuse) |

The project removes no data. Each output copies its input and adds two tables:

- `labels/` has one row for each sentence position. The decision is `yes`, `no`, `failed` or `skipped_unsplit`.
- `generations/` has one row for each unique sentence. It has the raw model output, the token counts and the provenance.

A pre-registered non-inferiority gate links the quality to the
[benchmark](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance).
The project admits a GPU type only after the GPU type passes the gate.
For the plan and the progress, see [epic #1](https://github.com/NoeFlandre/filter-osm-datasets-llms-landuse/issues/1).

## Prerequisites

- A Grid'5000 account, with SSH aliases for the sites (`ProxyJump access.grid5000.fr`).
- A Hugging Face login on the controller machine (`hf auth login`).
- On each site, a fine-grained HF token at `~/luf/hf_token` (mode 600). The owner puts the token there.
  The token has write access to the three output repositories and to the private bucket
  `NoeFlandre/landuse-filter-work`.
- [uv](https://docs.astral.sh/uv/).

## Quickstart (controller machine)

```bash
export UV_PROJECT_ENVIRONMENT=~/.venvs/luf   # keep the env off slow or external disks
export LUF_WORK=~/luf-work                     # controller ledger (small)
uv sync --extra tokenize --extra map
luf g5k inventory                              # eligible GPU clusters on all sites
```

## The pipeline, in order

```bash
# 1. Benchmark as work, then admit a GPU type (full benchmark through the gate)
luf bench plan
luf g5k run --datasets benchmark --namespace gpu-a100_sxm4_40gb --gpu-models a100_sxm4_40gb
luf bench admit --gpu a100_sxm4_40gb

# 2. Plan a dataset on a Grid'5000 CPU node (resumable; chunks go to the bucket)
luf g5k cpu-job plan --site grenoble --dataset osm-polygon-description-tag --revision <sha>

# 3. Generate on admitted GPU types only
luf g5k run --datasets osm-polygon-description-tag --bucket NoeFlandre/landuse-filter-work

# 4. Publish incrementally (mirror input, labels/, generations/, card)
luf g5k cpu-job publish --site grenoble --dataset osm-polygon-description-tag --revision <sha>

luf status                       # progress
luf g5k pause [--cancel]         # stop submitting (optionally cancel our jobs)
```

You can stop each step and run it again. The work is content-addressed.
The project does not repeat completed results.

## Development

```bash
make install    # env + pre-commit hook (ruff format/check on staged files)
make gauntlet   # ruff, ty, unit/property/Gherkin/architecture tests, CRAP, mutation, docs
```

Documentation: https://noeflandre.github.io/filter-osm-datasets-llms-landuse/ (the MkDocs sources are in `docs/`).
It has the architecture, the output schema, the benchmark parity, the sizing, the
Grid'5000 operations, the ADRs, the known weaknesses and the [glossary](docs/glossary.md).
