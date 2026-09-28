# filter-osm-datasets-llms-landuse

Labels **every sentence** of three OpenStreetMap polygon datasets as land-use
relevant (`yes`) or not (`no`) with **LiquidAI LFM2.5-2.6B + its DSpark draft**
(SGLang, BF16, greedy, thinking mode). It runs as short, resumable jobs across
Grid'5000 and publishes the results on the Hugging Face Hub.

| Input | Output |
|---|---|
| [`NoeFlandre/osm-polygon-description-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag) | [`…-description-tag-landuse`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag-landuse) |
| [`NoeFlandre/osm-polygon-wikidata-and-wikipedia`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia) | [`…-wikidata-and-wikipedia-landuse`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse) |
| [`NoeFlandre/osm-polygon-website-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag) | [`…-website-tag-landuse`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag-landuse) |

Nothing is dropped. Each output mirrors its input and adds `labels/` (one row per
sentence position: `yes`, `no`, `failed` or `skipped_unsplit`) and `generations/`
(one row per unique sentence: the raw model output, token counts and provenance).
Quality is tied to the [benchmark](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance)
by a pre-registered non-inferiority gate, and each GPU type is admitted only after
passing it. Plan and progress: [epic #1](https://github.com/NoeFlandre/filter-osm-datasets-llms-landuse/issues/1).

## Prerequisites

- A Grid'5000 account, with SSH aliases for the sites (`ProxyJump access.grid5000.fr`).
- A Hugging Face login on the controller machine (`hf auth login`).
- On every site, a fine-grained HF token at `~/luf/hf_token` (mode 600), placed by the
  owner, with write access to the three output repos and the private bucket
  `NoeFlandre/landuse-filter-work`.
- [uv](https://docs.astral.sh/uv/).

## Quickstart (controller machine)

```bash
export UV_PROJECT_ENVIRONMENT=~/.venvs/luf   # keep the env off slow or external disks
export LUF_WORK=~/luf-work                     # controller ledger (small)
uv sync --extra tokenize
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

Every step can be interrupted and re-run: work is content-addressed, and completed
results are never redone.

## Development

```bash
make install    # env + pre-commit hook (ruff format/check on staged files)
make gauntlet   # ruff, ty, unit/property/Gherkin/architecture tests, CRAP, mutation, docs
```

Docs: https://noeflandre.github.io/filter-osm-datasets-llms-landuse/ (MkDocs sources in `docs/`): architecture, output schema, benchmark parity, sizing,
Grid'5000 operations, ADRs, known weaknesses.
