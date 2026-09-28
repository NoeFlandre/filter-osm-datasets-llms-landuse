"""Dataset card for a ``-landuse`` output repo (pure rendering; the publisher uploads it)."""

from collections.abc import Mapping
from dataclasses import dataclass

JOINS = {
    "osm-polygon-description-tag": (
        "language-v1/data/*.parquet",
        "USING (description_identity, tag_key)",
        "description_identity, tag_key, sentence_index",
    ),
    "osm-polygon-wikidata-and-wikipedia": (
        "wikipedia/sentences/*.parquet",
        "USING (sentence_id)",
        "sentence_id",
    ),
    "osm-polygon-website-tag": (
        "polygons/*.parquet",
        "USING (polygon_id)",
        "polygon_id, field (website | contact_website), sentence_index",
    ),
}
TEXT_LICENSE = {
    "osm-polygon-wikidata-and-wikipedia": (
        "Wikipedia and Wikivoyage text is CC BY-SA 4.0; attribution columns are kept "
        "in the mirrored input."
    ),
}
FAILURES = {
    "truncated": "hit max_new_tokens before answering",
    "unclosed_think": "never closed its reasoning block",
    "empty": "nothing after the reasoning",
    "ambiguous": "answer mentions both labels",
    "non_english_token": "answered in another language (e.g. oui/ja), not mapped",
    "no_label": "no yes/no in the answer",
    "unsplit_upstream": "text not segmented upstream (skipped_unsplit, no LLM call)",
}


@dataclass(frozen=True, slots=True)
class CardFacts:
    dataset: str
    revision: str
    model: str
    draft: str
    max_new_tokens: int
    fingerprint: str
    labelled_files: int
    total_files: int
    decisions: Mapping[str, int]
    admitted: Mapping[str, Mapping[str, float]]  # gpu key -> gate numbers


def _gate_row(gpu: str, g: Mapping[str, float]) -> str:
    mcc = f"{g['delta_mcc']:+.4f} ({g['mcc_lower']:+.4f})"
    failed = f"{g['delta_failed_rate'] * 100:+.2f} pp"
    return f"| `{gpu}` | {g['delta_f1']:+.4f} | {mcc} | {g['delta_accuracy']:+.4f} | {failed} |"


def render_card(f: CardFacts) -> str:
    status = "complete" if f.labelled_files == f.total_files else "in_progress"
    source, using, keys = JOINS[f.dataset]
    decisions = (
        "\n".join(f"| `{k}` | {v:,} |" for k, v in sorted(f.decisions.items())) or "| — | 0 |"
    )
    failures = "\n".join(f"| `{k}` | {v} |" for k, v in FAILURES.items())
    rows = [_gate_row(gpu, g) for gpu, g in sorted(f.admitted.items())]
    gates = "\n".join(rows) or "| — | | | | |"
    extra_license = f"\n{TEXT_LICENSE[f.dataset]}" if f.dataset in TEXT_LICENSE else ""
    return f"""---
license: odbl
pretty_name: {f.dataset} (land-use labels)
tags: [openstreetmap, land-use, land-cover, remote-sensing, geospatial]
dataset_status: {status}
configs:
- config_name: labels
  data_files: "labels/**/*.parquet"
- config_name: generations
  data_files: "generations/**/*.parquet"
---
# {f.dataset}-landuse

Every in-scope sentence of [`NoeFlandre/{f.dataset}`](https://huggingface.co/datasets/NoeFlandre/{f.dataset})
(revision `{f.revision}`, mirrored here unchanged) labelled for land-use / land-cover
relevance by **{f.model}** with the DSpark draft **{f.draft}** (SGLang, BF16, greedy,
thinking mode, `max_new_tokens={f.max_new_tokens}`).

**Status: {status}**: {f.labelled_files:,} / {f.total_files:,} input files labelled.

| Decision | Sentence positions |
|---|---:|
{decisions}

## Files

* `labels/<input path>.parquet`: one row per sentence position. Columns: `label_id` and the
  join keys ({keys}), `text_sha256`, `decision` (`yes`, `no`, `failed`, `skipped_unsplit`),
  `parse_mode`, `failure_reason` and `generation_id`.
* `generations/<fingerprint>/*.parquet`: one row per **unique** sentence text, with the
  raw output (including reasoning), token counts, finish reason, DSpark acceptance,
  GPU, site, job and code commit. Duplicate sentences share one generation.

| failure_reason | meaning |
|---|---|
{failures}

## Join recipe (DuckDB)

```sql
SELECT i.*, l.decision, g.raw_output
FROM '{source}' i
JOIN 'labels/{source}' l {using}
LEFT JOIN 'generations/*/*.parquet' g USING (generation_id);
```

## Quality

Prompt and serving configuration are those of the benchmark
[`NoeFlandre/benchmark-llms-landuse-relevance`](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance).
Only GPU types that passed a pre-registered non-inferiority gate on the full
25,500-item benchmark were used. The table shows the Δ macro scores against the
published reference; margins are -0.01, failed rate +0.5 pp.

| GPU type | ΔF1 | ΔMCC (95 % low) | Δaccuracy | Δfailed |
|---|---:|---:|---:|---:|
{gates}

Caveats: greedy decoding is not batch- or GPU-invariant, so about 12 % of individual
decisions would flip between two runs of the reference setup; aggregate quality is
what the gate guarantees.

## License and citation

Labels: ODbL, like the OpenStreetMap-derived input.{extra_license}
Code: https://github.com/NoeFlandre/filter-osm-datasets-llms-landuse (config
fingerprint `{f.fingerprint}`).
"""
