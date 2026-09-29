"""Dataset card for a ``-landuse`` output repo (pure rendering; the publisher uploads it)."""

from collections.abc import Mapping
from dataclasses import dataclass

JOINS = {
    "osm-polygon-description-tag": (
        ("language-v1/data/*.parquet",),
        "USING (description_identity, tag_key)",
        "description_identity, tag_key, sentence_index",
    ),
    "osm-polygon-wikidata-and-wikipedia": (
        ("wikipedia/sentences/*.parquet", "wikivoyage/sentences/*.parquet"),
        "USING (sentence_id)",
        "sentence_id",
    ),
    "osm-polygon-website-tag": (
        ("polygons/*.parquet",),
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
    "truncated": "hit the token limit before answering",
    "unclosed_think": "never closed its reasoning block",
    "empty": "nothing after the reasoning",
    "ambiguous": "answer mentions both labels",
    "non_english_token": "answered with a non-English word that is not mapped",
    "no_label": "no yes/no in the answer",
}
DECISION_ORDER = ("yes", "no", "failed", "skipped_unsplit")
DECISION_MEANING = {
    "yes": "relevant to land use / land cover",
    "no": "not relevant",
    "failed": "the model output could not be parsed into yes/no",
    "skipped_unsplit": "text not segmented upstream; not sent to the model",
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
    decisions: Mapping[str, int]  # counted from the published labels/ tables
    failures: Mapping[str, int]  # failure_reason of the failed rows
    unique_texts: int  # rows of the published generations/ tables
    gpu_rows: Mapping[str, int]  # generations per GPU key
    admitted: Mapping[str, Mapping[str, float]]  # gpu key -> gate numbers


def _from(paths: tuple[str, ...], prefix: str = "") -> str:
    """DuckDB table expression for one or several parquet globs."""
    if len(paths) == 1:
        return f"'{prefix}{paths[0]}'"
    return "read_parquet([" + ", ".join(f"'{prefix}{p}'" for p in paths) + "])"


def _ref(model: str) -> str:
    """`org/name` (revision `abcdef0`) from a ``name@revision`` reference."""
    name, _, revision = model.partition("@")
    return f"`{name}`" + (f" (revision `{revision[:7]}`)" if revision else "")


def _gate_row(gpu: str, generations: int, share: float, g: Mapping[str, float]) -> str:
    mcc = f"{g['delta_mcc']:+.4f} ({g['mcc_lower']:+.4f})"
    failed = f"{g['delta_failed_rate'] * 100:+.2f} pp"
    return (
        f"| `{gpu}` | {generations:,} ({share:.1%}) | {g['delta_f1']:+.4f} | {mcc} | "
        f"{g['delta_accuracy']:+.4f} | {failed} |"
    )


def _decision_table(decisions: Mapping[str, int]) -> str:
    total = sum(decisions.values())
    rows = [
        f"| `{k}` | {decisions.get(k, 0):,} | {decisions.get(k, 0) / total:.1%} | "
        f"{DECISION_MEANING[k]} |"
        for k in DECISION_ORDER
        if total
    ]
    rows.append(f"| **total** | **{total:,}** | | |")
    return "\n".join(rows)


def _failure_table(failures: Mapping[str, int]) -> str:
    failed = sum(failures.values())
    return "\n".join(
        f"| `{k}` | {failures[k]:,} | {failures[k] / failed:.1%} | {FAILURES.get(k, '')} |"
        for k in sorted(failures, key=lambda r: (-failures[r], r))
    )


def render_card(f: CardFacts) -> str:
    status = "complete" if f.labelled_files == f.total_files else "in_progress"
    source, using, keys = JOINS[f.dataset]
    total = sum(f.decisions.values())
    generated = sum(f.gpu_rows.values())
    used = sorted(g for g in f.gpu_rows if g in f.admitted)
    gates = "\n".join(
        _gate_row(g, f.gpu_rows[g], f.gpu_rows[g] / generated, f.admitted[g]) for g in used
    )
    failed_section = (
        f"""
### Failed rows

{sum(f.failures.values()):,} rows ({f.decisions.get("failed", 0) / total:.1%} of all rows) have no
usable answer. Reasons:

| `failure_reason` | Rows | Share | Meaning |
|---|---:|---:|---|
{_failure_table(f.failures)}
"""
        if f.failures
        else ""
    )
    extra_license = f"\n{TEXT_LICENSE[f.dataset]}" if f.dataset in TEXT_LICENSE else ""
    return f"""---
license: odbl
pretty_name: {f.dataset} (land-use labels)
language: [multilingual]
task_categories: [text-classification]
tags: [openstreetmap, land-use, land-cover, geospatial]
dataset_status: {status}
configs:
- config_name: labels
  data_files: "labels/**/*.parquet"
- config_name: generations
  data_files: "generations/**/*.parquet"
---
# {f.dataset}-landuse

A land-use / land-cover relevance label for every sentence of [`NoeFlandre/{f.dataset}`](https://huggingface.co/datasets/NoeFlandre/{f.dataset}) (revision `{f.revision[:7]}`). The input is mirrored here unchanged; labels and model outputs are separate tables that join back to it.

**{status.replace("_", " ").capitalize()}: {f.labelled_files:,} of {f.total_files:,} input files labelled, {total:,} rows, {f.unique_texts:,} unique texts sent to the model.**

| `decision` | Rows | Share | Meaning |
|---|---:|---:|---|
{_decision_table(f.decisions)}
{failed_section}
## Tables

* `labels/<input path>.parquet`: one row per sentence. Join keys ({keys}), `text_sha256`, `decision`, `parse_mode`, `failure_reason`, `generation_id`.
* `generations/<fingerprint>/*.parquet`: one row per **unique** text: raw output including the reasoning, token counts, finish reason, speculative-decoding statistics, GPU, site, job and code commit. Identical texts are generated once and share a `generation_id`.

```sql
SELECT i.*, l.decision, g.raw_output
FROM {_from(source)} i
JOIN {_from(source, "labels/")} l {using}
LEFT JOIN 'generations/*/*.parquet' g USING (generation_id);
```

## Method

{_ref(f.model)} with the speculative-decoding draft {_ref(f.draft)}, served with SGLang (BF16), greedy decoding, thinking mode, at most {f.max_new_tokens:,} new tokens. The prompt is the one of [`NoeFlandre/benchmark-llms-landuse-relevance`](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance).

## Quality

Only GPU types that passed a pre-registered non-inferiority gate on the full 25,500-item benchmark generated labels. Differences are against the published reference run (one-sided 95 % lower bounds must stay above -0.02; failed-rate increase under +0.5 pp).

| GPU type | Generations | ΔF1 | ΔMCC (95 % low) | Δaccuracy | Δfailed |
|---|---:|---:|---:|---:|---:|
{gates}

Greedy decoding is not batch- or GPU-invariant: about 12 % of individual decisions would flip between two runs of the reference setup. The gate guarantees aggregate quality, not per-row reproducibility.

## License and citation

Labels: ODbL, like the OpenStreetMap-derived input.{extra_license}

Code and configuration (fingerprint `{f.fingerprint}`): https://github.com/NoeFlandre/filter-osm-datasets-llms-landuse
"""
