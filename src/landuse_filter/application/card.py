"""Dataset card for a ``-landuse`` output repo (pure rendering; the publisher uploads it)."""

from collections.abc import Mapping
from dataclasses import dataclass

from landuse_filter.application.datasets import SPECS

FAILURES = {
    "truncated": "hit the token limit before answering",
    "unclosed_think": "never closed its reasoning block",
    "empty": "nothing after the reasoning",
    "ambiguous": "answer mentions both labels",
    "non_english_token": "answered with a non-English word that is not mapped",
    "no_label": "no yes/no in the answer",
}
DECISION_ORDER = ("yes", "no", "failed", "skipped_unsplit", "pending")
DECISION_MEANING = {
    "yes": "relevant to land use / land cover",
    "no": "not relevant",
    "failed": "the model output could not be parsed into yes/no",
    "skipped_unsplit": "text not segmented upstream; not sent to the model",
    "pending": "in a partly labelled file; no model answer yet",
}


MAP_ASSET = "assets/yes_share_map.png"


@dataclass(frozen=True, slots=True)
class MapFacts:
    """Figures of the world map, all counted from the published label tables."""

    cells: int
    located: int  # yes+no sentences placed on the map
    labelled: int  # yes+no sentences in total
    share: float  # dataset-wide share of yes among the located sentences


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
    world_map: MapFacts | None = None  # None: the input has no coordinates
    partial_files: int = 0  # files published before all their sentences were labelled


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
        if total and (k != "pending" or decisions.get(k))
    ]
    rows.append(f"| **total** | **{total:,}** | | |")
    return "\n".join(rows)


def _map_section(m: MapFacts | None, dataset: str) -> str:
    if m is None:
        return ""
    return f"""
## Where the labels are

![Share of yes among yes/no sentences per H3 cell]({MAP_ASSET})

Each hexagon is an [H3](https://h3geo.org) cell (resolution 3) holding the `yes`/`no` sentences {SPECS[dataset].placement}. Colour is the share of `yes` in the cell, centred on the dataset-wide share ({m.share:.1%}); grey cells hold fewer than 10 sentences. {m.located:,} of {m.labelled:,} `yes`/`no` sentences ({m.located / m.labelled:.1%}) are placed, in {m.cells:,} cells; `failed` and `skipped_unsplit` rows are not.
"""


def _failure_table(failures: Mapping[str, int]) -> str:
    failed = sum(failures.values())
    return "\n".join(
        f"| `{k}` | {failures[k]:,} | {failures[k] / failed:.1%} | {FAILURES.get(k, '')} |"
        for k in sorted(failures, key=lambda r: (-failures[r], r))
    )


def _generations_config(f: CardFacts) -> str:
    """The `generations` config only once a generations file exists: an empty glob makes the
    Hub's dataset viewer fail that config (seen on the first partial website publish)."""
    if not f.unique_texts:
        return ""
    return '- config_name: generations\n  data_files: "generations/**/*.parquet"\n'


def _front_matter(f: CardFacts, status: str) -> str:
    return f"""---
license: odbl
pretty_name: {f.dataset} (land-use labels)
language: [multilingual]
task_categories: [text-classification]
tags: [openstreetmap, land-use, land-cover, geospatial]
dataset_status: {status}
configs:
- config_name: sentences
  default: true
  data_files: "viewer/**/*.parquet"
- config_name: labels
  data_files: "labels/**/*.parquet"
{_generations_config(f)}---
"""


def _texts_clause(f: CardFacts, status: str) -> str:
    if f.unique_texts and status == "complete":
        return f"{f.unique_texts:,} unique texts sent to the model"
    if f.unique_texts:
        return (
            f"{f.unique_texts:,} unique texts have their full model outputs published so far "
            "(outputs ship with the first complete file that contains the text; "
            "labels for the other sentences are in `labels/`)"
        )
    return "generations are published with each completed input file"


def _files_line(f: CardFacts) -> str:
    if f.partial_files:
        return (
            f"{f.labelled_files:,} of {f.total_files:,} input files fully labelled, "
            f"{f.partial_files:,} more partially"
        )
    return f"{f.labelled_files:,} of {f.total_files:,} input files labelled"


def _intro(f: CardFacts, status: str, total: int) -> str:
    return f"""# {f.dataset}-landuse

A land-use / land-cover relevance label for every sentence of [`NoeFlandre/{f.dataset}`](https://huggingface.co/datasets/NoeFlandre/{f.dataset}) (revision `{f.revision[:7]}`). The input is mirrored here unchanged; labels and model outputs are separate tables that join back to it.

**{status.replace("_", " ").capitalize()}: {_files_line(f)}, {total:,} rows, {_texts_clause(f, status)}.**

| `decision` | Rows | Share | Meaning |
|---|---:|---:|---|
{_decision_table(f.decisions)}
"""


def _failed_section(f: CardFacts, total: int) -> str:
    if not f.failures:
        return ""
    return f"""
### Failed rows

{sum(f.failures.values()):,} rows ({f.decisions.get("failed", 0) / total:.1%} of all rows) have no
usable answer. Reasons:

| `failure_reason` | Rows | Share | Meaning |
|---|---:|---:|---|
{_failure_table(f.failures)}
"""


def _tables_section(f: CardFacts) -> str:
    spec = SPECS[f.dataset]
    source, using, keys = spec.source.patterns, spec.card_using, spec.card_keys
    partial_note = (
        "\n* `pending` rows belong to partly labelled files and are only in `labels/`, never in the "
        "viewer; their labels are refreshed as the "
        "model works, and the `generations/` rows of a file are published once the file is complete."
        if f.partial_files
        else ""
    )
    return f"""
## Tables

* `viewer/<input path>.parquet` (the default view): `sentence`, `label`, `language` and `region` (the input file), nothing else. It holds only sentences with a model answer (`yes`, `no`, `failed`) or skipped (`skipped_unsplit`) and with at least 2 letters; debris such as `-` or `7` stays in `labels/`.
* `labels/<input path>.parquet`: one row per sentence. Join keys ({keys}), `text_sha256`, `decision`, `parse_mode`, `failure_reason`, `generation_id`.
* `generations/<fingerprint>/*.parquet`: one row per **unique** text: raw output including the reasoning, token counts, finish reason, speculative-decoding statistics, GPU, site, job and code commit. Identical texts are generated once and share a `generation_id`.{partial_note}

```sql
SELECT i.*, l.decision, g.raw_output
FROM {_from(source)} i
JOIN {_from(source, "labels/")} l {using}
LEFT JOIN 'generations/*/*.parquet' g USING (generation_id);
```
"""


def _method_section(f: CardFacts) -> str:
    return f"""
## Method

{_ref(f.model)} with the speculative-decoding draft {_ref(f.draft)}, served with SGLang (BF16), greedy decoding, thinking mode, at most {f.max_new_tokens:,} new tokens. The prompt is the one of [`NoeFlandre/benchmark-llms-landuse-relevance`](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance).
"""


def _quality_section(f: CardFacts) -> str:
    generated = sum(f.gpu_rows.values())
    used = sorted(g for g in f.gpu_rows if g in f.admitted)
    gates = "\n".join(
        _gate_row(g, f.gpu_rows[g], f.gpu_rows[g] / generated, f.admitted[g]) for g in used
    )
    return f"""
## Quality

Only GPU types that passed a pre-registered non-inferiority gate on the full 25,500-item benchmark generated labels. Differences are against the published reference run (one-sided 95 % lower bounds must stay above -0.02; failed-rate increase under +0.5 pp).

| GPU type | Generations | ΔF1 | ΔMCC (95 % low) | Δaccuracy | Δfailed |
|---|---:|---:|---:|---:|---:|
{gates}

Greedy decoding is not batch- or GPU-invariant: about 12 % of individual decisions would flip between two runs of the reference setup. The gate guarantees aggregate quality, not per-row reproducibility.
"""


def _license_section(f: CardFacts) -> str:
    license_text = SPECS[f.dataset].text_license
    extra = f"\n{license_text}" if license_text else ""
    return f"""
## License and citation

Labels: ODbL, like the OpenStreetMap-derived input.{extra}

Code and configuration (fingerprint `{f.fingerprint}`): https://github.com/NoeFlandre/filter-osm-datasets-llms-landuse
"""


def render_card(f: CardFacts) -> str:
    status = "complete" if f.labelled_files == f.total_files else "in_progress"
    total = sum(f.decisions.values())
    return (
        _front_matter(f, status)
        + _intro(f, status, total)
        + _failed_section(f, total)
        + _map_section(f.world_map, f.dataset)
        + _tables_section(f)
        + _method_section(f)
        + _quality_section(f)
        + _license_section(f)
    )
