"""Incremental publication of a ``<input>-landuse`` dataset.

1. Mirror every file of the input at its pinned revision (once, byte-identical).
2. For each input file whose in-scope texts are all generated, write and upload
   ``labels/<path>``; upload the new canonical generations under ``generations/``.
3. Rewrite the dataset card with coverage; ``status`` stays ``in_progress`` until
   every in-scope file is labelled.
Published files are recorded in ``published/<dataset>.jsonl``, so re-running resumes.
"""

import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from landuse_filter import config
from landuse_filter.adapters import hub
from landuse_filter.adapters.readers import SOURCES
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.assemble import (
    MissingGenerationError,
    build_generations,
    build_labels,
    resolve_all,
)


@dataclass
class PublishReport:
    labelled_files: int
    total_files: int
    new_files: int
    decisions: dict[str, int]


def output_repo(dataset: str) -> str:
    return f"NoeFlandre/{dataset}-landuse"


def publish(
    store: WorkStore, dataset: str, revision: str, *, dry_run: bool = False
) -> PublishReport:
    source = SOURCES[dataset]
    repo = output_repo(dataset)
    fp = config.GENERATION_FP
    out = store.path(f"publish/{dataset}")
    done = {r["path"] for r in store.read_jsonl(f"published/{dataset}.jsonl")}
    if not dry_run:
        hub.ensure_dataset(repo)
        _mirror(store, dataset, source.repo_id, revision, repo, done)
    resolved = resolve_all(store, fp)
    db = sqlite3.connect(store.path(f"index/{dataset}.sqlite"))
    files = [p for (p,) in db.execute("SELECT path FROM files WHERE done = 1 ORDER BY idx")]
    new: list[tuple[Path, str]] = []
    new_shas: set[str] = set()
    for path in files:
        target = f"labels/{path}"
        if target in done:
            continue
        local = Path(hub.download_all(source.repo_id, revision, [path])[0][0])
        try:
            build_labels(dataset, path, local, resolved=resolved, fp=fp, revision=revision, out=out)
        except MissingGenerationError:
            continue
        table = pq.read_table(out / target, columns=["text_sha256", "generation_id"])
        new_shas.update(
            s
            for s, g in zip(
                *[table.column(c).to_pylist() for c in ("text_sha256", "generation_id")],
                strict=True,
            )
            if g
        )
        new.append((out / target, target))
    if new_shas:
        batch = len(list(store.read_jsonl(f"published/{dataset}.jsonl")))
        gen_dir = store.path(f"publish/{dataset}-gen-{batch:05d}")
        build_generations(store, fp, gen_dir, new_shas)
        new += [
            (p, f"generations/{fp}/batch-{batch:05d}-{p.name}")
            for p in sorted((gen_dir / "generations" / fp).glob("*.parquet"))
        ]
    labelled = len([p for p in files if f"labels/{p}" in done]) + sum(
        1 for _, d in new if d.startswith("labels/")
    )
    decisions = _decision_counts(out)
    if new and not dry_run:
        hub.upload(repo, new, f"Add land-use labels ({fp})")
        store.append_jsonl(f"published/{dataset}.jsonl", [{"path": d} for _, d in new])
        _card(store, dataset, repo, revision, labelled, len(files), decisions)
    return PublishReport(labelled, len(files), len(new), decisions)


def _mirror(
    store: WorkStore, dataset: str, input_repo: str, revision: str, repo: str, done: set[str]
) -> None:
    marker = f"mirror:{revision}"
    if marker in done:
        return
    present = hub.remote_files(repo)
    wanted = [
        p for p in hub.list_files(input_repo, revision) if p not in present and p != "README.md"
    ]
    for start in range(0, len(wanted), hub.BATCH):
        hub.upload(
            repo,
            hub.download_all(input_repo, revision, wanted[start : start + hub.BATCH]),
            f"Mirror {input_repo}@{revision[:7]}",
        )
    store.append_jsonl(f"published/{dataset}.jsonl", [{"path": marker}])


def _decision_counts(out: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in (out / "labels").rglob("*.parquet"):
        counts.update(pq.read_table(path, columns=["decision"]).column("decision").to_pylist())
    return dict(counts)


def _card(
    store: WorkStore,
    dataset: str,
    repo: str,
    revision: str,
    labelled: int,
    total: int,
    decisions: dict[str, int],
) -> None:
    status = "complete" if labelled == total else "in_progress"
    cfg = config.reference_config()
    rows = "\n".join(f"| `{k}` | {v:,} |" for k, v in sorted(decisions.items()))
    text = f"""---
license: odbl
pretty_name: {dataset} (land-use labels)
tags: [openstreetmap, land-use, land-cover, remote-sensing, geospatial]
dataset_status: {status}
configs:
- config_name: labels
  data_files: "labels/**/*.parquet"
- config_name: generations
  data_files: "generations/**/*.parquet"
---
# {dataset}-landuse

Every in-scope sentence of [`NoeFlandre/{dataset}`](https://huggingface.co/datasets/NoeFlandre/{dataset})
(revision `{revision}`, mirrored here unchanged) labelled for land-use / land-cover
relevance by **{cfg["model"]}** with the DSpark draft **{cfg["draft"]}** (SGLang,
BF16, greedy, thinking mode, `max_new_tokens={cfg["sampling"]["max_new_tokens"]}`).

Status: **{status}** — {labelled:,} / {total:,} input files labelled.

| Decision | Sentences |
|---|---:|
{rows}

* `labels/<input path>.parquet`: one row per sentence position (`label_id` + the
  input's join keys), `decision` ∈ `yes`, `no`, `failed`, `skipped_unsplit`, parse
  mode / failure reason, `generation_id`.
* `generations/`: one row per unique text: raw output (including reasoning), token
  counts, finish reason, DSpark acceptance, GPU, site, job, code commit.

Nothing from the input is removed. Prompt and serving configuration are those of the
benchmark [`NoeFlandre/benchmark-llms-landuse-relevance`](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance);
the implementation passed its non-inferiority gate before any production run.
Code: https://github.com/NoeFlandre/filter-osm-datasets-llms-landuse · config fingerprint `{config.GENERATION_FP}`.
"""
    path = store.path(f"publish/{dataset}/README.md")
    path.write_text(text, encoding="utf-8")
    hub.upload(repo, [(path, "README.md")], "Update dataset card")
