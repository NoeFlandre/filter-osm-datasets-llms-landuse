"""Incremental publication of a ``<input>-landuse`` dataset.

1. Mirror every file of the input at its pinned revision (once, byte-identical).
2. For each input file whose in-scope texts are all generated, write and upload
   ``labels/<path>``; upload the new canonical generations under ``generations/``.
3. Rewrite the dataset card with coverage; ``status`` stays ``in_progress`` until
   every in-scope file is labelled.
Published files are recorded in ``published/<dataset>.jsonl``, so re-running resumes.
"""

import hashlib
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from landuse_filter import config
from landuse_filter.adapters import hub
from landuse_filter.adapters.indexes import ResolutionIndex
from landuse_filter.adapters.readers import SOURCES
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import published_stats
from landuse_filter.application.assemble import (
    MissingGenerationError,
    build_generations,
    build_labels,
)
from landuse_filter.application.card import CardFacts, render_card
from landuse_filter.application.results import canonical_generations


@dataclass
class PublishReport:
    labelled_files: int
    total_files: int
    new_files: int
    decisions: dict[str, int]


def output_repo(dataset: str) -> str:
    from landuse_filter.adapters.settings_file import load

    return load().output_repo(dataset)


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
        _mirror(store, dataset, input_repo=source.repo_id, revision=revision, repo=repo, done=done)
    resolved = ResolutionIndex(store.path(f"index/resolve-{fp}.sqlite"))
    resolved.build(canonical_generations(store, fp))
    db = sqlite3.connect(store.path(f"index/{dataset}.sqlite"))
    indexed = db.execute("SELECT idx, path FROM files WHERE done = 1 ORDER BY idx").fetchall()
    files = [p for _, p in indexed]
    first_file = dict(zip(files, [i for i, _ in indexed], strict=True))
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
        # A generation ships with the file where its text first appears (the planner index
        # records that file), so repeated texts are never uploaded twice.
        owned = {
            sha
            for (sha,) in db.execute(
                "SELECT sha FROM texts WHERE file_idx = ?", (first_file[path],)
            )
        }
        new_shas.update(
            s
            for s, g in zip(
                *[table.column(c).to_pylist() for c in ("text_sha256", "generation_id")],
                strict=True,
            )
            if g and s in owned
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
    if not dry_run:
        if new:
            hub.upload(repo, new, f"Add land-use labels ({fp})")
            store.append_jsonl(f"published/{dataset}.jsonl", [{"path": d} for _, d in new])
        decisions = _refresh_card(
            store,
            dataset,
            repo=repo,
            revision=revision,
            local={d: src for src, d in new},
            labelled=labelled,
            total=len(files),
        )
    return PublishReport(labelled, len(files), len(new), decisions)


def _mirror(
    store: WorkStore, dataset: str, *, input_repo: str, revision: str, repo: str, done: set[str]
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


def _refresh_card(
    store: WorkStore,
    dataset: str,
    *,
    repo: str,
    revision: str,
    local: dict[str, Path],
    labelled: int,
    total: int,
) -> dict[str, int]:
    """Recount the card's numbers from every published file, then update the card."""
    on_hub = {r["path"] for r in store.read_jsonl(f"published/{dataset}.jsonl")}
    stats = published_stats.totals(
        published_stats.complete(
            store,
            dataset,
            on_hub | set(local),
            lambda p: local[p] if p in local else hub.open_file(repo, p),
        )
    )
    if stats["decisions"]:
        _card(
            store,
            dataset,
            repo=repo,
            revision=revision,
            labelled=labelled,
            total=total,
            stats=stats,
        )
    return stats["decisions"]


def _card(
    store: WorkStore,
    dataset: str,
    *,
    repo: str,
    revision: str,
    labelled: int,
    total: int,
    stats: dict,
) -> None:
    cfg = config.reference_config()
    admitted = {
        p.stem: store.read_json(str(p.relative_to(store.root)))["gate"]
        for p in sorted(store.path("gates/admission").glob("*.json"))
        if store.read_json(str(p.relative_to(store.root))).get("status") == "admitted"
    }
    text = render_card(
        CardFacts(
            dataset=dataset,
            revision=revision,
            model=cfg["model"],
            draft=cfg["draft"],
            max_new_tokens=cfg["sampling"]["max_new_tokens"],
            fingerprint=config.GENERATION_FP,
            labelled_files=labelled,
            total_files=total,
            decisions=stats["decisions"],
            failures=stats["failures"],
            unique_texts=stats["unique_texts"],
            gpu_rows=stats["gpus"],
            admitted=admitted,
        )
    )
    digest = hashlib.sha256(text.encode()).hexdigest()
    marker = store.path(f"published/{dataset}.card.sha256")
    if marker.exists() and marker.read_text().strip() == digest:
        return
    path = store.path(f"publish/{dataset}/README.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    hub.upload(repo, [(path, "README.md")], "Update dataset card")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(digest + "\n")
