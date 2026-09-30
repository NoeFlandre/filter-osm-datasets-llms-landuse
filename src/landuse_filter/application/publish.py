"""Incremental publication of a ``<input>-landuse`` dataset.

1. Mirror every file of the input at its pinned revision (once, byte-identical).
2. For each input file whose in-scope texts are all generated, write and upload
   ``labels/<path>``; upload the new canonical generations under ``generations/``.
3. Rewrite the dataset card with coverage; ``status`` stays ``in_progress`` until
   every in-scope file is labelled.
A file with some but not all texts generated is published as a *partial* file
(sentences without an answer are ``pending``) once it gained :data:`PARTIAL_STEP` of its
sentences since its last partial upload; it is replaced when it completes. Partial files
are recorded in ``published/<dataset>.partial.jsonl``, never in the main ledger, so they
stay open; their ``generations/`` rows ship with the complete file.
Published files are recorded in ``published/<dataset>.jsonl``, so re-running resumes.
"""

import hashlib
import math
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from landuse_filter import config
from landuse_filter.adapters import hub
from landuse_filter.adapters.indexes import ResolutionIndex
from landuse_filter.adapters.readers import DESCRIPTION, SOURCES, WEBSITE
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import published_stats
from landuse_filter.application.assemble import (
    PENDING,
    MissingGenerationError,
    Stamp,
    build_generations,
    build_labels,
    build_viewer,
)
from landuse_filter.application.card import MAP_ASSET, CardFacts, MapFacts, render_card
from landuse_filter.application.geo import BBOX, Located, description_cells, website_cells
from landuse_filter.application.results import canonical_generations

PARTIAL_STEP = 0.10  # share of a file's sentences that must be newly labelled to refresh it
PARTIAL_FLUSH = 100  # partial files uploaded (and recorded) per commit while building


@dataclass
class PublishReport:
    labelled_files: int
    total_files: int
    new_files: int
    decisions: dict[str, int]
    partial_files: int = 0


def partial_ledger(dataset: str) -> str:
    return f"published/{dataset}.partial.jsonl"


def output_repo(dataset: str) -> str:
    from landuse_filter.adapters.settings_file import load

    return load().output_repo(dataset)


@dataclass(frozen=True)
class _Ctx:
    """What building one file's tables needs, shared by every file of a run."""

    dataset: str
    revision: str
    fp: str
    out: Path
    resolved: ResolutionIndex


@dataclass(frozen=True)
class Coverage:
    labelled: int  # files whose labels are complete
    total: int
    partial: list[str]  # labels paths of partly labelled files on the Hub


class _PartialSink:
    """Uploads partial files as they are built and records each commit in the ledger, so a
    job stopped at its walltime keeps what it already published."""

    def __init__(
        self,
        store: WorkStore,
        dataset: str,
        repo: str,
        *,
        dry_run: bool,
        on_progress: Callable[[], None] | None,
    ) -> None:
        self.store, self.dataset, self.repo, self.dry_run = store, dataset, repo, dry_run
        self.on_progress = on_progress
        self.pending: list[tuple[Path, str]] = []
        self.resolved: dict[str, int] = {}  # labels path -> resolved sentences, this run
        self.uploaded: dict[str, Path] = {}

    def add(self, labels: tuple[Path, str], viewer: tuple[Path, str], known: int) -> None:
        self.pending += [labels, viewer]
        self.resolved[labels[1]] = known
        if len(self.pending) >= PARTIAL_FLUSH:
            self.flush()

    def flush(self) -> None:
        if self.pending and not self.dry_run:
            hub.upload(
                self.repo, self.pending, f"Add partial land-use labels ({config.GENERATION_FP})"
            )
            self.store.append_jsonl(
                partial_ledger(self.dataset),
                [
                    {"path": d, "resolved": self.resolved[d]}
                    for _, d in self.pending
                    if d in self.resolved
                ],
            )
            self.uploaded.update({d: src for src, d in self.pending})
            if self.on_progress:
                self.on_progress()  # e.g. save the ledger off the node that may be stopped
        self.pending = []


@dataclass
class _Run:
    """Everything one publish run shares between its steps."""

    store: WorkStore
    dataset: str
    revision: str
    repo: str
    dry_run: bool
    ctx: _Ctx
    done: set[str]  # paths recorded as published (complete files, generations, mirror marker)
    resolved_before: dict[str, int]  # labels path -> sentences resolved at its last partial upload
    sink: _PartialSink
    db: sqlite3.Connection
    files: list[str]  # scanned input files, in planner order
    first_file: dict[str, int]  # input file -> planner index


def publish(
    store: WorkStore,
    dataset: str,
    revision: str,
    *,
    dry_run: bool = False,
    on_progress: Callable[[], None] | None = None,
) -> PublishReport:
    run = _prepare(store, dataset, revision, dry_run=dry_run, on_progress=on_progress)
    new, new_shas = _build_files(run)
    new += _missing_viewers(
        dataset, run.files, run.done, revision, resolved=run.ctx.resolved, out=run.ctx.out
    )
    new += _generation_files(run, new_shas)
    return _finish(run, new)


def _prepare(
    store: WorkStore,
    dataset: str,
    revision: str,
    *,
    dry_run: bool,
    on_progress: Callable[[], None] | None,
) -> _Run:
    repo = output_repo(dataset)
    fp = config.GENERATION_FP
    done = {r["path"] for r in store.read_jsonl(f"published/{dataset}.jsonl")}
    if not dry_run:
        hub.ensure_dataset(repo)
        _mirror(
            store,
            dataset,
            input_repo=SOURCES[dataset].repo_id,
            revision=revision,
            repo=repo,
            done=done,
        )
    resolved = ResolutionIndex(store.path(f"index/resolve-{fp}.sqlite"))
    resolved.build(canonical_generations(store, fp))
    db = sqlite3.connect(store.path(f"index/{dataset}.sqlite"))
    indexed = db.execute("SELECT idx, path FROM files WHERE done = 1 ORDER BY idx").fetchall()
    files = [p for _, p in indexed]
    return _Run(
        store=store,
        dataset=dataset,
        revision=revision,
        repo=repo,
        dry_run=dry_run,
        ctx=_Ctx(dataset, revision, fp, store.path(f"publish/{dataset}"), resolved),
        done=done,
        resolved_before={
            r["path"]: r["resolved"] for r in store.read_jsonl(partial_ledger(dataset))
        },
        sink=_PartialSink(store, dataset, repo, dry_run=dry_run, on_progress=on_progress),
        db=db,
        files=files,
        first_file=dict(zip(files, [i for i, _ in indexed], strict=True)),
    )


def _build_files(run: _Run) -> tuple[list[tuple[Path, str]], set[str]]:
    """Build the tables of every unpublished file: complete ones are returned (with the texts
    whose generation ships with them); partial ones go to the sink as they are built."""
    ctx, out = run.ctx, run.ctx.out
    new: list[tuple[Path, str]] = []
    new_shas: set[str] = set()
    for path in run.files:
        target = f"labels/{path}"
        if target in run.done:
            continue
        local = Path(hub.download_all(SOURCES[run.dataset].repo_id, run.revision, [path])[0][0])
        try:
            build_labels(
                run.dataset,
                path,
                local,
                resolved=ctx.resolved,
                stamp=Stamp(ctx.fp, run.revision),
                out=out,
            )
        except MissingGenerationError:
            known = _build_partial(ctx, path, local, last=run.resolved_before.get(target, 0))
            if known is not None:
                labels, viewer = (out / target, target), (out / f"viewer/{path}", f"viewer/{path}")
                run.sink.add(labels, viewer, known)
            continue
        new_shas |= _owned_generations(run.db, run.first_file[path], out / target)
        new += [(out / target, target), (out / f"viewer/{path}", f"viewer/{path}")]
    return new, new_shas


def _generation_files(run: _Run, new_shas: set[str]) -> list[tuple[Path, str]]:
    """The generations shipping with this run's complete files, as one numbered batch."""
    if not new_shas:
        return []
    fp = run.ctx.fp
    batch = len(list(run.store.read_jsonl(f"published/{run.dataset}.jsonl")))
    gen_dir = run.store.path(f"publish/{run.dataset}-gen-{batch:05d}")
    build_generations(run.store, fp, gen_dir, new_shas)
    return [
        (p, f"generations/{fp}/batch-{batch:05d}-{p.name}")
        for p in sorted((gen_dir / "generations" / fp).glob("*.parquet"))
    ]


def _finish(run: _Run, new: list[tuple[Path, str]]) -> PublishReport:
    """Upload what is left, refresh the card and report the coverage."""
    completed = {d for _, d in new if d.startswith("labels/")}
    labelled = len([p for p in run.files if f"labels/{p}" in run.done]) + len(completed)
    run.sink.flush()
    partial_paths = (set(run.resolved_before) | set(run.sink.resolved)) - run.done - completed
    decisions = _decision_counts(run.ctx.out)
    if not run.dry_run:
        decisions = _upload(
            run.store,
            (run.dataset, run.repo, run.revision),
            new,
            run.sink.uploaded,
            Coverage(labelled, len(run.files), sorted(partial_paths)),
        )
    return PublishReport(labelled, len(run.files), len(new), decisions, len(partial_paths))


def _upload(
    store: WorkStore,
    target: tuple[str, str, str],
    new: list[tuple[Path, str]],
    partial: dict[str, Path],
    coverage: Coverage,
) -> dict[str, int]:
    """Upload the complete files, record them and refresh the card (partial files are
    already up and recorded)."""
    dataset, repo, revision = target
    if new:
        hub.upload(repo, new, f"Add land-use labels ({config.GENERATION_FP})")
        store.append_jsonl(f"published/{dataset}.jsonl", [{"path": d} for _, d in new])
    return _refresh_card(
        store,
        dataset,
        repo=repo,
        revision=revision,
        local={**partial, **{d: src for src, d in new}},
        coverage=coverage,
    )


def _owned_generations(db: sqlite3.Connection, file_idx: int, labels: Path) -> set[str]:
    """Generated texts whose generation ships with this file: those that first appear in it
    (the planner index records that file), so repeated texts are never uploaded twice."""
    table = pq.read_table(labels, columns=["text_sha256", "generation_id"])
    owned = {sha for (sha,) in db.execute("SELECT sha FROM texts WHERE file_idx = ?", (file_idx,))}
    return {
        s
        for s, g in zip(
            table.column("text_sha256").to_pylist(),
            table.column("generation_id").to_pylist(),
            strict=True,
        )
        if g and s in owned
    }


def _build_partial(ctx: _Ctx, path: str, local: Path, *, last: int) -> int | None:
    """Build the partial tables of a file; ``None`` (and nothing left in ``out``) if it has
    not gained :data:`PARTIAL_STEP` of its sentences since the last partial upload."""
    build_labels(
        ctx.dataset,
        path,
        local,
        resolved=ctx.resolved,
        stamp=Stamp(ctx.fp, ctx.revision),
        out=ctx.out,
        allow_pending=True,
    )
    decisions = pq.read_table(ctx.out / "labels" / path, columns=["decision"]).column("decision")
    decisions = decisions.to_pylist()
    known = sum(d not in (PENDING, "skipped_unsplit") for d in decisions)
    sendable = sum(d != "skipped_unsplit" for d in decisions)
    if known - last >= max(1, math.ceil(PARTIAL_STEP * sendable)):
        return known
    (ctx.out / "labels" / path).unlink()
    (ctx.out / "viewer" / path).unlink()
    return None


def _missing_viewers(
    dataset: str,
    files: list[str],
    done: set[str],
    revision: str,
    *,
    resolved: ResolutionIndex,
    out: Path,
) -> list[tuple[Path, str]]:
    """Viewer tables of files that were labelled before the viewer table existed."""
    made = []
    for path in files:
        target = f"viewer/{path}"
        if f"labels/{path}" not in done or target in done:
            continue
        local = Path(hub.download_all(SOURCES[dataset].repo_id, revision, [path])[0][0])
        build_viewer(dataset, path, local, resolved=resolved, out=out)
        made.append((out / target, target))
    return made


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
    coverage: Coverage,
) -> dict[str, int]:
    """Recount the card's numbers from every published file, then update the card."""
    on_hub = {r["path"] for r in store.read_jsonl(f"published/{dataset}.jsonl")}
    stats = published_stats.totals(
        published_stats.complete(
            store,
            dataset,
            on_hub | set(coverage.partial) | set(local),
            lambda p: local[p] if p in local else hub.open_file(repo, p),
            _locator(dataset, revision),
            refresh=set(local),  # uploaded now: a partial record of the same path is stale
        )
    )
    if stats["decisions"]:
        _card(
            store,
            dataset,
            repo=repo,
            revision=revision,
            coverage=coverage,
            stats=stats,
        )
    return stats["decisions"]


def _locator(dataset: str, revision: str) -> published_stats.Locator | None:
    """Where the rows of a labels file are, for datasets whose input has coordinates."""
    if dataset not in (DESCRIPTION, WEBSITE):
        return None
    input_repo = SOURCES[dataset].repo_id

    def locate(path: str, labels_source: published_stats.Source) -> Located:
        from landuse_filter.adapters import hexmap

        rel = path.removeprefix("labels/")
        if dataset == WEBSITE:
            return website_cells(
                pq.read_table(labels_source, columns=["polygon_id", "decision"]),
                pq.read_table(
                    hub.open_file(input_repo, rel, revision), columns=["polygon_id", "lat", "lon"]
                ),
                hexmap.cell_of,
            )
        language_file = hub.open_file(input_repo, rel, revision)
        polygon_file = hub.open_file(input_repo, f"data/{Path(rel).name}", revision)
        return description_cells(
            pq.read_table(labels_source, columns=["description_identity", "decision"]),
            pq.read_table(language_file, columns=["description_identity", "osm_type", "osm_id"]),
            pq.read_table(polygon_file, columns=["osm_type", "osm_id", *BBOX]),
            hexmap.cell_of,
        )

    return locate


def _world_map(stats: dict, png: Path, dataset: str) -> MapFacts | None:
    """Render the yes-share map and return its figures; ``None`` without coordinates."""
    if not stats["cells"]:
        return None
    from landuse_filter.adapters import hexmap
    from landuse_filter.domain import geomap

    png.parent.mkdir(parents=True, exist_ok=True)
    hexmap.render(stats["cells"], png, title=f"{dataset}-landuse: share of yes per H3 cell")
    return MapFacts(
        cells=len(stats["cells"]),
        located=stats["located"],
        labelled=stats["labelled"],
        share=geomap.global_share(stats["cells"]),
    )


def _card(
    store: WorkStore,
    dataset: str,
    *,
    repo: str,
    revision: str,
    coverage: Coverage,
    stats: dict,
) -> None:
    cfg = config.reference_config()
    png = store.path(f"publish/{dataset}/{MAP_ASSET}")
    world_map = _world_map(stats, png, dataset)
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
            labelled_files=coverage.labelled,
            total_files=coverage.total,
            decisions=stats["decisions"],
            failures=stats["failures"],
            unique_texts=stats["unique_texts"],
            gpu_rows=stats["gpus"],
            admitted=admitted,
            world_map=world_map,
            partial_files=len(coverage.partial),
        )
    )
    files = [("README.md", text.encode())]
    if world_map:
        files.append((MAP_ASSET, png.read_bytes()))
    digest = hashlib.sha256(b"\0".join(name.encode() + data for name, data in files)).hexdigest()
    marker = store.path(f"published/{dataset}.card.sha256")
    if marker.exists() and marker.read_text().strip() == digest:
        return
    path = store.path(f"publish/{dataset}/README.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    uploads = [(path, "README.md"), *([(png, MAP_ASSET)] if world_map else [])]
    hub.upload(repo, uploads, "Update dataset card")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(digest + "\n")
