"""Incremental publication of a ``<input>-landuse`` dataset.

1. Mirror every file of the input at its pinned revision (once, byte-identical).
2. For each input file whose in-scope texts are all generated, write and upload
   ``labels/<path>``; upload the new canonical generations under ``generations/``.
3. Rewrite the dataset card with coverage; ``status`` stays ``in_progress`` until
   every in-scope file is labelled.
A file with some but not all texts generated is published as a *partial* file
(sentences without an answer are ``pending``) once it gained :data:`PARTIAL_STEP` of its
sentences since its last partial upload (a file never published yet, as soon as one sentence is
resolved); it is replaced when it completes. Partial files
are recorded in ``published/<dataset>.partial.jsonl``, never in the main ledger, so they
stay open; their ``generations/`` rows ship with the complete file.
Published files are recorded in ``published/<dataset>.jsonl``, so re-running resumes.
"""

import contextlib
import hashlib
import logging
import math
import sqlite3
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from landuse_filter import config
from landuse_filter.adapters.hub import BATCH, HfHub, Hub
from landuse_filter.adapters.indexes import ResolutionIndex
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import published_stats
from landuse_filter.application.assemble import (
    Lookup,
    Stamp,
    build_generations,
    build_labels,
    build_viewer,
)
from landuse_filter.application.card import MAP_ASSET, CardFacts, MapFacts, render_card
from landuse_filter.application.datasets import SPECS
from landuse_filter.application.results import canonical_generations
from landuse_filter.domain.sentences import SentenceRef

PARTIAL_STEP = 0.01  # share of a file's sentences that must be newly labelled to refresh it
PARTIAL_FLUSH = 100  # partial files uploaded (and recorded) per commit while building
MIRROR_WORKERS = 4  # parallel downloads while mirroring the input repo

log = logging.getLogger(__name__)


@dataclass
class PublishReport:
    labelled_files: int
    total_files: int
    new_files: int
    decisions: dict[str, int]
    partial_files: int = 0
    stopped: str | None = None  # why the run ended early (signal, deadline), else None


def partial_ledger(dataset: str) -> str:
    return f"published/{dataset}.partial.jsonl"


def output_repo(dataset: str) -> str:
    from landuse_filter.adapters.settings_file import load

    return load().output_repo(dataset)


class _Stop:
    """A stop request that, once seen, stays seen: the reason of the first check that fired."""

    def __init__(self, check: Callable[[], str | None]) -> None:
        self.check = check
        self.reason: str | None = None

    def __call__(self) -> bool:
        if self.reason is None:
            self.reason = self.check()
        return self.reason is not None


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
        hub: Hub,
        *,
        dry_run: bool,
        on_progress: Callable[[], None] | None,
    ) -> None:
        self.store, self.dataset, self.repo, self.dry_run = store, dataset, repo, dry_run
        self.hub = hub
        self.on_progress = on_progress
        self.pending: list[tuple[Path, str]] = []
        self.resolved: dict[str, int] = {}  # labels path -> resolved sentences, this run
        self.uploaded: dict[str, Path] = {}
        self.unrefreshed: dict[str, Path] = {}  # uploaded, not yet counted into the card
        self.refresh_card: Callable[[dict[str, Path]], None] | None = None

    def add(self, files: list[tuple[Path, str]], known: int) -> None:
        self.pending += files
        labels = files[0]
        self.resolved[labels[1]] = known
        if len(self.pending) >= PARTIAL_FLUSH:
            self.flush()

    def flush(self) -> None:
        if self.pending and not self.dry_run:
            self.hub.upload(
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
            batch = {d: src for src, d in self.pending}
            self.uploaded.update(batch)
            self.unrefreshed.update(batch)
            if self.on_progress:
                self.on_progress()  # e.g. save the ledger off the node that may be stopped
            self._refresh()
        self.pending = []

    def _refresh(self) -> None:
        """Bring the card up to date with what is on the Hub; a failure must not fail the job
        (the final refresh retries it, with the same still-uncounted files)."""
        if self.refresh_card is None:
            return
        try:
            self.refresh_card(dict(self.unrefreshed))
            self.unrefreshed = {}
            if self.on_progress:
                self.on_progress()  # the card marker too
        except Exception:
            log.exception("progressive card refresh failed; continuing")


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
    hub: Hub
    db: sqlite3.Connection
    files: list[str]  # scanned input files, in planner order
    first_file: dict[str, int]  # input file -> planner index
    stop: _Stop


def publish(
    store: WorkStore,
    dataset: str,
    revision: str,
    *,
    dry_run: bool = False,
    on_progress: Callable[[], None] | None = None,
    hub: Hub | None = None,
    should_stop: Callable[[], str | None] = lambda: None,
) -> PublishReport:
    """Publish what is new. ``should_stop`` returns a reason once the run must wind down
    (signal, deadline): the current file is finished, then the partial sink is flushed, the card
    refreshed and the ledgers saved as in a normal end, and the report says why it stopped."""
    run = _prepare(
        store,
        dataset,
        revision,
        hub or HfHub(),
        dry_run=dry_run,
        on_progress=on_progress,
        stop=_Stop(should_stop),
    )
    _card_from_ledgers(run)
    new, new_shas = _build_files(run)
    new += _missing_viewers(run, resolved=run.ctx.resolved, out=run.ctx.out)
    new += _generation_files(run, new_shas)  # ships with the complete files already built
    return _finish(run, new)


def _prepare(
    store: WorkStore,
    dataset: str,
    revision: str,
    hub: Hub,
    *,
    dry_run: bool,
    on_progress: Callable[[], None] | None,
    stop: _Stop,
) -> _Run:
    repo = output_repo(dataset)
    fp = config.GENERATION_FP
    done = {r["path"] for r in store.read_jsonl(f"published/{dataset}.jsonl")}
    if not dry_run:
        hub.ensure_dataset(repo)
        _mirror(
            hub,
            store,
            dataset,
            input_repo=SPECS[dataset].source.repo_id,
            revision=revision,
            repo=repo,
            done=done,
            on_progress=on_progress,
            should_stop=stop,
        )
    resolved = ResolutionIndex(store.path(f"index/resolve-{fp}.sqlite"))
    resolved.build(canonical_generations(store, fp))
    db = sqlite3.connect(store.path(f"index/{dataset}.sqlite"))
    indexed = db.execute("SELECT idx, path FROM files WHERE done = 1 ORDER BY idx").fetchall()
    files = [p for _, p in indexed]
    run = _Run(
        store=store,
        dataset=dataset,
        revision=revision,
        repo=repo,
        dry_run=dry_run,
        ctx=_Ctx(dataset, revision, fp, store.path(f"publish/{dataset}"), resolved),
        done=done,
        resolved_before={
            r["path"]: r["resolved"] for r in store.compact_jsonl(partial_ledger(dataset))
        },
        sink=_PartialSink(store, dataset, repo, hub, dry_run=dry_run, on_progress=on_progress),
        hub=hub,
        db=db,
        files=files,
        first_file=dict(zip(files, [i for i, _ in indexed], strict=True)),
        stop=stop,
    )
    run.sink.refresh_card = lambda local: _progress_card(run, local)
    return run


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
        if run.stop():
            break
        local = Path(
            run.hub.download_all(SPECS[run.dataset].source.repo_id, run.revision, [path])[0][0]
        )
        refs = list(SPECS[run.dataset].source.read(local, path))
        known, sendable, complete = _resolved_counts(refs, ctx.resolved)
        if not complete:
            if _worth_publishing(known, sendable, last=run.resolved_before.get(target, 0)):
                build_labels(
                    run.dataset,
                    path,
                    local,
                    resolved=ctx.resolved,
                    stamp=Stamp(ctx.fp, run.revision),
                    out=out,
                    allow_pending=True,
                    refs=refs,
                )
                run.sink.add(_tables(out, path), known)
            continue
        build_labels(
            run.dataset,
            path,
            local,
            resolved=ctx.resolved,
            stamp=Stamp(ctx.fp, run.revision),
            out=out,
            refs=refs,
        )
        new_shas |= _owned_generations(run.db, run.first_file[path], out / target)
        new += _tables(out, path)
    return new, new_shas


def _tables(out: Path, path: str) -> list[tuple[Path, str]]:
    """The labels table of a built file and its viewer table, when it has rows to show."""
    files = [(out / "labels" / path, f"labels/{path}")]
    if (out / "viewer" / path).exists():
        files.append((out / "viewer" / path, f"viewer/{path}"))
    return files


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
    decisions: dict[str, int] = {}
    if run.dry_run:
        decisions = _decision_counts(run.ctx.out)
    else:
        decisions = _upload(
            run.store,
            run.hub,
            (run.dataset, run.repo, run.revision),
            new,
            partial=run.sink.unrefreshed,
            coverage=Coverage(labelled, len(run.files), sorted(partial_paths)),
            on_progress=run.sink.on_progress,
        )
    return PublishReport(
        labelled, len(run.files), len(new), decisions, len(partial_paths), run.stop.reason
    )


def _card_from_ledgers(run: _Run) -> None:
    """Bring the card in line with the ledgers before the slow loop over input files: a run
    stopped at its walltime while downloading them must not leave a stale card."""
    if run.dry_run or not (run.done or run.resolved_before):
        return
    try:
        _progress_card(run, {})
    except Exception:
        log.exception("card refresh at the start failed; continuing")


def _progress_card(run: _Run, local: dict[str, Path]) -> None:
    """Refresh the card after a partial flush from the ledgers as they stand: ``local`` are the
    partial files uploaded since the last refresh (their old records are stale)."""
    partial = (
        set(run.resolved_before) | set(run.sink.uploaded) & set(run.sink.resolved)
    ) - run.done
    labelled = len([p for p in run.files if f"labels/{p}" in run.done])
    _refresh_card(
        run.store,
        run.hub,
        run.dataset,
        repo=run.repo,
        revision=run.revision,
        local=local,
        coverage=Coverage(labelled, len(run.files), sorted(partial)),
        on_progress=run.sink.on_progress,
    )


def _upload(
    store: WorkStore,
    hub: Hub,
    target: tuple[str, str, str],
    new: list[tuple[Path, str]],
    *,
    partial: dict[str, Path],
    coverage: Coverage,
    on_progress: Callable[[], None] | None = None,
) -> dict[str, int]:
    """Upload the complete files, record them and refresh the card (partial files are
    already up and recorded)."""
    dataset, repo, revision = target
    if new:
        hub.upload(repo, new, f"Add land-use labels ({config.GENERATION_FP})")
        store.append_jsonl(f"published/{dataset}.jsonl", [{"path": d} for _, d in new])
    return _refresh_card(
        store,
        hub,
        dataset,
        repo=repo,
        revision=revision,
        local={**partial, **{d: src for src, d in new}},
        coverage=coverage,
        on_progress=on_progress,
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


def _resolved_counts(refs: list[SentenceRef], resolved: Lookup) -> tuple[int, int, bool]:
    """(sentences with an answer, sentences that can get one, all answered) without building
    any table: one index lookup per sentence."""
    sendable = [r for r in refs if not r.unsplit]
    known = sum(resolved.get(r.text_sha256) is not None for r in sendable)
    return known, len(sendable), known == len(sendable)


def _worth_publishing(known: int, sendable: int, *, last: int) -> bool:
    """A partial file goes up when never published and answered once, or after it gained
    :data:`PARTIAL_STEP` of its sentences since its last partial upload."""
    first = last == 0 and known >= 1  # every file shows up as soon as it has one answer
    return first or known - last >= max(1, math.ceil(PARTIAL_STEP * sendable))


def _missing_viewers(run: _Run, *, resolved: ResolutionIndex, out: Path) -> list[tuple[Path, str]]:
    """Viewer tables of files that were labelled before the viewer table existed."""
    made = []
    for path in run.files:
        target = f"viewer/{path}"
        if f"labels/{path}" not in run.done or target in run.done:
            continue
        if run.stop():
            break
        local = Path(
            run.hub.download_all(SPECS[run.dataset].source.repo_id, run.revision, [path])[0][0]
        )
        build_viewer(run.dataset, path, local, resolved=resolved, out=out)
        if (out / target).exists():
            made.append((out / target, target))
    return made


def mirror_ledger(dataset: str) -> str:
    return f"published/{dataset}.mirror.jsonl"


def _discard(files: list[tuple[Path, str]]) -> None:
    """Free the Hub-cache copy of mirrored files (symlinks to blobs) once uploaded."""
    for src, _ in files:
        if src.is_symlink():
            with contextlib.suppress(OSError):
                src.resolve().unlink(missing_ok=True)
                src.unlink(missing_ok=True)


def _fetch(hub: Hub, repo: str, revision: str, path: str) -> list[tuple[Path, str]]:
    return hub.download_all(repo, revision, [path])


def _unmirrored(  # noqa: PLR0917
    hub: Hub, store: WorkStore, dataset: str, input_repo: str, revision: str, repo: str
) -> list[str]:
    recorded = {
        r["path"] for r in store.compact_jsonl(mirror_ledger(dataset)) if r["revision"] == revision
    }
    present = recorded or hub.remote_files(repo)
    return [
        p for p in hub.list_files(input_repo, revision) if p not in present and p != "README.md"
    ]


def _mirror(  # noqa: PLR0913
    hub: Hub,
    store: WorkStore,
    dataset: str,
    *,
    input_repo: str,
    revision: str,
    repo: str,
    done: set[str],
    on_progress: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> None:
    """Copy the input files to the output repo, resumably: each uploaded batch is recorded in
    ``published/<dataset>.mirror.jsonl`` (so a restart does not even list the output repo
    once something was recorded), downloads run in parallel and each batch is deleted after
    its commit. A stop request is honoured between batches (the ledger then matches what is
    up) and leaves the mirror marker unwritten, so the next run resumes."""
    marker = f"mirror:{revision}"
    if marker in done:
        return
    wanted = _unmirrored(hub, store, dataset, input_repo, revision, repo)
    with ThreadPoolExecutor(MIRROR_WORKERS) as pool:
        for start in range(0, len(wanted), BATCH):
            if should_stop():
                return
            batch = wanted[start : start + BATCH]
            files = [
                f
                for part in pool.map(lambda p: _fetch(hub, input_repo, revision, p), batch)
                for f in part
            ]
            hub.upload(repo, files, f"Mirror {input_repo}@{revision[:7]}")
            store.append_jsonl(
                mirror_ledger(dataset), [{"path": p, "revision": revision} for p in batch]
            )
            _discard(files)
            if on_progress:
                on_progress()
    store.append_jsonl(f"published/{dataset}.jsonl", [{"path": marker}])


def _decision_counts(out: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in (out / "labels").rglob("*.parquet"):
        counts.update(pq.read_table(path, columns=["decision"]).column("decision").to_pylist())
    return dict(counts)


def _refresh_card(  # noqa: PLR0913
    store: WorkStore,
    hub: Hub,
    dataset: str,
    *,
    repo: str,
    revision: str,
    local: dict[str, Path],
    coverage: Coverage,
    on_progress: Callable[[], None] | None = None,
) -> dict[str, int]:
    """Recount the card's numbers from every published file, then update the card."""
    on_hub = {r["path"] for r in store.read_jsonl(f"published/{dataset}.jsonl")}
    stats = published_stats.totals(
        published_stats.complete(
            store,
            dataset,
            on_hub | set(coverage.partial) | set(local),
            lambda p: local[p] if p in local else hub.open_file(repo, p),
            _map_locator(dataset, revision, hub),
            refresh=set(local),  # uploaded now: a partial record of the same path is stale
            on_progress=on_progress,
        )
    )
    if stats.decisions:
        _card(
            store,
            hub,
            dataset,
            repo=repo,
            revision=revision,
            coverage=coverage,
            stats=stats,
        )
    return stats.decisions


def _map_locator(dataset: str, revision: str, hub: Hub) -> published_stats.Locator | None:
    """Where the rows of a labels file are, for datasets whose input has coordinates."""
    spec = SPECS[dataset]
    if spec.map_locator is None:
        return None
    return spec.map_locator(spec.source.repo_id, hub, revision)


def _world_map(stats: published_stats.PublishedStats, png: Path, dataset: str) -> MapFacts | None:
    """Render the yes-share map and return its figures; ``None`` without coordinates."""
    if not stats.cells:
        return None
    from landuse_filter.adapters import hexmap
    from landuse_filter.domain import geomap

    png.parent.mkdir(parents=True, exist_ok=True)
    hexmap.render(stats.cells, png, title=f"{dataset}-landuse: share of yes per H3 cell")
    return MapFacts(
        cells=len(stats.cells),
        located=stats.located,
        labelled=stats.labelled,
        share=geomap.global_share(stats.cells),
    )


def _card(
    store: WorkStore,
    hub: Hub,
    dataset: str,
    *,
    repo: str,
    revision: str,
    coverage: Coverage,
    stats: published_stats.PublishedStats,
) -> None:
    cfg = config.reference_config()
    png = store.path(f"publish/{dataset}/{MAP_ASSET}")
    world_map = _world_map(stats, png, dataset)
    admitted = store.admitted_gates()
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
            decisions=stats.decisions,
            failures=stats.failures,
            unique_texts=stats.unique_texts,
            gpu_rows=stats.gpus,
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
