"""Publishing as a Grid'5000 job: everything bulky stays on node-local scratch."""

from collections.abc import Callable

from landuse_filter.adapters.hub import Hub
from landuse_filter.adapters.remote import Remote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.progress import NULL, Progress
from landuse_filter.application.publish import PublishReport, publish, refresh_card_only
from landuse_filter.application.publish_loop import status_path
from landuse_filter.application.remote_plan import restore_index
from landuse_filter.application.resolution_sync import fetch_parts, restore_resolution
from landuse_filter.application.sync import fetch
from landuse_filter.domain.publish_loop import PublishStatus


def run_publish(  # noqa: PLR0913
    remote: Remote,
    scratch: WorkStore,
    dataset: str,
    revision: str,
    fp: str,
    *,
    hub: Hub | None = None,
    should_stop: Callable[[], str | None] = lambda: None,
    progress: Progress = NULL,
) -> PublishReport:
    """Restore the planner index and the resolution snapshot (reading only the result parts it
    has not seen, ADR-0026), then publish incrementally.

    ``published/<dataset>.jsonl`` is round-tripped through the bucket so a later job
    only uploads what is new.
    """
    progress.event("publish_start", dataset=dataset, revision=revision[:10])
    if not restore_index(remote, scratch, dataset, progress):
        raise FileNotFoundError(f"no planner index for {dataset} in the bucket; run planning first")
    resolved = restore_resolution(
        remote, scratch, fp, should_stop=lambda: should_stop() is not None, progress=progress
    )
    progress.event("fetch_ledgers")
    fetch(remote, scratch, [g for g in remote.ls("gates/admission/") if g.endswith(".json")])
    ledgers = [
        f"published/{dataset}.jsonl",
        f"published/{dataset}.stats.jsonl",
        f"published/{dataset}.partial.jsonl",
        f"published/{dataset}.mirror.jsonl",
    ]
    fetch(remote, scratch, [p for p in ledgers if p in remote.ls("published/")])
    card_marker = f"published/{dataset}.card.sha256"
    fetch(remote, scratch, [card_marker] if card_marker in remote.ls("published/") else [])

    def save() -> None:
        remote.put([(scratch.path(p), p) for p in [*ledgers, card_marker] if scratch.exists(p)])

    try:
        report = publish(
            scratch,
            dataset,
            revision,
            on_progress=save,
            hub=hub,
            should_stop=should_stop,
            resolved=resolved,
            fetch_parts=lambda keys: fetch_parts(remote, scratch, fp, keys),
            progress=progress,
        )
    except BaseException as error:
        progress.finish(f"error: {type(error).__name__}: {error}"[:300])
        raise
    finally:
        save()  # whatever happened, the ledgers and the card marker leave the node
    _save_status(remote, scratch, revision, report)
    progress.finish(report.stopped, labelled=report.labelled_files, total=report.total_files)
    return report


def _save_status(remote: Remote, scratch: WorkStore, revision: str, report: PublishReport) -> None:
    """The machine-readable end-of-run marker ``luf g5k publish-loop`` waits for."""
    path = status_path(report.dataset)
    status = PublishStatus(
        report.dataset,
        revision,
        report.total_files,
        report.labelled_files,
        report.partial_files,
        report.mirrored,
        report.unscanned,
        report.stopped,
    )
    scratch.write_json(path, status.to_json())
    remote.put([(scratch.path(path), path)])


def reset_card_cache(remote: Remote, scratch: WorkStore, dataset: str) -> None:
    """Forget the generation counts and the card hash so the next publish recounts the card."""
    stats = f"published/{dataset}.stats.jsonl"
    if stats in remote.ls("published/"):
        fetch(remote, scratch, [stats])
        keep = [r for r in scratch.read_jsonl(stats) if not r["path"].startswith("generations/")]
        scratch.path(stats).unlink()
        scratch.append_jsonl(stats, keep)
        remote.put([(scratch.path(stats), stats)])
    marker = f"published/{dataset}.card.sha256"
    scratch.path(marker).parent.mkdir(parents=True, exist_ok=True)
    scratch.path(marker).write_text("reset\n")
    remote.put([(scratch.path(marker), marker)])


def run_card_only(
    remote: Remote,
    scratch: WorkStore,
    dataset: str,
    revision: str,
    *,
    hub: Hub | None = None,
    progress: Progress = NULL,
) -> PublishReport:
    """Refresh the dataset card from the bucket's ledgers and gates alone (minutes, not hours).

    Skips the planner index, the result parts and the resolution index; counts any missing
    per-file stats from the Hub and saves the ledgers back as it goes."""
    progress.event("card_job_start", dataset=dataset)
    fetch(remote, scratch, [g for g in remote.ls("gates/admission/") if g.endswith(".json")])
    ledgers = [
        f"published/{dataset}{suffix}"
        for suffix in (".jsonl", ".stats.jsonl", ".partial.jsonl", ".mirror.jsonl", ".card.sha256")
    ]
    on_bucket = remote.ls("published/")
    fetch(remote, scratch, [p for p in ledgers if p in on_bucket])

    def save() -> None:
        remote.put([(scratch.path(p), p) for p in ledgers if scratch.exists(p)])

    report = refresh_card_only(
        scratch, dataset, revision, on_progress=save, hub=hub, progress=progress
    )
    save()
    progress.finish(None, labelled=report.labelled_files, total=report.total_files)
    return report
