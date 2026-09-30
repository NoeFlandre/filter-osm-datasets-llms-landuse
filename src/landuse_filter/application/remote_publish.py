"""Publishing as a Grid'5000 job: everything bulky stays on node-local scratch."""

from landuse_filter.adapters.hub import Hub
from landuse_filter.adapters.remote import Remote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.publish import PublishReport, publish
from landuse_filter.application.remote_plan import restore_index
from landuse_filter.application.sync import fetch


def run_publish(
    remote: Remote,
    scratch: WorkStore,
    dataset: str,
    revision: str,
    fp: str,
    *,
    hub: Hub | None = None,
) -> PublishReport:
    """Restore the planner index and all result parts, then publish incrementally.

    ``published/<dataset>.jsonl`` is round-tripped through the bucket so a later job
    only uploads what is new.
    """
    if not restore_index(remote, scratch, dataset):
        raise FileNotFoundError(f"no planner index for {dataset} in the bucket; run planning first")
    fetch(remote, scratch, [p for p in remote.ls(f"parts/{fp}/") if p.endswith(".parquet")])
    fetch(remote, scratch, [g for g in remote.ls("gates/admission/") if g.endswith(".json")])
    ledgers = [
        f"published/{dataset}.jsonl",
        f"published/{dataset}.stats.jsonl",
        f"published/{dataset}.partial.jsonl",
    ]
    fetch(remote, scratch, [p for p in ledgers if p in remote.ls("published/")])
    card_marker = f"published/{dataset}.card.sha256"
    fetch(remote, scratch, [card_marker] if card_marker in remote.ls("published/") else [])

    def save() -> None:
        remote.put([(scratch.path(p), p) for p in [*ledgers, card_marker] if scratch.exists(p)])

    report = publish(scratch, dataset, revision, on_progress=save, hub=hub)
    save()
    return report


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
