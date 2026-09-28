"""Publishing as a Grid'5000 job: everything bulky stays on node-local scratch."""

from landuse_filter.adapters.remote import Remote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.publish import PublishReport, publish
from landuse_filter.application.remote_plan import restore_index
from landuse_filter.application.sync import fetch


def run_publish(
    remote: Remote, scratch: WorkStore, dataset: str, revision: str, fp: str
) -> PublishReport:
    """Restore the planner index and all result parts, then publish incrementally.

    ``published/<dataset>.jsonl`` is round-tripped through the bucket so a later job
    only uploads what is new.
    """
    if not restore_index(remote, scratch, dataset):
        raise FileNotFoundError(f"no planner index for {dataset} in the bucket; run planning first")
    fetch(remote, scratch, [p for p in remote.ls(f"parts/{fp}/") if p.endswith(".parquet")])
    ledger = f"published/{dataset}.jsonl"
    if ledger in remote.ls("published/"):
        fetch(remote, scratch, [ledger])
    report = publish(scratch, dataset, revision)
    if scratch.exists(ledger):
        remote.put([(scratch.path(ledger), ledger)])
    return report
