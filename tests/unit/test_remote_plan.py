from pathlib import Path

from landuse_filter.adapters.readers import WEBSITE
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.remote_plan import PlanInputs, run_plan

INPUTS = Path(__file__).parents[1] / "fixtures" / "inputs"
FILES = ["polygons/a.parquet", "polygons/b.parquet", "polygons/c.parquet"]


INPUTS_PLAN = PlanInputs(
    files=FILES,
    fetch=lambda _: INPUTS / "website.parquet",
    encode=lambda ps: [[len(p)] for p in ps],
    template="S: {}",
    chunk_size=7,
)


def plan(tmp_path, scratch_name, stop_after=None):
    remote = DirRemote(tmp_path / "bucket")
    scratch = WorkStore(tmp_path / scratch_name)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return stop_after is not None and calls["n"] > stop_after

    report = run_plan(
        remote,
        scratch,
        WEBSITE,
        "fp",
        INPUTS_PLAN,
        should_stop=stop,
    )
    return remote, scratch, report


def test_plan_publishes_chunks_and_keeps_scratch_empty(tmp_path):
    remote, scratch, report = plan(tmp_path, "s1")
    assert report["files_done"] == 3
    chunks = remote.ls("chunks/")
    assert chunks
    assert remote.ls(f"plans/{WEBSITE}/fp/chunks.jsonl")
    assert not list(scratch.path("chunks").glob("*.parquet"))  # nothing bulky left behind
    assert remote.ls(f"index/{WEBSITE}.sqlite")


def test_killed_plan_resumes_from_bucket_on_another_node(tmp_path):
    remote, _, first = plan(tmp_path, "s1", stop_after=1)
    assert first["files_done"] < 3
    _, _, second = plan(tmp_path, "s2")  # fresh scratch: resumes from the bucket index
    assert second["files_done"] == 3
    lines = remote.root / f"plans/{WEBSITE}/fp/chunks.jsonl"
    ids = lines.read_text().splitlines()
    assert len(ids) == len(set(ids))


def test_resume_works_when_the_downloaded_index_is_read_only(tmp_path, monkeypatch):
    """Regression (Grenoble job 3123094): a read-only restored index broke resumed planning."""
    remote, _, first = plan(tmp_path, "s1", stop_after=1)
    original_get = remote.get

    def read_only_get(files):
        original_get(files)
        for _, dst in files:
            dst.chmod(0o444)

    monkeypatch.setattr(remote, "get", read_only_get)
    from landuse_filter.application.remote_plan import run_plan

    scratch = WorkStore(tmp_path / "s3")
    report = run_plan(
        remote,
        scratch,
        WEBSITE,
        "fp",
        INPUTS_PLAN,
        should_stop=lambda: False,
    )
    assert report["files_done"] == 3


def plan_line_ids(remote):
    import json

    text = (remote.root / f"plans/{WEBSITE}/fp/chunks.jsonl").read_text()
    return {json.loads(line)["chunk_id"] for line in text.splitlines()}


def test_resumed_plan_keeps_the_plan_lines_of_earlier_nodes(tmp_path):
    """Regression (wiki planning): each resumed job started with an empty scratch plan file
    and overwrote the bucket's, losing every earlier chunk line (1,586 of 26,389 survived)."""
    remote, _, _ = plan(tmp_path, "s1", stop_after=1)
    before = plan_line_ids(remote)
    assert before
    _, _, report = plan(tmp_path, "s2")
    after = plan_line_ids(remote)
    assert before <= after
    assert len(after) == report["chunks_emitted"]


def test_missing_plan_lines_are_rebuilt_from_the_index(tmp_path):
    """Recovery for the loss above: chunks the index assigned but the plan file forgot."""
    remote, _, report = plan(tmp_path, "s1")
    plan_file = remote.root / f"plans/{WEBSITE}/fp/chunks.jsonl"
    first_line = plan_file.read_text().splitlines()[0]
    plan_file.write_text(first_line + "\n")
    plan(tmp_path, "s2")  # nothing left to scan: only the rebuild runs
    assert len(plan_line_ids(remote)) == report["chunks_emitted"]
