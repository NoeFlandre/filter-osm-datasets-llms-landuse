from pathlib import Path

from landuse_filter.adapters.readers import WEBSITE
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.remote_plan import run_plan

INPUTS = Path(__file__).parents[1] / "fixtures" / "inputs"
FILES = ["polygons/a.parquet", "polygons/b.parquet", "polygons/c.parquet"]


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
        files=FILES,
        fetch=lambda _: INPUTS / "website.parquet",
        encode=lambda p: [len(p)],
        template="S: {}",
        chunk_size=7,
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
