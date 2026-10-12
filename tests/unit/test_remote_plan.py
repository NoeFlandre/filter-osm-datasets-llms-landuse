import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from landuse_filter.adapters.readers import SOURCES, WEBSITE
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import remote_plan
from landuse_filter.application.job_progress import NULL, Progress
from landuse_filter.application.remote_plan import PlanInputs, run_plan, run_replan

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


def locator_of(calls: list[str]):
    """A replan locator that puts every text of a file in the cell named after the file."""

    def locate(dataset, path, local):
        calls.append(path)
        return [
            (ref.text_sha256, f"cell-{path}")
            for ref in SOURCES[dataset].read(local, path)
            if not ref.unsplit
        ]

    return locate


def replan(tmp_path, scratch_name, *, locate, should_stop=lambda: False, progress=NULL):
    return run_replan(
        DirRemote(tmp_path / "bucket"),
        WorkStore(tmp_path / scratch_name),
        WEBSITE,
        "fp",
        INPUTS_PLAN,
        locate=locate,
        should_stop=should_stop,
        progress=progress,
    )


def planned_texts(remote) -> list[str]:
    """Every text the plan lines of the bucket cover, read from the chunk files."""
    lines = (remote.root / f"plans/{WEBSITE}/fp/chunks.jsonl").read_text().splitlines()
    texts: list[str] = []
    for line in lines:
        chunk = remote.root / f"chunks/{json.loads(line)['chunk_id']}.parquet"
        texts += pq.read_table(chunk, columns=["text_sha256"]).column("text_sha256").to_pylist()
    return texts


def test_a_stopped_replan_resumes_without_locating_a_file_again(tmp_path):
    plan(tmp_path, "s1")
    located: list[str] = []
    stopped = replan(tmp_path, "r1", locate=locator_of(located), should_stop=lambda: bool(located))
    assert located == [FILES[0]]  # the stop came before the second file
    assert stopped["stopped"] == "stop requested"
    assert stopped["released_chunks"] > 0
    again: list[str] = []
    resumed = replan(tmp_path, "r2", locate=locator_of(again))
    assert again == FILES[1:]  # the located file is not read a second time
    assert resumed["stopped"] is None
    assert resumed["released_chunks"] == stopped["released_chunks"]  # released once, in total
    marker = (tmp_path / "bucket" / f"plans/{WEBSITE}/fp/replan.json").read_text()
    assert json.loads(marker)["stopped"] is None


def test_a_replan_killed_after_a_checkpoint_resumes_without_releasing_again(tmp_path, monkeypatch):
    class JobKilledError(Exception):
        """Stands in for a job killed at its walltime."""

    monkeypatch.setattr(remote_plan, "CHECKPOINT_SECONDS", -1.0)  # checkpoint at every chance
    plan(tmp_path, "s1")
    remote = DirRemote(tmp_path / "bucket")
    before = plan_line_ids(remote)
    texts_before = sorted(planned_texts(remote))

    def killed_after_a_batch():
        if plan_line_ids(remote):  # the first batch of the new order is in the plan
            raise JobKilledError
        return False

    with pytest.raises(JobKilledError):
        replan(tmp_path, "r1", locate=locator_of([]), should_stop=killed_after_a_batch)
    emitted = plan_line_ids(remote)
    assert emitted
    resumed = replan(tmp_path, "r2", locate=locator_of([]))
    assert resumed["released_chunks"] == len(before)  # the release is the one of the first run
    assert emitted <= plan_line_ids(remote)  # the chunks emitted before the kill are kept
    assert sorted(planned_texts(remote)) == texts_before  # every text is planned exactly once


class Ticking:
    """A clock that moves on every read, so each progress line is due."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 61.0
        return self.now


def test_replan_reports_each_phase_and_batch_and_records_how_it_ended(tmp_path):
    plan(tmp_path, "s1")
    lines: list[str] = []
    progress = Progress(dataset=WEBSITE, out=lines.append, clock=Ticking())
    report = replan(tmp_path, "r1", locate=locator_of([]), progress=progress)
    phases = [line.split()[2] for line in lines]
    assert [p for p in phases if p.endswith(("_start", "_done"))] == [
        "replan_start",
        "restore_index_start",
        "restore_index_done",
        "release_done",
        "locate_start",
        "locate_done",
        "emit_start",
    ]
    assert any(" emit chunks=" in line and "chunks_per_min=" in line for line in lines)
    assert lines[-1].split()[2:4] == ["finished", "stop_reason=none"]
    written = json.loads((tmp_path / "bucket" / f"plans/{WEBSITE}/fp/replan.json").read_text())
    assert written == report
