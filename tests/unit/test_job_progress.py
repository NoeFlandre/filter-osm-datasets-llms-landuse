import json

from typer.testing import CliRunner

from landuse_filter import config
from landuse_filter.adapters.readers import WEBSITE
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import remote_publish
from landuse_filter.application.job_progress import (
    Progress,
    bucket_progress,
    progress_path,
    read_progress,
)
from landuse_filter.application.plan import Planner
from tests.unit.test_publish import INPUTS, fake_hub, generate_all


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def make(clock, lines, puts):
    return Progress(
        dataset="d", job_id="42", out=lines.append, put=puts.append, clock=clock, wall=lambda: 1000
    )


def test_events_print_every_phase_and_put_at_most_once_a_minute():
    clock, lines, puts = Clock(), [], []
    p = make(clock, lines, puts)
    p.event("a", bytes=5)
    clock.now = 10
    p.event("b")
    clock.now = 59
    p.event("c")
    clock.now = 61
    p.event("d", n=1)
    assert lines == ["luf: [0s] a bytes=5", "luf: [10s] b", "luf: [59s] c", "luf: [61s] d n=1"]
    assert [s["phase"] for s in puts] == ["a", "d"]
    assert puts[1] == {
        "dataset": "d",
        "job_id": "42",
        "phase": "d",
        "counters": {"n": 1},
        "elapsed_seconds": 61,
        "updated_epoch": 1000,
    }


def test_tick_is_throttled_and_finish_always_uploads():
    clock, lines, puts = Clock(), [], []
    p = make(clock, lines, puts)
    p.tick("read", n=1)
    clock.now = 30
    p.tick("read", n=2)
    clock.now = 60
    p.tick("read", n=3)
    assert [s["counters"]["n"] for s in puts] == [1, 3]
    assert len(lines) == 2
    clock.now = 61
    p.finish("deadline", files=7)
    assert puts[-1]["phase"] == "finished"
    assert puts[-1]["counters"] == {"stop_reason": "deadline", "files": 7}
    assert lines[-1] == "luf: [61s] finished stop_reason=deadline files=7"


def test_a_failing_put_never_fails_the_job():
    lines = []

    def boom(_state):
        raise OSError("429")

    Progress(out=lines.append, put=boom).event("x")
    assert any("progress marker not saved" in line for line in lines)


def test_bucket_marker_round_trips_with_one_get(tmp_path):
    remote = DirRemote(tmp_path / "b")
    assert read_progress(remote, "d") is None
    bucket_progress(remote, "d", "7").event("restore_index_done", bytes=3)
    state = read_progress(remote, "d")
    assert state["phase"] == "restore_index_done"
    assert state["job_id"] == "7"
    assert json.loads((tmp_path / "b" / progress_path("d")).read_text())["counters"] == {"bytes": 3}


def test_publish_job_reports_its_phases_in_order(tmp_path):
    hub = fake_hub(tmp_path)
    planning = WorkStore(tmp_path / "planning-node")
    planner = Planner(planning, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")
    planner.db.commit()
    generate_all(planning)
    remote = DirRemote(tmp_path / "bucket")
    remote.put([(planning.path(f"index/{WEBSITE}.sqlite"), f"index/{WEBSITE}.sqlite")])
    for p in planning.part_paths(config.GENERATION_FP):
        remote.put([(p, str(p.relative_to(planning.root)))])
    lines: list[str] = []
    progress = Progress(dataset=WEBSITE, out=lines.append)
    remote_publish.run_publish(
        remote,
        WorkStore(tmp_path / "node"),
        WEBSITE,
        "rev",
        config.GENERATION_FP,
        hub=hub,
        progress=progress,
    )
    phases = [line.split()[2] for line in lines]
    order = [
        "publish_start",
        "restore_index_start",
        "restore_index_done",
        "parts_listed",
        "resolution_ready",
        "build_start",
        "final_upload",
        "finished",
    ]
    positions = [phases.index(p) for p in order]
    assert positions == sorted(positions)
    assert "bytes=" in next(line for line in lines if "restore_index_done" in line)
    commits = [line for line in lines if "hub_commit" in line]
    assert any("kind=complete files=" in line for line in commits)


def test_publish_status_command_prints_the_marker(tmp_path, monkeypatch):
    from landuse_filter.cli import app

    remote = DirRemote(tmp_path / "b")
    bucket_progress(remote, "d", "9").event("parts_read", read=3, total=10)
    monkeypatch.setattr("landuse_filter.adapters.remote.BucketRemote", lambda _bucket: remote)
    result = CliRunner().invoke(app, ["g5k", "publish-status", "--dataset", "d"])
    assert result.exit_code == 0
    assert "job 9" in result.output
    assert "phase parts_read" in result.output
    assert "read=3 total=10" in result.output
    missing = CliRunner().invoke(app, ["g5k", "publish-status", "--dataset", "zz"])
    assert missing.exit_code == 1
