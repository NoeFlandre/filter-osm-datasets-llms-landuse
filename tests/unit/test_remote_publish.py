from pathlib import Path

import pytest

from landuse_filter import config
from landuse_filter.adapters.readers import WEBSITE
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import remote_publish
from landuse_filter.application.plan import Planner
from tests.unit.test_publish import fake_hub, generate_all

INPUTS = Path(__file__).parents[1] / "fixtures" / "inputs"


def test_publish_job_restores_index_and_parts_from_the_bucket(tmp_path, monkeypatch):
    uploads = fake_hub(monkeypatch, tmp_path)
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
    report = remote_publish.run_publish(
        remote, WorkStore(tmp_path / "publish-node"), WEBSITE, "rev", config.GENERATION_FP
    )
    assert report.labelled_files == 1
    assert "labels/polygons/a.parquet" in [p for batch in uploads for p in batch]
    assert remote.ls(f"published/{WEBSITE}.jsonl")


def test_publish_job_requires_planning(tmp_path):
    with pytest.raises(FileNotFoundError, match="planning"):
        remote_publish.run_publish(
            DirRemote(tmp_path / "b"), WorkStore(tmp_path / "s"), WEBSITE, "rev", "fp"
        )
