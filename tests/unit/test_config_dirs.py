from pathlib import Path

from landuse_filter import config


def test_scratch_dir_default_and_env(monkeypatch):
    monkeypatch.delenv("LUF_SCRATCH", raising=False)
    assert config.scratch_dir() == Path("/tmp/luf-scratch")  # noqa: S108
    monkeypatch.setenv("LUF_SCRATCH", "/x/y")
    assert config.scratch_dir() == Path("/x/y")


def test_work_dir_default_and_env(monkeypatch):
    monkeypatch.delenv("LUF_WORK", raising=False)
    assert config.work_dir() == Path("work")
    monkeypatch.setenv("LUF_WORK", "/w")
    assert config.work_dir() == Path("/w")
