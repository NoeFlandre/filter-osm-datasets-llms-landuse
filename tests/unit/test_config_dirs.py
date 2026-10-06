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


def _cli_work_default(env_value):
    """The CLI's --work default as resolved at import time in a fresh interpreter."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "LUF_WORK"}
    if env_value is not None:
        env["LUF_WORK"] = env_value
    code = "from landuse_filter.cli import WORK; print(WORK.default)"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def test_cli_work_option_honours_luf_work_env():
    assert _cli_work_default("/from/env") == "/from/env"


def test_cli_work_option_defaults_to_work_without_env():
    assert _cli_work_default(None) == "work"
