"""Importing the CLI reads no settings and no environment; each command reads them when it runs.

Regression for issue #217: the operator settings (luf.toml, LUF_CONFIG, LUF_<FIELD>) and
LUF_WORK used to be read at import time, so a changed environment was ignored and a broken
config broke every command, including ``luf version``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from landuse_filter.application import controller as ctl_mod
from landuse_filter.cli import app
from landuse_filter.cli import g5k as g5k_cli

runner = CliRunner()

# Count every settings load and every --work default resolution while the CLI is imported.
PROBE = """
import landuse_filter.adapters.settings_file as settings_file
import landuse_filter.config as config

calls = []
real_load, real_work_dir = settings_file.load, config.work_dir
settings_file.load = lambda *a, **k: calls.append("load") or real_load(*a, **k)
config.work_dir = lambda: calls.append("work_dir") or real_work_dir()

import landuse_filter.cli  # noqa: F401

print(",".join(calls) or "none")
"""


def _probe(extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LUF_")}
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=False
    )


def test_importing_the_cli_reads_no_settings_or_work_dir(tmp_path: Path):
    # A config that does not exist would make any import-time read fail the import.
    result = _probe(
        {
            "LUF_CONFIG": str(tmp_path / "missing.toml"),
            "LUF_WORK": "/from/env",
            "LUF_BUCKET": "env-bucket",
        }
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "none"


def test_commands_that_need_no_settings_run_with_a_broken_config(tmp_path: Path):
    # Fresh interpreter: the CLI must be imported under the broken config, not after it.
    script = (
        "from typer.testing import CliRunner\n"
        "from landuse_filter.cli import app\n"
        "runner = CliRunner()\n"
        "assert runner.invoke(app, ['version']).exit_code == 0\n"
        "assert runner.invoke(app, ['g5k', 'run', '--help']).exit_code == 0\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("LUF_")}
    env["LUF_CONFIG"] = str(tmp_path / "missing.toml")
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.fixture
def loop(monkeypatch):
    seen = {}
    monkeypatch.setattr(g5k_cli, "_controller", lambda work, settings: (work, settings))
    monkeypatch.setattr(ctl_mod, "run_loop", lambda ctl, **kw: seen.update(ctl=ctl, **kw))
    return seen


def test_environment_changed_after_import_is_honoured_by_g5k_run(monkeypatch, tmp_path, loop):
    monkeypatch.setenv("LUF_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("LUF_BUCKET", "env-bucket")
    result = runner.invoke(app, ["g5k", "run", "--datasets", "benchmark", "--once"])
    assert result.exit_code == 0, result.output
    work, settings = loop["ctl"]
    assert work == tmp_path / "work"
    assert settings.bucket == "env-bucket"
    assert loop["once"] is True
