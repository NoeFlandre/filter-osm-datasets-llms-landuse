import json
import os
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "ops" / "gpu_watch.sh"

FAKE_SSH = """#!/bin/sh
# usage: ssh -o a -o b <site> <command>
site=$5
case "$site" in
  nancy) echo "1001 u x x x R 0:10 luf-a"; echo "1002 u x x x W 0:00 luf-b"; exit 0 ;;
  lille) exit 255 ;;   # ssh itself failed
  *) exit 1 ;;         # reachable, grep found no luf job
esac
"""

FAKE_LUF = """#!/bin/sh
echo "osm-polygon-description-tag: {'chunks_complete': 3, 'chunks': 10}"
echo "osm-polygon-website-tag: {'chunks_complete': 0, 'chunks': 4}"
"""


def _stub(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_watch(tmp_path: Path, *, extra_assignments=(), log=""):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir / "ssh", FAKE_SSH)
    _stub(bin_dir / "luf", FAKE_LUF)
    work = tmp_path / "work"
    (work / "assignments").mkdir(parents=True)
    a = {"site": "nancy", "job_id": "1001", "gpu": "l40s", "fp": "abc"}
    (work / "assignments" / "a.json").write_text(json.dumps(a))
    (work / "assignments" / "broken.json").write_text("{not json")
    if log:
        (work / "controller-production.log").write_text(log)
    env = {
        **os.environ,
        "LUF_WORK": str(work),
        "GPU_WATCH_ONCE": "1",
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    # Run a copy so that the script's project .venv (with the real luf) is not on its PATH.
    script = tmp_path / "scripts" / "ops" / "gpu_watch.sh"
    script.parent.mkdir(parents=True)
    shutil.copy(SCRIPT, script)
    return subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, timeout=120, check=False
    )


def test_one_cycle_prints_a_summary_line_and_exits(tmp_path):
    done = run_watch(tmp_path)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.strip().splitlines()
    assert len(lines) == 1
    assert re.fullmatch(
        r"\d\d:\d\d GPUS running=1 waiting=1 prod:l40s=1 UNREACHABLE=lille "
        r"\| chunks desc=3/10 webs=0/4 \| ingest \d+m ago \| admission pending: none ?",
        lines[0],
    ), lines[0]


def test_a_failing_ssh_marks_the_site_unreachable_not_empty(tmp_path):
    line = run_watch(tmp_path).stdout
    assert "UNREACHABLE=lille" in line
    assert "grenoble" not in line  # a site with no jobs is simply empty


def test_the_last_controller_error_is_appended(tmp_path):
    done = run_watch(tmp_path, log="ok\nxx cycle failed (boom)\n")
    assert "| error: xx cycle failed (boom)" in done.stdout


def _with_index(tmp_path: Path, age_seconds: float) -> Path:
    progress = tmp_path / "work" / "index" / "progress-abc.sqlite"
    progress.parent.mkdir(parents=True)
    progress.write_bytes(b"")
    stamp = time.time() - age_seconds
    os.utime(progress, (stamp, stamp))
    return progress


def test_a_stale_progress_index_is_reported_in_minutes(tmp_path):
    """A controller that keeps submitting but stopped ingesting results (wedged for an hour on
    2026-10-01) must be visible: the line carries the minutes since the progress index changed."""
    _with_index(tmp_path, 3 * 3600)
    assert re.search(r"ingest 1[78]\dm ago", run_watch(tmp_path).stdout)


def test_a_fresh_progress_index_reads_zero_minutes(tmp_path):
    _with_index(tmp_path, 0)
    assert "ingest 0m ago" in run_watch(tmp_path).stdout
