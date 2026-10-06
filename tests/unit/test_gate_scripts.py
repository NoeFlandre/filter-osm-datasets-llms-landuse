import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[2] / "scripts"


def test_mutation_gate_actually_runs_its_main(tmp_path):
    """Regression: without a __main__ guard the gate printed nothing and always passed."""
    bad_allowlist = tmp_path / "allow.txt"
    bad_allowlist.write_text("not-a-valid-line-without-reason\n")
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "check_mutants.py"), "--allowlist", str(bad_allowlist)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert done.returncode != 0
    assert "mutation gate" in done.stdout


def test_crap_gate_has_an_entrypoint():
    assert '__name__ == "__main__"' in (SCRIPTS / "crap.py").read_text()


ROOT = Path(__file__).parents[2]
GATED = [
    "src/landuse_filter/domain",
    "src/landuse_filter/application/assignment.py",
    "src/landuse_filter/application/card.py",
    "src/landuse_filter/application/repair.py",
    "src/landuse_filter/application/results.py",
    "src/landuse_filter/application/status.py",
    "src/landuse_filter/application/plan.py",
]


def _quality_scope():
    import importlib.util

    spec = importlib.util.spec_from_file_location("quality_scope", SCRIPTS / "quality_scope.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gated_scope_is_pinned_and_widening_it_is_deliberate():
    assert _quality_scope().gated_paths() == GATED
    assert all((ROOT / path).exists() for path in GATED)


def test_crap_limits_derive_from_the_mutation_scope():
    args = _quality_scope().crap_limit_args(6.0)
    assert args == [a for path in GATED for a in ("--limit", f"{path}=6")]


def test_the_makefile_lists_no_scope_of_its_own():
    makefile = (ROOT / "Makefile").read_text()
    crap = next(line for line in makefile.splitlines() if "scripts/crap.py" in line)
    assert "quality_scope.py" in crap
    assert "$(shell" not in makefile  # $(shell) would drop a failing helper's exit status
    assert crap.split("quality_scope.py")[1].split("&&")[1].strip().startswith("$(RUN)")
    assert "src/landuse_filter" not in makefile
    assert "PURE_APPLICATION" not in makefile


def test_the_scope_script_prints_the_crap_arguments():
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "quality_scope.py"), "--limit", "6"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.split() == _quality_scope().crap_limit_args(6.0)


def test_an_empty_or_missing_scope_fails_the_helper_and_so_the_crap_gate(tmp_path):
    for body in ("[tool.mutmut]\nsource_paths = []\n", "[tool.other]\nx = 1\n"):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(body)
        with pytest.raises(SystemExit, match="no gated scope"):
            _quality_scope().gated_paths(pyproject)
    # The recipe's shape: a failing helper must stop the command that follows it.
    failing = subprocess.run(
        ["bash", "-ec", 'limits="$(false)" && echo ran'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failing.returncode != 0
    assert "ran" not in failing.stdout
