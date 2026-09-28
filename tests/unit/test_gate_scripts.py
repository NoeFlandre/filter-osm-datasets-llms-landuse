import subprocess
import sys
from pathlib import Path

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
