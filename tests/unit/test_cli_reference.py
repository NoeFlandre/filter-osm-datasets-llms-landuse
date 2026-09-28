import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_cli_reference_is_up_to_date():
    """docs/cli.md is generated from the CLI; regenerate it when commands change."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "cli_reference.py")],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout
    assert out == (ROOT / "docs" / "cli.md").read_text(encoding="utf-8"), (
        "run: python scripts/cli_reference.py > docs/cli.md"
    )
