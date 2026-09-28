from pathlib import Path

HOOK = Path(__file__).parents[2] / "scripts" / "hooks" / "pre-commit"


def test_pre_commit_hook_checks_format_and_lint_on_staged_python():
    text = HOOK.read_text()
    assert "ruff format --check" in text
    assert "ruff check" in text
    assert "--diff-filter=ACMR" in text
    assert HOOK.stat().st_mode & 0o111
