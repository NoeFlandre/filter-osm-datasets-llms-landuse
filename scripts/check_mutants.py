"""Fail when the named mutation results change outside a reviewed allowlist."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = PROJECT_ROOT / "mutation-allowlist.txt"
STATUSES = {
    "killed",
    "survived",
    "no tests",
    "timeout",
    "suspicious",
    "skipped",
    "caught by type check",
    "check was interrupted by user",
    "not checked",
    "segfault",
}


def parse_results(output: str) -> dict[str, str]:
    """Parse the stable per-mutant rows printed by ``mutmut results --all``."""
    results: dict[str, str] = {}
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ValueError("mutmut results were empty or unparsable")
    for line in lines:
        mutant_id, separator, status = line.rpartition(": ")
        if not separator or not mutant_id or status not in STATUSES:
            raise ValueError(f"mutmut results were empty or unparsable: {line!r}")
        if mutant_id in results:
            raise ValueError(f"mutmut results repeat mutant id {mutant_id!r}")
        results[mutant_id] = status
    if not any(status == "killed" for status in results.values()):
        raise ValueError("mutmut killed no mutants; the run did not test anything")
    return results


def parse_allowlist(content: str) -> dict[str, str]:
    """Read one exact mutant id and one human-readable equivalence reason per line."""
    entries: dict[str, str] = {}
    for line_number, line in enumerate(content.splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        mutant_id, separator, reason = value.partition(":")
        mutant_id, reason = mutant_id.strip(), reason.strip()
        if not separator or not mutant_id or not reason:
            raise ValueError(f"allowlist line {line_number} needs a mutant id and reason")
        if mutant_id in entries:
            raise ValueError(f"allowlist repeats mutant id {mutant_id!r}")
        entries[mutant_id] = reason
    return entries


def validate_results(results: dict[str, str], allowlist: dict[str, str]) -> list[str]:
    """Check survivor identities, stale entries, and every unresolved mutant status."""
    survivors = {mutant_id for mutant_id, status in results.items() if status == "survived"}
    allowed = set(allowlist)
    errors = [f"unreviewed survivor: {mutant_id}" for mutant_id in sorted(survivors - allowed)]
    errors.extend(
        f"stale allowlist entry: {mutant_id}" for mutant_id in sorted(allowed - survivors)
    )
    errors.extend(
        f"unresolved mutant status {status}: {mutant_id}"
        for mutant_id, status in sorted(results.items())
        if status not in {"killed", "survived"}
    )
    return errors


def read_results(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Read full mutmut state and refuse missing, crashed, or empty runs."""
    completed = subprocess.run(
        [sys.executable, "-m", "mutmut", "results", "--all", "true"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"mutmut results failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return parse_results(completed.stdout + completed.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    args = parser.parse_args()

    try:
        results = read_results()
        allowlist = parse_allowlist(args.allowlist.read_text(encoding="utf-8"))
        errors = validate_results(results, allowlist)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"mutation gate error: {exc}")
        return 1

    counts: dict[str, int] = {}
    for status in results.values():
        counts[status] = counts.get(status, 0) + 1
    print(
        "mutmut status counts: "
        + ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    )
    for mutant_id, reason in sorted(allowlist.items()):
        print(f"reviewed equivalent: {mutant_id} — {reason}")
    for error in errors:
        print(error)
    if errors:
        print(f"\nmutation gate failed with {len(errors)} issue(s)")
        return 1
    print(f"\nmutation gate passed: {len(results)} exact mutant id(s) checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
