"""Report the CRAP score of every domain function and fail if any exceeds a limit.

CRAP(m) = complexity(m)^2 * (1 - coverage(m))^3 + complexity(m)

Reads `coverage.json` (written by `pytest --cov-report=json`) and radon's cyclomatic
complexity, joining them on the line ranges of each function.
"""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = "src/landuse_filter/domain"
FULL_COVERAGE_PERCENT = 100.0


@dataclass(slots=True)
class CrapAllowlist:
    """Reviewed score exceptions and the entries used during one gate run."""

    entries: dict[str, dict[str, str]]
    used: set[str] = field(default_factory=set)


def crap(complexity: int, coverage: float) -> float:
    return complexity**2 * (1 - coverage) ** 3 + complexity


def radon_blocks(target: Path) -> dict[str, list[dict]]:
    # Runs radon from this interpreter over a repository path; no untrusted input.
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "radon", "cc", "-s", "-j", str(target)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def coverage_by_file(report: Path) -> dict[str, dict[str, set[int]]]:
    payload = json.loads(report.read_text(encoding="utf-8"))
    return {
        _normalise(name): {
            "executed": set(data.get("executed_lines", [])),
            "missing": set(data.get("missing_lines", [])),
        }
        for name, data in payload["files"].items()
    }


def _normalise(name: str | Path) -> str:
    path = Path(name)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _score_key(name: str) -> str:
    """Return a score's repository-relative ``path:function`` allowlist key."""
    source, separator, function = name.rpartition(":")
    return f"{_normalise(source)}:{function}" if separator else name


def load_allowlist(path: Path) -> dict[str, dict[str, str]]:
    """Read explicit CRAP exceptions, each with its reason and exercising tests."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CRAP allowlist must be a JSON object")
    allowlist: dict[str, dict[str, str]] = {}
    for name, details in payload.items():
        if (
            not isinstance(name, str)
            or ":" not in name
            or not isinstance(details, dict)
            or not isinstance(details.get("reason"), str)
            or not details["reason"].strip()
            or not isinstance(details.get("tests"), str)
            or not details["tests"].strip()
        ):
            raise ValueError("each CRAP allowlist entry needs a path:function, reason, and tests")
        allowlist[_score_key(name)] = {
            "reason": details["reason"].strip(),
            "tests": details["tests"].strip(),
        }
    return allowlist


def source_files(target: Path) -> set[str]:
    """Python source files under one target, named the same way as coverage JSON."""
    path = target if target.is_absolute() else PROJECT_ROOT / target
    candidates = [path] if path.is_file() else list(path.rglob("*.py"))
    return {_normalise(candidate) for candidate in candidates}


def missing_coverage_files(target: Path, covered_files: set[str]) -> list[str]:
    """Source modules for a target that the coverage run did not load at all."""
    normalized = {_normalise(filename) for filename in covered_files}
    return sorted(source_files(target) - normalized)


def parse_limit(value: str) -> tuple[str, float]:
    """Parse a ``TARGET=MAX`` CRAP gate argument."""
    target, separator, raw_maximum = value.rpartition("=")
    try:
        maximum = float(raw_maximum)
    except ValueError as exc:
        raise ValueError("expected TARGET=MAX with a finite non-negative MAX") from exc
    if not separator or not target or not math.isfinite(maximum) or maximum < 0:
        raise ValueError("expected TARGET=MAX with a finite non-negative MAX")
    return target, maximum


def coverage_percent(target: Path, report: Path) -> float:
    """Combined line and branch coverage for a target, as reported by coverage.py."""
    payload = json.loads(report.read_text(encoding="utf-8"))
    summaries = {
        _normalise(filename): file_data["summary"]
        for filename, file_data in payload["files"].items()
    }
    filenames = source_files(target)
    absent = sorted(filenames - summaries.keys())
    if absent:
        raise ValueError("source files absent from coverage: " + ", ".join(absent))

    covered = 0
    total = 0
    for filename in filenames:
        summary = summaries[filename]
        covered += summary.get("covered_lines", 0) + summary.get("covered_branches", 0)
        total += summary.get("num_statements", 0) + summary.get("num_branches", 0)
    return FULL_COVERAGE_PERCENT if total == 0 else FULL_COVERAGE_PERCENT * covered / total


def scores(target: Path, report: Path) -> list[tuple[str, int, float, float]]:
    coverage = coverage_by_file(report)
    rows: list[tuple[str, int, float, float]] = []
    for filename, blocks in radon_blocks(target).items():
        lines = coverage.get(_normalise(filename))
        if lines is None:
            continue
        for block in blocks:
            span = range(block["lineno"], block["endline"] + 1)
            executed = len(lines["executed"] & set(span))
            missed = len(lines["missing"] & set(span))
            measurable = executed + missed
            ratio = executed / measurable if measurable else 1.0
            rows.append(
                (
                    f"{filename}:{block['name']}",
                    block["complexity"],
                    ratio,
                    crap(block["complexity"], ratio),
                )
            )
    return sorted(rows, key=lambda row: -row[3])


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        action="append",
        default=[],
        metavar="TARGET=MAX",
        help="Gate a target at a maximum CRAP score; repeat for multiple layers.",
    )
    parser.add_argument("--max", type=float, default=6.0, help="Legacy single-target CRAP limit.")
    parser.add_argument("--target", help="Legacy single target [default: domain].")
    parser.add_argument(
        "--full-coverage",
        action="append",
        default=[],
        metavar="TARGET",
        help="Require 100 percent line and branch coverage for a target.",
    )
    parser.add_argument("--coverage-json", default="coverage.json")
    parser.add_argument(
        "--allowlist", help="Reviewed JSON exceptions for complex, tested functions."
    )
    return parser


def _limits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[tuple[str, float]]:
    try:
        limits = [parse_limit(value) for value in args.limit]
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit and args.target:
        parser.error("use either --limit or --target, not both")
    return limits or [(args.target or DEFAULT_TARGET, args.max)]


def _print_scores(
    target_name: str,
    maximum: float,
    layer_coverage: float,
    rows: list[tuple[str, int, float, float]],
    allowlist: CrapAllowlist,
) -> bool:
    width = max(len(name) for name, *_ in rows)
    print(f"\n{target_name}: {layer_coverage:.2f}% coverage, CRAP limit {maximum:g}")
    print(f"{'function'.ljust(width)}  cc   cov     crap  headroom")
    for name, complexity, ratio, score in rows:
        print(
            f"{name.ljust(width)}  {complexity:<3}  {ratio:5.1%}  "
            f"{score:6.2f}  {maximum - score:8.2f}"
        )
    offenders = []
    for row in rows:
        if row[3] <= maximum:
            continue
        key = _score_key(row[0])
        exception = allowlist.entries.get(key)
        if exception is None:
            offenders.append(row)
            continue
        allowlist.used.add(key)
        print(f"approved exception: {row[0]} ({exception['reason']}; tests: {exception['tests']})")
    if offenders:
        print(f"CRAP above {maximum:g}: " + ", ".join(name for name, *_ in offenders))
        return False
    print(f"all {len(rows)} functions are at or below a CRAP score of {maximum:g}")
    return True


def _gate_target(
    target_name: str,
    maximum: float,
    *,
    full_coverage: bool,
    report: Path,
    allowlist: CrapAllowlist,
) -> bool:
    target = Path(target_name)
    target_path = target if target.is_absolute() else PROJECT_ROOT / target
    if not target_path.exists():
        print(f"target does not exist: {target_name}")
        return False

    covered_files = set(coverage_by_file(report))
    missing = missing_coverage_files(target_path, covered_files)
    if missing:
        print(f"source files absent from coverage for {target_name}: " + ", ".join(missing))
        return False
    try:
        layer_coverage = coverage_percent(target_path, report)
    except ValueError as exc:
        print(f"coverage error for {target_name}: {exc}")
        return False

    passed = True
    if full_coverage and layer_coverage < FULL_COVERAGE_PERCENT:
        print(
            f"coverage for {target_name} is {layer_coverage:.2f}%; "
            f"required {FULL_COVERAGE_PERCENT:.2f}% line and branch coverage"
        )
        passed = False
    rows = scores(target_path, report)
    if not rows:
        print(f"no functions measured in {target_name}; check the coverage run")
        return False
    return _print_scores(target_name, maximum, layer_coverage, rows, allowlist) and passed


def _run_gates(
    limits: list[tuple[str, float]],
    full_coverage_targets: list[str],
    report: Path,
    allowlist: CrapAllowlist,
) -> int:
    results = [
        _gate_target(
            target_name,
            maximum,
            full_coverage=target_name in full_coverage_targets,
            report=report,
            allowlist=allowlist,
        )
        for target_name, maximum in limits
    ]
    stale = sorted(set(allowlist.entries) - allowlist.used)
    if stale:
        print(
            "stale CRAP allowlist entries (no current score above its limit): " + ", ".join(stale)
        )
    return 0 if all(results) and not stale else 1


def main() -> int:
    parser = _argument_parser()
    args = parser.parse_args()
    limits = _limits(args, parser)
    report = PROJECT_ROOT / args.coverage_json
    allowlist: dict[str, dict[str, str]] = {}
    if args.allowlist:
        try:
            allowlist = load_allowlist(PROJECT_ROOT / args.allowlist)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(f"invalid CRAP allowlist: {exc}")
    return _run_gates(limits, args.full_coverage, report, CrapAllowlist(allowlist))


if __name__ == "__main__":
    raise SystemExit(main())
