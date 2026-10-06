"""The gated scope, read once from ``[tool.mutmut] source_paths`` in pyproject.toml.

That list is the single source of truth for what mutation testing and CRAP cover: mutmut
reads it directly, and the Makefile's ``crap`` target gets its ``--limit`` arguments from here.
"""

import argparse
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def gated_paths(pyproject: Path = PROJECT_ROOT / "pyproject.toml") -> list[str]:
    """Repository-relative source paths under the gates (directories keep no trailing slash)."""
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return [p.rstrip("/") for p in config["tool"]["mutmut"]["source_paths"]]


def crap_limit_args(limit: float, pyproject: Path = PROJECT_ROOT / "pyproject.toml") -> list[str]:
    return [arg for path in gated_paths(pyproject) for arg in ("--limit", f"{path}={limit:g}")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=float, default=6.0, help="CRAP limit per gated path")
    print(" ".join(crap_limit_args(parser.parse_args().limit)))


if __name__ == "__main__":
    main()
