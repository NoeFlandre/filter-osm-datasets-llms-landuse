"""Which project-owned paths on a site can be deleted (pure; ``luf g5k clean``).

Only paths under ``luf/`` are ever considered. Never: the token, the live job's code,
the current lockfile's environment, unsynced spool parts, or anything recent.
"""

from collections.abc import Iterable
from dataclasses import dataclass

ROOT = "luf/"
LOG_DAYS = 7


@dataclass(frozen=True, slots=True)
class Entry:
    path: str  # relative to the site home, e.g. "luf/code/<commit>"
    age_days: float


@dataclass(frozen=True, slots=True)
class Keep:
    commits: frozenset[str]  # code used by live jobs (and the current HEAD)
    lock_sha: str  # environment of the current lockfile
    synced_parts: frozenset[str]  # spool parts already safe locally or in the bucket


def _removable(e: Entry, keep: Keep) -> bool:
    name = e.path.removeprefix(ROOT)
    kind, _, rest = name.partition("/")
    if kind == "code":
        return rest not in keep.commits
    if kind == "cache" and rest.startswith("venv-"):
        return not rest.startswith(f"venv-{keep.lock_sha}")
    if kind == "logs":
        return e.age_days > LOG_DAYS
    if kind == "work" and rest.startswith("parts/"):
        return rest.removeprefix("parts/") in keep.synced_parts
    return False


def cleanup_plan(entries: Iterable[Entry], keep: Keep) -> list[str]:
    """Paths safe to delete, sorted; anything outside ``luf/`` is never returned."""
    return sorted(e.path for e in entries if e.path.startswith(ROOT) and _removable(e, keep))
