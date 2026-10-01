"""Validated operational settings (pure). Loaded from ``luf.toml`` by an adapter."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from landuse_filter.domain.policy_check import MODES, PER_JOB

T = TypeVar("T")

SCHEMA_VERSION = 1


class SettingsError(ValueError):
    """``luf.toml`` is missing a field, has a wrong type, or a wrong schema version."""


@dataclass(frozen=True, slots=True)
class OpsSettings:
    namespace: str
    bucket: str
    sites: tuple[str, ...]
    walltime_minutes: int
    night_walltime_minutes: int
    night_fallback_walltime_minutes: int
    max_jobs: int
    max_jobs_per_site: int
    interval_seconds: int
    cuda_module: str
    day_walltime_minutes: int | None = None  # None: same as walltime_minutes
    day_long_max_failures: int = 3
    chunk_overflow: float = 1.2  # work assigned per job = capacity x this (ADR-0018)
    policy_check: str = PER_JOB  # usagepolicycheck cadence (ADR-0019)
    submit_workers: int = 1  # sites submitted to in parallel (ADR-0020)

    def output_repo(self, dataset: str) -> str:
        return f"{self.namespace}/{dataset}-landuse"


INT_FIELDS = (
    "walltime_minutes",
    "night_walltime_minutes",
    "night_fallback_walltime_minutes",
    "max_jobs",
    "max_jobs_per_site",
    "interval_seconds",
)


def _need(table: Mapping[str, Any], key: str, kind: type[T]) -> T:
    value = table.get(key)
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is int):
        raise SettingsError(f"luf.toml: {key!r} must be {kind.__name__}")
    return value


def _table(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    table = raw.get(name)
    if not isinstance(table, Mapping):
        raise SettingsError(f"luf.toml: missing [{name}] table")
    return table


def _sites(g5k: Mapping[str, Any]) -> tuple[str, ...]:
    sites = _need(g5k, "sites", list)
    if not sites or not all(isinstance(s, str) and s for s in sites):
        raise SettingsError("luf.toml: 'sites' must be a non-empty list of site names")
    return tuple(sites)


OPTIONAL_INT_FIELDS = ("day_walltime_minutes", "day_long_max_failures", "submit_workers")


def _positive(g5k: Mapping[str, Any], key: str) -> int:
    value = _need(g5k, key, int)
    if value < 1:
        raise SettingsError(f"luf.toml: {key!r} must be positive")
    return value


def _optional(g5k: Mapping[str, Any], key: str, default: int | None) -> int | None:
    return _positive(g5k, key) if key in g5k else default


def _overflow(g5k: Mapping[str, Any]) -> float:
    if "chunk_overflow" not in g5k:
        return 1.2
    value = g5k["chunk_overflow"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError("luf.toml: 'chunk_overflow' must be float")
    if value < 1.0:
        raise SettingsError("luf.toml: 'chunk_overflow' must be at least 1.0")
    return float(value)


def _policy_check(g5k: Mapping[str, Any]) -> str:
    value = g5k.get("policy_check", PER_JOB)
    if value not in MODES:
        raise SettingsError(f"luf.toml: 'policy_check' must be one of {', '.join(MODES)}")
    return value


def parse_settings(raw: Mapping[str, Any]) -> OpsSettings:
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise SettingsError(f"luf.toml: schema_version must be {SCHEMA_VERSION}")
    hub, g5k = _table(raw, "hub"), _table(raw, "grid5000")
    return OpsSettings(
        day_walltime_minutes=_optional(g5k, "day_walltime_minutes", None),
        day_long_max_failures=_positive(g5k, "day_long_max_failures")
        if "day_long_max_failures" in g5k
        else 3,
        chunk_overflow=_overflow(g5k),
        policy_check=_policy_check(g5k),
        submit_workers=_positive(g5k, "submit_workers") if "submit_workers" in g5k else 1,
        namespace=_need(hub, "namespace", str),
        bucket=_need(hub, "bucket", str),
        sites=_sites(g5k),
        cuda_module=_need(g5k, "cuda_module", str),
        **{key: _positive(g5k, key) for key in INT_FIELDS},
    )
