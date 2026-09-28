"""Validated operational settings (pure). Loaded from ``luf.toml`` by an adapter."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

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
    max_jobs: int
    max_jobs_per_site: int
    interval_seconds: int
    cuda_module: str

    def output_repo(self, dataset: str) -> str:
        return f"{self.namespace}/{dataset}-landuse"


def _need(table: Mapping[str, Any], key: str, kind: type[T]) -> T:
    value = table.get(key)
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is int):
        raise SettingsError(f"luf.toml: {key!r} must be {kind.__name__}")
    return value


def parse_settings(raw: Mapping[str, Any]) -> OpsSettings:
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise SettingsError(f"luf.toml: schema_version must be {SCHEMA_VERSION}")
    hub, g5k = raw.get("hub", {}), raw.get("grid5000", {})
    sites = _need(g5k, "sites", list)
    if not sites or not all(isinstance(s, str) and s for s in sites):
        raise SettingsError("luf.toml: 'sites' must be a non-empty list of site names")
    ints = {
        k: _need(g5k, k, int)
        for k in (
            "walltime_minutes",
            "night_walltime_minutes",
            "max_jobs",
            "max_jobs_per_site",
            "interval_seconds",
        )
    }
    if any(v <= 0 for v in ints.values()):
        raise SettingsError("luf.toml: numeric settings must be positive")
    return OpsSettings(
        namespace=_need(hub, "namespace", str),
        bucket=_need(hub, "bucket", str),
        sites=tuple(sites),
        cuda_module=_need(g5k, "cuda_module", str),
        **ints,
    )
