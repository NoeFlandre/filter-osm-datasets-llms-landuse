"""Validated operational settings (pure). Loaded from ``luf.toml`` by an adapter."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, TypeVar

from landuse_filter.domain.policy_check import MODES, PER_JOB
from landuse_filter.domain.scheduling import DEFAULT_CHUNK_OVERFLOW, LONG_FAILURES

T = TypeVar("T")

SCHEMA_VERSION = 1


class SettingsError(ValueError):
    """``luf.toml`` is missing a field, has a wrong type, or a wrong schema version."""


DEFAULT_LONG_FAILURES = LONG_FAILURES
DEFAULT_SUBMIT_WORKERS = 1  # sites submitted to in parallel (ADR-0020)
DEFAULT_BUCKET = "NoeFlandre/landuse-filter-work"  # private work bucket; luf.toml [hub] bucket


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
    day_long_max_failures: int = DEFAULT_LONG_FAILURES
    chunk_overflow: float = DEFAULT_CHUNK_OVERFLOW
    policy_check: str = PER_JOB  # usagepolicycheck cadence (ADR-0019)
    submit_workers: int = DEFAULT_SUBMIT_WORKERS

    def output_repo(self, dataset: str) -> str:
        return f"{self.namespace}/{dataset}-landuse"


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


def _str(table: Mapping[str, Any], key: str) -> str:
    return _need(table, key, str)


def _sites(g5k: Mapping[str, Any], key: str) -> tuple[str, ...]:
    sites = _need(g5k, key, list)
    if not sites or not all(isinstance(s, str) and s for s in sites):
        raise SettingsError("luf.toml: 'sites' must be a non-empty list of site names")
    return tuple(sites)


def _positive(g5k: Mapping[str, Any], key: str) -> int:
    value = _need(g5k, key, int)
    if value < 1:
        raise SettingsError(f"luf.toml: {key!r} must be positive")
    return value


def _overflow(g5k: Mapping[str, Any], key: str) -> float:
    value = g5k[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"luf.toml: {key!r} must be float")
    if value < 1.0:
        raise SettingsError(f"luf.toml: {key!r} must be at least 1.0")
    return float(value)


def _policy_check(g5k: Mapping[str, Any], key: str) -> str:
    value = g5k[key]
    if value not in MODES:
        raise SettingsError(f"luf.toml: {key!r} must be one of {', '.join(MODES)}")
    return value


@dataclass(frozen=True, slots=True)
class Field:
    """One ``luf.toml`` setting: where it lives, how it is parsed and coerced from text."""

    name: str
    table: str
    kind: str  # "str", "int", "float" or "list": drives LUF_<NAME> coercion
    parse: Callable[[Mapping[str, Any], str], Any]
    required: bool = True
    default: Any = None


def _optional(
    name: str, kind: str, parse: Callable[[Mapping[str, Any], str], Any], default: object
) -> Field:
    return Field(name, "grid5000", kind, parse, required=False, default=default)


# The one place that lists the settings; parsing and environment overrides derive from it.
# Order is the order in which a bad value is reported.
FIELDS: Final = (
    _optional("day_walltime_minutes", "int", _positive, None),
    _optional("day_long_max_failures", "int", _positive, DEFAULT_LONG_FAILURES),
    _optional("chunk_overflow", "float", _overflow, DEFAULT_CHUNK_OVERFLOW),
    _optional("policy_check", "str", _policy_check, PER_JOB),
    _optional("submit_workers", "int", _positive, DEFAULT_SUBMIT_WORKERS),
    Field("namespace", "hub", "str", _str),
    Field("bucket", "hub", "str", _str),
    Field("sites", "grid5000", "list", _sites),
    Field("cuda_module", "grid5000", "str", _str),
    Field("walltime_minutes", "grid5000", "int", _positive),
    Field("night_walltime_minutes", "grid5000", "int", _positive),
    Field("night_fallback_walltime_minutes", "grid5000", "int", _positive),
    Field("max_jobs", "grid5000", "int", _positive),
    Field("max_jobs_per_site", "grid5000", "int", _positive),
    Field("interval_seconds", "grid5000", "int", _positive),
)


def parse_settings(raw: Mapping[str, Any]) -> OpsSettings:
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise SettingsError(f"luf.toml: schema_version must be {SCHEMA_VERSION}")
    tables = {"hub": _table(raw, "hub"), "grid5000": _table(raw, "grid5000")}
    values = {
        f.name: f.default
        if not f.required and f.name not in tables[f.table]
        else f.parse(tables[f.table], f.name)
        for f in FIELDS
    }
    return OpsSettings(**values)
