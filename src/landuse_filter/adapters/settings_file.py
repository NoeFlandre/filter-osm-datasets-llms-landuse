"""Operational settings: defaults < ``luf.toml`` < environment < CLI option.

``luf.toml`` (repo root, or ``$LUF_CONFIG``) holds the defaults; ``LUF_<FIELD>``
environment variables override a field (see ``ENV_FIELDS``); a CLI option, where one
exists, wins over both because Typer uses this loader's value only as the default.
"""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from landuse_filter.domain.settings import (
    INT_FIELDS,
    OPTIONAL_INT_FIELDS,
    OpsSettings,
    SettingsError,
    parse_settings,
)

DEFAULT = Path(__file__).resolve().parents[3] / "luf.toml"

# field -> toml table; the environment variable is LUF_<FIELD in upper case>
ENV_FIELDS = {
    "namespace": "hub",
    "bucket": "hub",
    "sites": "grid5000",
    "cuda_module": "grid5000",
    "chunk_overflow": "grid5000",
    "policy_check": "grid5000",
    **dict.fromkeys((*INT_FIELDS, *OPTIONAL_INT_FIELDS), "grid5000"),
}


def _env_value(field: str, text: str) -> Any:
    if field == "sites":
        return [s.strip() for s in text.split(",")]
    if field == "chunk_overflow":
        try:
            return float(text)
        except ValueError:
            raise SettingsError(f"LUF_{field.upper()}: {text!r} is not a number") from None
    if field in INT_FIELDS or field in OPTIONAL_INT_FIELDS:
        try:
            return int(text)
        except ValueError:
            raise SettingsError(f"LUF_{field.upper()}: {text!r} is not an integer") from None
    return text


def with_environment(raw: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
    """Copy of the parsed toml with every ``LUF_<FIELD>`` variable applied."""
    merged = {k: dict(v) if isinstance(v, Mapping) else v for k, v in raw.items()}
    for field, table in ENV_FIELDS.items():
        text = environ.get(f"LUF_{field.upper()}")
        if text is not None and isinstance(merged.get(table), dict):
            merged[table][field] = _env_value(field, text)
    return merged


def load(path: Path | None = None, environ: Mapping[str, str] | None = None) -> OpsSettings:
    env = os.environ if environ is None else environ
    target = path or Path(env.get("LUF_CONFIG", DEFAULT))
    with target.open("rb") as f:
        return parse_settings(with_environment(tomllib.load(f), env))
