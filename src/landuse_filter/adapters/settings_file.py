"""Read ``luf.toml`` (repo root by default, or ``$LUF_CONFIG``)."""

import os
import tomllib
from pathlib import Path

from landuse_filter.domain.settings import OpsSettings, parse_settings

DEFAULT = Path(__file__).resolve().parents[3] / "luf.toml"


def load(path: Path | None = None) -> OpsSettings:
    target = path or Path(os.environ.get("LUF_CONFIG", DEFAULT))
    with target.open("rb") as f:
        return parse_settings(tomllib.load(f))
