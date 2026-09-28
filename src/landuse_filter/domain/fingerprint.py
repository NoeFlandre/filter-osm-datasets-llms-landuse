"""Identity of everything that can change a generation (ADR-0005)."""

import json
from collections.abc import Mapping
from typing import Any

from landuse_filter.domain.hashing import sha256_text

# SGLang arguments that change throughput only. Whether they are truly output-neutral
# is decided by the benchmark gate, never assumed: a config that changes any of them
# still needs its own passing gate file (ADR-0006).
SPEED_ONLY_ARGS = frozenset(
    {
        "mem_fraction_static",
        "max_running_requests",
        "cuda_graph_max_bs",
        "chunked_prefill_size",
        "disable_radix_cache",
        "dp_size",
        "log_level",
    }
)


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """sha256 of the output-affecting fields of ``config``.

    ``config`` holds ``engine`` (SGLang kwargs) and top-level identity fields (model,
    revisions, prompt and template digests, sampling). Speed-only engine args are
    dropped so profiles tuned per GPU share one generation cache.
    """
    engine = {k: v for k, v in config.get("engine", {}).items() if k not in SPEED_ONLY_ARGS}
    identity = {k: v for k, v in config.items() if k != "engine"}
    return sha256_text(canonical_json({**identity, "engine": engine}))[:16]


def serving_fingerprint(config: Mapping[str, Any]) -> str:
    """sha256 of the *whole* config, speed args included: what a gate file approves."""
    return sha256_text(canonical_json(config))[:16]
