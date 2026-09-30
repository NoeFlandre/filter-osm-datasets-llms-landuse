"""The production serving configuration (one place, versioned with the code)."""

import os
from pathlib import Path
from typing import Any, TypedDict

from landuse_filter.domain.fingerprint import config_fingerprint
from landuse_filter.domain.prompting import PROMPT_SHA256

MODEL_ID = "LiquidAI/LFM2.5-2.6B"
MODEL_REVISION = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
DRAFT_ID = "LiquidAI/LFM2.5-2.6B-DSpark"
DRAFT_REVISION = "458cedab07d0f7b2b05700c77e1aa463d43d6f04"
MAX_NEW_TOKENS = 4096  # kept: lower caps fail the gate or save < 1% (docs/tuning.md)


def scratch_dir() -> Path:
    """Node-local scratch: ``$LUF_SCRATCH`` (set per job by node_job.sh), else /tmp/luf-scratch."""
    return Path(os.environ.get("LUF_SCRATCH", "/tmp/luf-scratch"))  # noqa: S108


def work_dir() -> Path:
    """Local work tree: ``$LUF_WORK``, else ``work``."""
    return Path(os.environ.get("LUF_WORK", "work"))


class Sampling(TypedDict):
    temperature: float
    max_new_tokens: int


class EngineArgs(TypedDict):
    dtype: str
    random_seed: int
    speculative_algorithm: str
    speculative_draft_attention_backend: str
    disable_radix_cache: bool
    mem_fraction_static: float


class ReferenceConfig(TypedDict):
    """The output-affecting serving configuration (its fingerprint is ``GENERATION_FP``)."""

    model: str
    draft: str
    prompt_sha256: str
    chat_template_kwargs: dict[str, bool]
    sampling: Sampling
    engine: EngineArgs


def reference_config() -> ReferenceConfig:
    """Benchmark run ``LiquidAI/LFM2.5-2.6B+DSpark-throughput-b16``, output-affecting part."""
    return {
        "model": f"{MODEL_ID}@{MODEL_REVISION}",
        "draft": f"{DRAFT_ID}@{DRAFT_REVISION}",
        "prompt_sha256": PROMPT_SHA256,
        "chat_template_kwargs": {"enable_thinking": False},
        "sampling": {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS},
        "engine": {
            "dtype": "bfloat16",
            "random_seed": 0,
            "speculative_algorithm": "DSPARK",
            "speculative_draft_attention_backend": "flashinfer",
            "disable_radix_cache": True,
            "mem_fraction_static": 0.75,
        },
    }


def engine_kwargs(config: ReferenceConfig, speed: dict[str, Any]) -> dict[str, Any]:
    """``sglang.Engine`` kwargs: model paths, output args and per-GPU speed args."""
    return {
        "model_path": MODEL_ID,
        "revision": MODEL_REVISION,
        "speculative_draft_model_path": DRAFT_ID,
        "speculative_draft_model_revision": DRAFT_REVISION,
        **config["engine"],
        **speed,
    }


GENERATION_FP = config_fingerprint(reference_config())
