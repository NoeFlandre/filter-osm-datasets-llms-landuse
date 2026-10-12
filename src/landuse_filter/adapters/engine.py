"""SGLang offline engine with the DSpark draft, driven asynchronously.

Only this module imports SGLang (the ``gpu`` extra); the benchmark's shutdown lesson
applies: GPU memory is released only when the spawned scheduler exits.
"""

import multiprocessing
import re
import time
from collections.abc import Mapping
from typing import Any

from landuse_filter.domain.records import MAX_RECORDED_TOKEN_COUNT

SHUTDOWN_TIMEOUT = 120.0
_CONTEXT_LENGTH_OVERFLOW = re.compile(
    r"The input \((?P<input_tokens>\d+) tokens\) is longer than the model's context length "
    r"\((?P<context_limit>\d+) tokens\)",
    re.IGNORECASE,
)
_REQUESTED_TOKEN_BUDGET_OVERFLOW = re.compile(
    r"Requested token count exceeds the model's maximum context length\s+"
    r"of (?P<context_limit>\d+) tokens\.\s+You requested a total of "
    r"(?P<total_tokens>\d+) tokens:\s+"
    r"(?P<input_tokens>\d+) tokens from the input messages and "
    r"(?P<max_new_tokens>\d+) tokens for the completion\.",
    re.IGNORECASE,
)


class ContextLengthExceededError(RuntimeError):
    """One request was rejected at SGLang's configured context limit."""

    def __init__(
        self, input_tokens: int, context_limit: int, max_new_tokens: int | None = None
    ) -> None:
        self.input_tokens = input_tokens
        self.context_limit = context_limit
        self.max_new_tokens = max_new_tokens
        super().__init__(f"input has {input_tokens} tokens; context limit is {context_limit}")


def context_length_overflow(message: str) -> tuple[int, int, int | None] | None:
    """Return counts for a SGLang request rejected by its context validation."""
    match = _CONTEXT_LENGTH_OVERFLOW.search(message)
    if match is not None:
        input_tokens, context_limit = (
            int(match[name]) for name in ("input_tokens", "context_limit")
        )
        if not 0 < context_limit <= input_tokens <= MAX_RECORDED_TOKEN_COUNT:
            return None
        return input_tokens, context_limit, None

    match = _REQUESTED_TOKEN_BUDGET_OVERFLOW.search(message)
    if match is None:
        return None
    context_limit, total_tokens, input_tokens, max_new_tokens = (
        int(match[name])
        for name in ("context_limit", "total_tokens", "input_tokens", "max_new_tokens")
    )
    if (
        not 0 < input_tokens < context_limit <= MAX_RECORDED_TOKEN_COUNT
        or not 0 < max_new_tokens <= MAX_RECORDED_TOKEN_COUNT
        or total_tokens != input_tokens + max_new_tokens
        or not context_limit < total_tokens <= MAX_RECORDED_TOKEN_COUNT
    ):
        return None
    return input_tokens, context_limit, max_new_tokens


class SGLangEngine:
    def __init__(self, engine_kwargs: Mapping[str, Any], sampling: Mapping[str, Any]) -> None:
        import sglang  # ty: ignore[unresolved-import]  # the `gpu` extra, nodes only

        self._engine: Any = sglang.Engine(**engine_kwargs)
        self._sampling = dict(sampling)
        self.version = str(getattr(sglang, "__version__", "unknown"))

    async def generate(self, input_ids: list[int]) -> dict:
        try:
            return await self._engine.async_generate(
                input_ids=input_ids, sampling_params=self._sampling
            )
        except ValueError as exc:
            overflow = context_length_overflow(str(exc))
            if overflow is None:
                raise
            raise ContextLengthExceededError(*overflow) from exc

    def shutdown(self) -> None:
        self._engine.shutdown()
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT
        for child in multiprocessing.active_children():
            child.join(max(0.0, deadline - time.monotonic()))


def gpu_names() -> list[str]:
    """Names of the visible CUDA devices, from nvidia-smi (no torch import needed)."""
    import subprocess

    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]
