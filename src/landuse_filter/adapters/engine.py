"""SGLang offline engine with the DSpark draft, driven asynchronously.

Only this module imports SGLang (the ``gpu`` extra); the benchmark's shutdown lesson
applies: GPU memory is released only when the spawned scheduler exits.
"""

import multiprocessing
import time
from typing import Any

SHUTDOWN_TIMEOUT = 120.0


class SGLangEngine:
    def __init__(self, engine_kwargs: dict[str, Any], sampling: dict[str, Any]) -> None:
        import sglang  # ty: ignore[unresolved-import]  # the `gpu` extra, nodes only

        self._engine: Any = sglang.Engine(**engine_kwargs)
        self._sampling = dict(sampling)
        self.version = str(getattr(sglang, "__version__", "unknown"))

    async def generate(self, input_ids: list[int]) -> dict:
        return await self._engine.async_generate(
            input_ids=input_ids, sampling_params=self._sampling
        )

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
