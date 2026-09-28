"""GPU eligibility and per-model serving profiles."""

from dataclasses import dataclass
from enum import StrEnum

MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_MEMORY_MIB = 16 * 1024


class Admission(StrEnum):
    """One-time smoke-gate status of a GPU model (issue #18)."""

    PENDING = "pending"
    ADMITTED = "admitted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class GpuSpec:
    model: str
    compute_capability: tuple[int, int]
    memory_mib: int


def ineligibility(spec: GpuSpec) -> str | None:
    """Why ``spec`` cannot serve BF16 LFM2.5 + DSpark with FlashInfer, or ``None``."""
    if spec.compute_capability < MIN_COMPUTE_CAPABILITY:
        major, minor = spec.compute_capability
        return f"compute capability {major}.{minor} < 8.0"
    if spec.memory_mib < MIN_MEMORY_MIB:
        return f"{spec.memory_mib} MiB < {MIN_MEMORY_MIB} MiB"
    return None


def gpu_key(model: str) -> str:
    """Normalised GPU model name used for profiles and gate files."""
    cleaned = model.lower().replace("nvidia", "").replace("-", " ")
    return "_".join(cleaned.split())


@dataclass(frozen=True, slots=True)
class Profile:
    """Speed-only SGLang settings for one GPU model, plus measured throughput."""

    gpu: str
    max_running_requests: int = 64
    mem_fraction_static: float = 0.75
    cuda_graph_max_bs: int | None = None
    sentences_per_second: float = 1.4  # benchmark L40S reference until calibrated
    calibrated: bool = False

    def engine_args(self) -> dict[str, object]:
        args: dict[str, object] = {
            "max_running_requests": self.max_running_requests,
            "mem_fraction_static": self.mem_fraction_static,
        }
        if self.cuda_graph_max_bs is not None:
            args["cuda_graph_max_bs"] = self.cuda_graph_max_bs
        return args
