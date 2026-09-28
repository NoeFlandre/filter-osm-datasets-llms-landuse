"""GPU clusters we may use: the cached inventory, eligibility, admission, profiles."""

from landuse_filter.adapters.store import WorkStore
from landuse_filter.domain.capacity import Cluster
from landuse_filter.domain.gpu import Admission, GpuSpec, Profile, gpu_key, ineligibility


def load_clusters(store: WorkStore) -> list[Cluster]:
    return [
        Cluster(
            c["site"],
            c["cluster"],
            c["gpu"],
            c["memory_mib"],
            tuple(c["compute_capability"]),
            c["gpus_per_node"],
            c["nodes"],
            tuple(c["queues"]),
            c["exotic"],
            c.get("vendor", "Nvidia"),
            c.get("arch", "x86_64"),
        )
        for c in store.read_json("inventory.json")
    ]


def eligible(cluster: Cluster) -> bool:
    spec = GpuSpec(cluster.gpu, cluster.compute_capability, cluster.memory_mib)
    # SGLang/FlashInfer wheels are x86_64 only (excludes e.g. Lyon's GH200 nodes).
    return (
        cluster.vendor.lower() == "nvidia"
        and cluster.arch == "x86_64"
        and cluster.submittable
        and ineligibility(spec) is None
    )


def admission(store: WorkStore, gpu: str) -> Admission:
    path = f"gates/admission/{gpu_key(gpu)}.json"
    return Admission(store.read_json(path)["status"]) if store.exists(path) else Admission.PENDING


def profile_for(store: WorkStore, gpu: str) -> Profile:
    path = f"profiles/{gpu_key(gpu)}.json"
    if store.exists(path):
        return Profile(**store.read_json(path))
    return Profile(gpu_key(gpu), max_running_requests=16)
