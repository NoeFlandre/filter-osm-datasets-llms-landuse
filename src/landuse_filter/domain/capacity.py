"""Free GPUs per cluster from the Grid'5000 status API, and OAR submission arguments."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True, slots=True)
class Cluster:
    site: str
    name: str
    gpu: str
    memory_mib: int
    compute_capability: tuple[int, int]
    gpus_per_node: int
    nodes: int
    queues: tuple[str, ...]
    exotic: bool
    vendor: str = "Nvidia"
    arch: str = "x86_64"

    @property
    def production(self) -> bool:
        return "production" in self.queues or "abaca" in self.queues

    @property
    def submittable(self) -> bool:
        """Reachable through abaca or default (e.g. Nancy's ``granite`` is testing-only)."""
        return self.production or "default" in self.queues


def _blocked(state: Mapping, now: float, walltime_s: float) -> bool:
    """A waiting reservation on the node starts before our job would end."""
    return any(
        r.get("state") == "waiting"
        and r.get("scheduled_at")
        and r["scheduled_at"] < now + walltime_s
        for r in state.get("reservations", [])
    )


_USABLE = ("alive", "standby", None)


def _node_gpus(state: Mapping, gpus_per_node: int, *, besteffort_counts: bool) -> int:
    """Whole GPUs free on one node, from its CPU-slot accounting."""
    freeable = state.get("freeable_slots", 0)
    slots = state.get("free_slots", 0) + (freeable if besteffort_counts else 0)
    all_slots = slots + state.get("busy_slots", 0) + (0 if besteffort_counts else freeable)
    if all_slots <= 0:
        return 0
    return slots * gpus_per_node // all_slots


def _usable(cluster: Cluster, host: str, state: Mapping, now: float, walltime_s: float) -> bool:
    if not host.startswith(f"{cluster.name}-") or state.get("hard") not in _USABLE:
        return False
    return not (walltime_s and _blocked(state, now, walltime_s))


def free_gpus(
    cluster: Cluster,
    nodes: Mapping[str, Mapping],
    *,
    besteffort_counts: bool,
    now: float = 0.0,
    walltime_s: float = 0.0,
) -> int:
    """GPUs we could hold *for the whole walltime* on ``cluster``.

    Free slots (plus besteffort-held ones, which a regular job with priority preempts)
    are converted to whole GPUs; a node with a waiting reservation starting before
    ``now + walltime_s`` counts as full, since the scheduler keeps it for that job.
    """
    usable = [s for host, s in nodes.items() if _usable(cluster, host, s, now, walltime_s)]
    return sum(
        _node_gpus(s, cluster.gpus_per_node, besteffort_counts=besteffort_counts) for s in usable
    )


def walltime_text(walltime: timedelta) -> str:
    minutes = walltime // timedelta(minutes=1)
    return f"{minutes // 60}:{minutes % 60:02d}"


def _queue_and_types(cluster: Cluster, job_type: str | None, *, besteffort: bool) -> list[str]:
    """Always name the queue: some sites route unqualified jobs elsewhere (regression:
    Lyon rejected an unqualified sirius job with "queue 'abaca' does not exist")."""
    if besteffort:
        queue = ["-t", "besteffort"]
    else:
        queue = ["-q", "abaca" if cluster.production else "default"]
    exotic = ["-t", "exotic"] if cluster.exotic else []
    return queue + exotic + _job_type(cluster, job_type, besteffort=besteffort)


def _job_type(cluster: Cluster, job_type: str | None, *, besteffort: bool) -> list[str]:
    """``-t night``/``-t day`` only applies to the default queue."""
    usable = job_type and not cluster.production and not besteffort
    return ["-t", job_type] if usable and job_type else []


def oarsub_arguments(  # noqa: PLR0913 - one parameter per OAR option
    cluster: Cluster,
    walltime: timedelta,
    job_type: str | None,
    name: str,
    *,
    command: str,
    besteffort: bool = False,
    gpus: int = 1,
    log_dir: str = "luf/logs",
) -> list[str]:
    args = _queue_and_types(cluster, job_type, besteffort=besteffort)
    args += [
        "-p",
        f"cluster='{cluster.name}'",
        "-l",
        f"host=1/gpu={gpus},walltime={walltime_text(walltime)}",
        "--checkpoint",
        # OAR refuses checkpoints longer than 60 s for besteffort (regression: Grenoble).
        "60" if besteffort else "300",
        "-n",
        name,
        "-O",
        f"{log_dir}/%jobid%.out",
        "-E",
        f"{log_dir}/%jobid%.err",
        command,
    ]
    return args
