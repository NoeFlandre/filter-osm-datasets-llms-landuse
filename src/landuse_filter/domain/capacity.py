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
    total = 0
    for host, state in nodes.items():
        if not host.startswith(f"{cluster.name}-") or state.get("hard") not in (
            "alive",
            "standby",
            None,
        ):
            continue
        if walltime_s and _blocked(state, now, walltime_s):
            continue
        slots = state.get("free_slots", 0) + (
            state.get("freeable_slots", 0) if besteffort_counts else 0
        )
        all_slots = (
            slots
            + state.get("busy_slots", 0)
            + (0 if besteffort_counts else state.get("freeable_slots", 0))
        )
        if all_slots <= 0:
            continue
        per_gpu = all_slots / cluster.gpus_per_node
        total += int(slots // per_gpu) if per_gpu else 0
    return total


def walltime_text(walltime: timedelta) -> str:
    minutes = int(walltime.total_seconds() // 60)
    return f"{minutes // 60}:{minutes % 60:02d}"


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
    args: list[str] = []
    # Always name the queue: some sites route unqualified jobs elsewhere
    # (regression: Lyon rejected an unqualified sirius job with "queue 'abaca' does not exist").
    if besteffort:
        pass
    elif cluster.production:
        args += ["-q", "abaca"]
    else:
        args += ["-q", "default"]
    if besteffort:
        args += ["-t", "besteffort"]
    if cluster.exotic:
        args += ["-t", "exotic"]
    if job_type and not cluster.production and not besteffort:
        args += ["-t", job_type]
    args += [
        "-p",
        f"cluster='{cluster.name}'",
        "-l",
        f"host=1/gpu={gpus},walltime={walltime_text(walltime)}",
        "--checkpoint",
        "300",
        "-n",
        name,
        "-O",
        f"{log_dir}/%jobid%.out",
        "-E",
        f"{log_dir}/%jobid%.err",
        command,
    ]
    return args
