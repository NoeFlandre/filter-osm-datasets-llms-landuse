"""Grid'5000 through SSH to site frontends (only light commands run there)."""

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

REMOTE_ROOT = "luf"  # relative to the site home
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15"]
JOB_PREFIX = "luf-"


class RemoteError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Job:
    site: str
    job_id: str
    name: str
    state: str
    queue: str


def ssh(site: str, command: str, *, timeout: float = 120, stdin: bytes | None = None) -> str:
    try:
        done = subprocess.run(
            ["ssh", *SSH_OPTIONS, site, command],
            input=stdin, capture_output=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RemoteError(f"{site}: timed out: {command[:80]}") from exc
    if done.returncode != 0:
        raise RemoteError(f"{site}: exit {done.returncode}: {done.stderr.decode(errors='replace')[-500:]}")
    return done.stdout.decode()


def rsync(src: str, dst: str, *, extra: tuple[str, ...] = (), timeout: float = 900) -> None:
    command = ["rsync", "-a", "--partial", "-e", "ssh " + " ".join(SSH_OPTIONS), *extra, src, dst]
    done = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    if done.returncode not in (0, 24):  # 24: files vanished during transfer
        raise RemoteError(f"rsync {src} -> {dst}: {done.stderr.decode(errors='replace')[-500:]}")


def our_jobs(site: str) -> list[Job]:
    """Our live jobs on ``site`` (name prefix ``luf-``); others are never touched."""
    payload = json.loads(ssh(site, "oarstat -u -J") or "{}")
    return [
        Job(site, str(job_id), j.get("name") or "", j.get("state", ""), j.get("queue", ""))
        for job_id, j in payload.items()
        if (j.get("name") or "").startswith(JOB_PREFIX)
    ]


def policy_check(site: str) -> None:
    ssh(site, "usagepolicycheck -t", timeout=180)


_JOB_ID = re.compile(r"OAR_JOB_ID=(\d+)")


def submit(site: str, arguments: list[str]) -> str:
    out = ssh(site, "oarsub " + " ".join(shlex.quote(a) for a in arguments), timeout=180)
    match = _JOB_ID.search(out)
    if not match:
        raise RemoteError(f"{site}: oarsub gave no job id: {out[-300:]}")
    return match.group(1)


def scheduled_start(site: str, job_id: str) -> tuple[str, int | None]:
    """(state, predicted start epoch) of one of our jobs."""
    payload = json.loads(ssh(site, f"oarstat -j {int(job_id)} -J") or "{}")
    job = next(iter(payload.values()), {})
    return job.get("state", "Unknown"), job.get("scheduled_start") or job.get("start_time")


def cancel(site: str, job_id: str) -> None:
    ssh(site, f"oardel {int(job_id)}")


def site_status(site: str) -> dict:
    url = f"https://api.grid5000.fr/stable/sites/{site}/status?disks=no&job_details=yes&waiting=yes"
    return json.loads(ssh(site, f"curl -sf {shlex.quote(url)}"))


def inventory(site: str, script: Path) -> list[dict]:
    return json.loads(ssh(site, "python3 -", stdin=script.read_bytes(), timeout=600))


def deploy_code(site: str, commit: str, archive: bytes) -> str:
    """Unpack a ``git archive`` of ``commit`` into ``~/luf/code/<commit>``; idempotent."""
    target = f"{REMOTE_ROOT}/code/{commit}"
    ssh(site, f"test -f {target}/.complete || (rm -rf {target} && mkdir -p {target} && "
              f"tar -x -C {target} && touch {target}/.complete)", stdin=archive, timeout=300)
    ssh(site, "command -v uv >/dev/null || test -x ~/.local/bin/uv || "
              "(curl -LsSf https://astral.sh/uv/install.sh | sh) >/dev/null 2>&1")
    return target


def home_usage(site: str) -> str:
    return ssh(site, "quota -s 2>/dev/null | tail -1; du -sh ~/luf 2>/dev/null || true")
