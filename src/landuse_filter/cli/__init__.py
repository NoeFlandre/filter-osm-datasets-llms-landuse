"""``luf``: plan, run, gate, orchestrate and publish the land-use labelling.

Exit codes: 0 ok, 1 failure, 2 usage error, 3 gate failed, 4 nothing to do,
5 GPU not eligible.
"""

import hashlib
import json
import sys
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from landuse_filter import __version__, config
from landuse_filter.adapters.settings_file import load as load_settings

OPS = load_settings()  # luf.toml (or $LUF_CONFIG): sites, bucket, walltimes, caps


if TYPE_CHECKING:
    from landuse_filter.adapters.store import WorkStore


app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


bench_app = typer.Typer(no_args_is_help=True, help="Benchmark parity: plan, budget, compare, gate.")


g5k_app = typer.Typer(no_args_is_help=True, help="Grid'5000: inventory, controller, jobs, storage.")


node_app = typer.Typer(no_args_is_help=True, help="Commands run on a reserved GPU node.")


app.add_typer(bench_app, name="bench")


app.add_typer(g5k_app, name="g5k")


app.add_typer(node_app, name="node")


PROMPT = resources.files("landuse_filter") / "data" / "prompt.txt"  # packaged with the wheel


WORK = typer.Option(config.work_dir(), "--work", help="Local work tree.")


JSON_OUT = typer.Option(False, "--json", help="Machine-readable output.")


def _store(work: Path) -> "WorkStore":
    from landuse_filter.adapters.store import WorkStore

    return WorkStore(work)


def _template() -> str:
    from landuse_filter.domain.prompting import check_prompt_digest

    raw = PROMPT.read_bytes()
    check_prompt_digest(hashlib.sha256(raw).hexdigest())
    return raw.decode("utf-8")


def _emit(payload: object, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, indent=1, default=str))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            typer.echo(f"{key}: {value}")
    else:
        typer.echo(str(payload))


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def fingerprint(as_json: bool = JSON_OUT) -> None:
    """Show the production generation fingerprint and serving config."""
    _emit(
        {"config_fingerprint": config.GENERATION_FP, "config": config.reference_config()}, as_json
    )


SITES = ",".join(OPS.sites)


def main() -> None:  # pragma: no cover
    sys.exit(app())


# Command groups register themselves on the apps above when imported.
from landuse_filter.cli import bench, data, g5k, node  # noqa: E402, F401
