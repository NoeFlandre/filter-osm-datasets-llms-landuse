"""`luf store` commands."""

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from landuse_filter.cli import (
    OPS,
    WORK,
    _store,
    app,
)

if TYPE_CHECKING:
    pass


store_app = typer.Typer(no_args_is_help=True, help="Mirror the work tree to a private HF Bucket.")


app.add_typer(store_app, name="store")


BUCKET = typer.Option(OPS.bucket, help="Private Hugging Face Bucket id.")


@store_app.command("push")
def store_push(work: Path = WORK, bucket: str = BUCKET) -> None:
    """Upload the local work tree (plans, chunks, parts, ledger, gates) to the bucket."""
    from landuse_filter.adapters.store import Bucket

    target = Bucket(bucket)
    target.ensure()
    target.push(_store(work))
    typer.echo(f"pushed {work} -> hf://buckets/{bucket}")


@store_app.command("pull")
def store_pull(work: Path = WORK, bucket: str = BUCKET) -> None:
    """Restore the work tree from the bucket (e.g. on a new controller machine)."""
    from landuse_filter.adapters.store import Bucket

    Bucket(bucket).pull(_store(work))
    typer.echo(f"pulled hf://buckets/{bucket} -> {work}")
