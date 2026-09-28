"""Remote object store for the work tree: a private HF Bucket, or a directory (tests).

Nodes read chunk inputs from it and write result parts plus small manifests to it,
so neither Grid'5000 homes nor the controller's disk accumulate bulk data. Paths are
the work-tree paths (``chunks/<id>.parquet``, ``parts/<fp>/<chunk>/<part>.parquet``...).
"""

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class Remote(Protocol):
    def put(self, files: Sequence[tuple[Path, str]]) -> None: ...
    def get(self, files: Sequence[tuple[str, Path]]) -> None: ...
    def ls(self, prefix: str) -> list[str]: ...
    def delete(self, paths: Sequence[str]) -> None: ...


class DirRemote:
    """A directory standing in for the bucket (tests, dry runs)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, files: Sequence[tuple[Path, str]]) -> None:
        for src, dst in files:
            target = self.root / dst
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)

    def get(self, files: Sequence[tuple[str, Path]]) -> None:
        for src, dst in files:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / src, dst)

    def ls(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if base.is_file():
            return [prefix]
        return (
            sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file())
            if base.exists()
            else []
        )

    def delete(self, paths: Sequence[str]) -> None:
        for p in paths:
            (self.root / p).unlink(missing_ok=True)


class BucketRemote:
    """A private Hugging Face Bucket (token from the environment or HF login)."""

    BATCH = 500

    def __init__(self, bucket_id: str) -> None:
        self.bucket_id = bucket_id

    def _api(self):  # noqa: ANN202 - huggingface_hub.HfApi, imported lazily
        from huggingface_hub import HfApi

        return HfApi()

    def ensure(self) -> None:
        self._api().create_bucket(self.bucket_id, private=True, exist_ok=True)

    def put(self, files: Sequence[tuple[Path, str]]) -> None:
        api = self._api()
        for start in range(0, len(files), self.BATCH):
            batch = [(str(src), dst) for src, dst in files[start : start + self.BATCH]]
            api.batch_bucket_files(self.bucket_id, add=batch)

    def get(self, files: Sequence[tuple[str, Path]]) -> None:
        if files:
            self._api().download_bucket_files(
                self.bucket_id, [(src, str(dst)) for src, dst in files], raise_on_missing_files=True
            )

    def ls(self, prefix: str) -> list[str]:
        entries = self._api().list_bucket_tree(self.bucket_id, prefix=prefix, recursive=True)
        return sorted(e.path for e in entries if getattr(e, "type", "file") == "file")

    def delete(self, paths: Sequence[str]) -> None:
        api = self._api()
        for start in range(0, len(paths), self.BATCH):
            api.batch_bucket_files(self.bucket_id, delete=list(paths[start : start + self.BATCH]))
