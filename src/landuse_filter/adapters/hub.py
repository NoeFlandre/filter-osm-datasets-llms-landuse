"""Publishing to the Hugging Face Hub (dataset repos)."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import IO, Protocol

BATCH = 200  # files per commit


def ensure_dataset(repo_id: str) -> None:
    from huggingface_hub import HfApi

    HfApi().create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)


def remote_files(repo_id: str) -> set[str]:
    from huggingface_hub import HfApi

    return set(HfApi().list_repo_files(repo_id, repo_type="dataset"))


def upload(repo_id: str, files: Sequence[tuple[Path, str]], message: str) -> None:
    """Commit ``(local path, path in repo)`` pairs in batches (Xet dedupes identical bytes)."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    for start in range(0, len(files), BATCH):
        ops = [
            CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src))
            for src, dst in files[start : start + BATCH]
        ]
        api.create_commit(
            repo_id,
            ops,
            commit_message=f"{message} ({start + len(ops)}/{len(files)})",
            repo_type="dataset",
        )


def download_all(repo_id: str, revision: str, paths: Iterable[str]) -> list[tuple[Path, str]]:
    from huggingface_hub import hf_hub_download

    return [
        (Path(hf_hub_download(repo_id, p, repo_type="dataset", revision=revision)), p)
        for p in paths
    ]


def list_files(repo_id: str, revision: str) -> list[str]:
    from huggingface_hub import HfApi

    return sorted(HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision))


def open_file(repo_id: str, path: str, revision: str | None = None) -> IO[bytes]:
    """Read-only file object of a file already on the Hub (ranged reads, no full download)."""
    from huggingface_hub import HfFileSystem

    at = f"@{revision}" if revision else ""
    return HfFileSystem().open(f"datasets/{repo_id}{at}/{path}", "rb")


def delete(repo_id: str, paths: Sequence[str], message: str) -> None:
    from huggingface_hub import CommitOperationDelete, HfApi

    HfApi().create_commit(
        repo_id,
        [CommitOperationDelete(path_in_repo=p) for p in paths],
        commit_message=message,
        repo_type="dataset",
    )


class Hub(Protocol):
    """The Hub operations publication needs (tests pass a fake, production :class:`HfHub`)."""

    def ensure_dataset(self, repo_id: str) -> None: ...
    def remote_files(self, repo_id: str) -> set[str]: ...
    def upload(self, repo_id: str, files: Sequence[tuple[Path, str]], message: str) -> None: ...
    def download_all(
        self, repo_id: str, revision: str, paths: Iterable[str]
    ) -> list[tuple[Path, str]]: ...
    def open_file(self, repo_id: str, path: str, revision: str | None = None) -> IO[bytes]: ...
    def list_files(self, repo_id: str, revision: str) -> list[str]: ...


class HfHub:
    """The real Hugging Face Hub, through the module functions above."""

    def ensure_dataset(self, repo_id: str) -> None:
        ensure_dataset(repo_id)

    def remote_files(self, repo_id: str) -> set[str]:
        return remote_files(repo_id)

    def upload(self, repo_id: str, files: Sequence[tuple[Path, str]], message: str) -> None:
        upload(repo_id, files, message)

    def download_all(
        self, repo_id: str, revision: str, paths: Iterable[str]
    ) -> list[tuple[Path, str]]:
        return download_all(repo_id, revision, paths)

    def open_file(self, repo_id: str, path: str, revision: str | None = None) -> IO[bytes]:
        return open_file(repo_id, path, revision)

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        return list_files(repo_id, revision)
