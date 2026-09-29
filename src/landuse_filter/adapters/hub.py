"""Publishing to the Hugging Face Hub (dataset repos)."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import IO

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


def open_file(repo_id: str, path: str) -> IO[bytes]:
    """Read-only file object of a file already on the Hub (ranged reads, no full download)."""
    from huggingface_hub import HfFileSystem

    return HfFileSystem().open(f"datasets/{repo_id}/{path}", "rb")


def delete(repo_id: str, paths: Sequence[str], message: str) -> None:
    from huggingface_hub import CommitOperationDelete, HfApi

    HfApi().create_commit(
        repo_id,
        [CommitOperationDelete(path_in_repo=p) for p in paths],
        commit_message=message,
        repo_type="dataset",
    )
