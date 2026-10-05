"""Remote object store for the work tree: a private HF Bucket, or a directory (tests).

Nodes read chunk inputs from it and write result parts plus small manifests to it,
so neither Grid'5000 homes nor the controller's disk accumulate bulk data. Paths are
the work-tree paths (``chunks/<id>.parquet``, ``parts/<fp>/<chunk>/<part>.parquet``...).
"""

import logging
import random
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, TypeVar

from landuse_filter.domain.retry import LONG, RetryBudget, give_up, next_delay, parse_retry_after

log = logging.getLogger(__name__)
T = TypeVar("T")


def rate_limit_headers(exc: BaseException) -> dict[str, str] | None:
    """The response headers when ``exc`` is an HTTP 429 (Hub or httpx), else ``None``."""
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) != 429:  # noqa: PLR2004
        return None
    return dict(getattr(response, "headers", None) or {})


def with_retry(
    call: Callable[[], T],
    *,
    what: str = "hub call",
    budget: RetryBudget = LONG,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> T:
    """Run ``call``; on HTTP 429 wait as the server asks (ADR-0028), then retry.

    Gives up after the bounded attempts and total wait of ``domain.retry`` and re-raises the
    original error. Other errors pass through at once.
    """
    attempt, waited = 0, 0.0
    while True:
        attempt += 1
        try:
            return call()
        except Exception as exc:
            headers = rate_limit_headers(exc)
            if headers is None:
                raise
            delay = next_delay(attempt, parse_retry_after(headers), jitter(), budget)
            if give_up(attempt, waited, delay, budget):
                raise
            log.warning("%s rate-limited (429); waiting %.0f s (attempt %d)", what, delay, attempt)
            sleep(delay)
            waited += delay


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

    def __init__(self, bucket_id: str, budget: RetryBudget = LONG) -> None:
        self.bucket_id = bucket_id
        self.budget = budget  # SHORT for ingest: give up fast, the next pull retries (ADR-0031)

    def _api(self):  # noqa: ANN202 - huggingface_hub.HfApi, imported lazily
        from huggingface_hub import HfApi

        return HfApi()

    def ensure(self) -> None:
        with_retry(
            lambda: self._api().create_bucket(self.bucket_id, private=True, exist_ok=True),
            what="create_bucket",
            budget=self.budget,
        )

    def put(self, files: Sequence[tuple[Path, str]]) -> None:
        api = self._api()
        for start in range(0, len(files), self.BATCH):
            batch = [(str(src), dst) for src, dst in files[start : start + self.BATCH]]
            with_retry(
                lambda b=batch: api.batch_bucket_files(self.bucket_id, add=b),
                what="put",
                budget=self.budget,
            )

    def get(self, files: Sequence[tuple[str, Path]]) -> None:
        if files:
            pairs = [(src, str(dst)) for src, dst in files]
            with_retry(
                lambda: self._api().download_bucket_files(
                    self.bucket_id, pairs, raise_on_missing_files=True
                ),
                what="get",
                budget=self.budget,
            )

    def _page(self, url: str, params: dict | None) -> tuple[list[dict], str | None]:
        """One page of the tree listing and the URL of the next one, if any."""
        from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

        r = get_session().get(url, params=params, headers=build_hf_headers())
        hf_raise_for_status(r)
        return r.json(), r.links.get("next", {}).get("url")

    def ls(self, prefix: str) -> list[str]:
        """Files under ``prefix``; a 429 on any page retries that page only."""
        from urllib.parse import quote

        url: str | None = (
            f"{self._api().endpoint}/api/buckets/{self.bucket_id}/tree"
            f"{'/' + quote(prefix, safe='') if prefix else ''}"
        )
        params: dict | None = {"recursive": True}
        paths: list[str] = []
        while url is not None:
            items, nxt = with_retry(
                lambda u=url, p=params: self._page(u, p), what=f"ls {prefix}", budget=self.budget
            )
            paths += [i["path"] for i in items if i.get("type") == "file"]
            url, params = nxt, None  # the next link already carries its query
        return sorted(paths)

    def delete(self, paths: Sequence[str]) -> None:
        api = self._api()
        for start in range(0, len(paths), self.BATCH):
            chunk = list(paths[start : start + self.BATCH])
            with_retry(
                lambda c=chunk: api.batch_bucket_files(self.bucket_id, delete=c),
                what="delete",
                budget=self.budget,
            )
