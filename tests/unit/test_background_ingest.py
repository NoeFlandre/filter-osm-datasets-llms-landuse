"""Ingest never starves submission (ADR-0031): background thread, short budget, bounded pull."""

import threading
import time
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError
from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.adapters.remote import with_retry
from landuse_filter.application.ingest_worker import BackgroundIngest
from landuse_filter.application.staging import PullReport
from landuse_filter.domain.retry import LONG, SHORT, give_up, next_delay
from tests.unit.test_controller import NOW
from tests.unit.test_controller_parallel import SITES, build
from tests.unit.test_incremental_ingest import setup, upload


def limited(headers=None):
    response = httpx.Response(429, headers=headers or {}, request=httpx.Request("GET", "http://x"))
    return HfHubHTTPError("429", response=response)


# --- the short budget -------------------------------------------------------------


def test_short_budget_gives_up_after_three_attempts_and_caps_each_wait():
    waits, calls = [], []

    def call():
        calls.append(1)
        raise limited({"Retry-After": "500"})

    with pytest.raises(HfHubHTTPError):
        with_retry(call, budget=SHORT, sleep=waits.append, jitter=lambda: 0.0)
    assert 2 <= len(calls) <= SHORT.max_attempts == 3
    assert waits
    assert max(waits) <= SHORT.max_delay == 60.0
    assert sum(waits) <= SHORT.max_total_wait


def test_long_budget_is_unchanged():
    assert (LONG.max_attempts, LONG.max_delay, LONG.max_total_wait) == (10, 300.0, 1200.0)


@given(
    attempt=st.integers(1, 30),
    retry_after=st.one_of(st.none(), st.floats(0, 10_000)),
    jitter=st.floats(0, 0.999),
    waited=st.floats(0, 5000),
)
def test_every_wait_and_the_total_stay_within_the_budget(attempt, retry_after, jitter, waited):
    for budget in (SHORT, LONG):
        delay = next_delay(attempt, retry_after, jitter, budget)
        assert 0 <= delay <= budget.max_delay
        if not give_up(attempt, waited, delay, budget):
            assert attempt < budget.max_attempts
            assert waited + delay <= budget.max_total_wait


# --- a bounded pull ---------------------------------------------------------------


def test_a_rate_limited_listing_ends_the_pull_and_stays_owed(tmp_path):
    transport, progress, remote, clock = setup(tmp_path)
    transport.pull(progress, {"c1", "c2"})
    upload(remote, "c1", "p1", ["a"])
    upload(remote, "c2", "p1", ["b"])
    listed = []
    real = remote.ls

    def ls(prefix):
        listed.append(prefix)
        raise limited()

    remote.ls = ls
    report = transport.pull(progress, set())
    assert report.limited
    assert report.failed == 1
    assert len(listed) == 1  # no hammering: the first rate limit ends the pull
    remote.ls = real
    clock.now = 180
    report = transport.pull(progress, set())
    assert not report.limited
    assert report.manifests == 2
    assert progress.index("fp").count("c1") == progress.index("fp").count("c2") == 1


def test_a_failed_first_full_listing_is_retried_and_catches_up_completely(tmp_path):
    transport, progress, remote, clock = setup(tmp_path)
    upload(remote, "c1", "p1", ["a"])
    real = remote.ls

    def fail(prefix):
        raise limited()

    remote.ls = fail
    assert transport.pull(progress, {"c1"}).limited
    assert progress.index("fp").count("c1") == 0
    remote.ls = real
    clock.now = 180
    report = transport.pull(progress, {"c1"})
    assert report.full
    assert report.manifests == 1


def test_a_pull_starts_no_listing_after_its_time_budget(tmp_path):
    transport, progress, remote, clock = setup(tmp_path)
    transport.pull(progress, {"c1", "c2", "c3"})
    for c in ("c1", "c2", "c3"):
        upload(remote, c, "p1", [c])
    real = remote.ls

    def slow(prefix):
        clock.now += 100  # each listing takes 100 s
        return real(prefix)

    remote.ls = slow
    report = transport.pull(progress, set(), budget=150)
    assert report.manifests == 2  # two chunk listings fit the budget
    assert report.deferred == 1
    remote.ls = real
    assert transport.pull(progress, set()).manifests == 1  # the deferred chunk was still owed


def test_the_pull_line_reports_counts_and_seconds():
    line = PullReport(listings=4, manifests=7, failed=1, deferred=2, seconds=12.4).line()
    assert line == (
        "ingest: incremental pull, 4 listings, 7 new manifests, 1 rate-limited, 2 deferred, 12 s"
    )
    assert PullReport(full=True).line().startswith("ingest: full pull")


# --- the worker -------------------------------------------------------------------


def test_the_worker_logs_each_pull_and_survives_a_failure():
    lines, errors, calls = [], [], []
    done = threading.Event()

    def pull():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("boom")
        if len(calls) == 3:
            done.set()
        return f"pull {len(calls)}"

    worker = BackgroundIngest(pull, interval=0.01, log=lines.append, on_error=errors.append)
    worker.start()
    worker.start()  # idempotent: one thread
    assert done.wait(5)
    worker.stop()
    assert len(errors) == 1
    assert lines[:2] == ["pull 2", "pull 3"]


def test_kick_wakes_the_worker_before_its_interval():
    calls = threading.Semaphore(0)

    def pull():
        calls.release()
        return "x"

    worker = BackgroundIngest(pull, interval=3600, log=lambda m: None, on_error=print)
    worker.start()
    assert calls.acquire(timeout=5)
    worker.kick()
    assert calls.acquire(timeout=5)
    worker.stop()


# --- the controller ---------------------------------------------------------------


class StuckRemote:
    """Every listing hangs (as a long 429 wait would) until released."""

    def __init__(self, inner):
        self.inner = inner
        self.release = threading.Event()
        self.entered = threading.Event()

    def ls(self, prefix):
        self.entered.set()
        self.release.wait(30)
        raise limited()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_a_cycle_submits_and_ends_while_every_ingest_listing_is_stuck(tmp_path, monkeypatch):
    c, _ = build(tmp_path, monkeypatch, workers=8, background_ingest=True)
    stuck = StuckRemote(c.transport.remote)
    c.transport.ingest_remote = stuck
    try:
        started = time.monotonic()
        report = c.cycle(NOW)
        assert time.monotonic() - started < 10  # the cycle did not wait for the ingest
        assert len(report.submitted) == 2 * len(SITES)
        assert stuck.entered.wait(5)  # ...which was running meanwhile, in the background
    finally:
        stuck.release.set()
        c.ingest.stop()


def test_background_ingest_feeds_the_cycles_own_view_of_progress(tmp_path, monkeypatch):
    c, _ = build(tmp_path, monkeypatch, workers=8, chunks=2, background_ingest=True)
    manifest = Path(tmp_path / "bucket" / "parts" / c.work_fp / "c00" / "p1.json")
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"text_sha256s": ["c00s"]}')
    try:
        c.ingest.start()
        deadline = time.monotonic() + 10
        while c.progress.done_count("c00") == 0 and time.monotonic() < deadline:
            c.ingest.kick()
            time.sleep(0.05)
        assert c.progress.done_count("c00") == 1  # seen through the cycle's own connection
    finally:
        c.ingest.stop()
