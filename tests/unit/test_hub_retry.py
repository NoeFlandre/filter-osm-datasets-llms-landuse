"""Rate-limit retry of the bucket adapter and its pure decision functions (ADR-0028)."""

from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError
from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.adapters.remote import BucketRemote, rate_limit_headers, with_retry
from landuse_filter.domain.retry import (
    MAX_ATTEMPTS,
    MAX_DELAY,
    MAX_TOTAL_WAIT,
    give_up,
    next_delay,
    parse_retry_after,
)


def err(status=429, headers=None):
    request = httpx.Request("GET", "http://x")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return HfHubHTTPError("boom", response=response)


def test_parse_retry_after():
    assert parse_retry_after({"Retry-After": "46"}) == 46.0
    assert parse_retry_after({"retry-after": " 1.5 "}) == 1.5
    assert parse_retry_after({"Retry-After": "Wed, 21 Oct"}) is None
    assert parse_retry_after({"x": "1"}) is None
    assert parse_retry_after(None) is None
    assert parse_retry_after({}) is None


def test_next_delay_server_value_wins_and_exponential_fallback():
    assert next_delay(1, 46.0, 0.0) == 47.0
    assert next_delay(1, 600.0, 0.5) == MAX_DELAY
    assert next_delay(1, None, 0.0) == 1.0
    assert next_delay(3, None, 1.0) == 8.0
    assert next_delay(20, None, 0.5) == MAX_DELAY
    assert next_delay(1, -5.0, 0.0) == 1.0


@given(
    st.integers(1, 50), st.one_of(st.none(), st.floats(0, 1e6)), st.floats(0, 1, exclude_max=True)
)
def test_delay_always_bounded(attempt, retry_after, jitter):
    assert 0 <= next_delay(attempt, retry_after, jitter) <= MAX_DELAY


@given(st.integers(1, 30), st.floats(0, 3000), st.floats(0, MAX_DELAY))
def test_give_up_rule(attempt, waited, delay):
    assert give_up(attempt, waited, delay) == (
        attempt >= MAX_ATTEMPTS or waited + delay > MAX_TOTAL_WAIT
    )


def test_rate_limit_headers_hub_httpx_and_others():
    assert rate_limit_headers(err(429, {"Retry-After": "3"}))["retry-after"] == "3"
    status_error = httpx.HTTPStatusError("x", request=None, response=err(429).response)
    assert rate_limit_headers(status_error) == {}
    assert rate_limit_headers(err(500)) is None
    assert rate_limit_headers(ValueError("x")) is None
    assert rate_limit_headers(SimpleNamespace(response=None)) is None


def flaky(failures, error):
    state = {"n": 0}

    def call():
        state["n"] += 1
        if state["n"] <= failures:
            raise error
        return "ok"

    return call, state


def test_retries_after_retry_after_then_succeeds():
    call, state = flaky(2, err(429, {"Retry-After": "46"}))
    waits = []
    assert with_retry(call, sleep=waits.append, jitter=lambda: 0.0) == "ok"
    assert waits == [47.0, 47.0]
    assert state["n"] == 3


def test_other_errors_are_not_retried():
    call, state = flaky(1, err(500))
    with pytest.raises(HfHubHTTPError):
        with_retry(call, sleep=lambda s: pytest.fail("slept"))
    assert state["n"] == 1


def test_gives_up_after_max_attempts_with_original_error():
    error = err(429)
    call, state = flaky(99, error)
    waits = []
    with pytest.raises(HfHubHTTPError) as caught:
        with_retry(call, sleep=waits.append, jitter=lambda: 0.0)
    assert caught.value is error
    assert state["n"] == MAX_ATTEMPTS
    assert len(waits) == MAX_ATTEMPTS - 1


def test_gives_up_when_total_wait_exhausted():
    call, state = flaky(99, err(429, {"Retry-After": "300"}))
    waits = []
    with pytest.raises(HfHubHTTPError):
        with_retry(call, sleep=waits.append, jitter=lambda: 0.0)
    assert sum(waits) <= MAX_TOTAL_WAIT
    assert state["n"] == len(waits) + 1 < MAX_ATTEMPTS


def test_ls_retries_only_the_failing_page(monkeypatch):
    pages = {
        "E/api/buckets/u/b/tree/index%2F": ([{"type": "file", "path": "b"}], "u2"),
        "u2": ([{"type": "file", "path": "a"}, {"type": "directory", "path": "d"}], None),
    }
    calls = []
    boom = [err(429, {"Retry-After": "1"})]

    def page(self, url, params):
        calls.append(url)
        if url == "u2" and boom:
            raise boom.pop()
        return pages[url]

    monkeypatch.setattr(BucketRemote, "_page", page)
    monkeypatch.setattr(BucketRemote, "_api", lambda self: SimpleNamespace(endpoint="E"))
    monkeypatch.setattr("landuse_filter.adapters.remote.time.sleep", lambda s: None)
    assert BucketRemote("u/b").ls("index/") == ["a", "b"]
    assert calls == ["E/api/buckets/u/b/tree/index%2F", "u2", "u2"]


def test_put_get_delete_retry(monkeypatch, tmp_path):
    monkeypatch.setattr("landuse_filter.adapters.remote.time.sleep", lambda s: None)
    seen = []
    limited = [err(429)]

    class Api:
        def batch_bucket_files(self, bucket, **kw):
            if limited:
                raise limited.pop()
            seen.append(kw)

        def download_bucket_files(self, bucket, files, **kw):
            if limited:
                raise limited.pop()
            seen.append(files)

    monkeypatch.setattr(BucketRemote, "_api", lambda self: Api())
    remote = BucketRemote("u/b")
    remote.put([(tmp_path / "f", "a")])
    assert seen == [{"add": [(str(tmp_path / "f"), "a")]}]
    limited.append(err(429))
    remote.get([("a", tmp_path / "o")])
    assert seen[-1] == [("a", str(tmp_path / "o"))]
    limited.append(err(429))
    remote.delete(["a"])
    assert seen[-1] == {"delete": ["a"]}


def test_next_delay_exact_values_with_jitter():
    assert next_delay(1, 10.0, 0.5) == 11.5
    assert next_delay(2, None, 0.5) == 3.0
    assert next_delay(2, None, 0.0) == 2.0
    assert next_delay(1, 0.0, 0.0) == 1.0


def test_give_up_boundaries():
    assert not give_up(MAX_ATTEMPTS - 1, 0.0, 1.0)
    assert give_up(MAX_ATTEMPTS, 0.0, 1.0)
    assert not give_up(1, MAX_TOTAL_WAIT - 5.0, 5.0)
    assert give_up(1, MAX_TOTAL_WAIT - 5.0, 5.5)
    assert not give_up(1, 0.0, MAX_DELAY)
