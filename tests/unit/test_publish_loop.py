import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.adapters.g5k import RemoteError
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.application.publish_loop import LoopIO, read_status, run_loop, status_path
from landuse_filter.domain.publish_loop import (
    FINISHED,
    SUBMIT,
    WAIT,
    PublishStatus,
    decide,
    is_done,
    job_name,
    pause,
)


def status(**kw):
    base = {
        "dataset": "d",
        "revision": "r",
        "files_total": 3,
        "files_complete": 3,
        "files_partial": 0,
        "mirrored": True,
        "unscanned": 0,
        "stopped": None,
    }
    return PublishStatus(**{**base, **kw})


def test_done_needs_everything():
    assert status().done
    for change in (
        {"mirrored": False},
        {"unscanned": 1},
        {"files_total": 0, "files_complete": 0},
        {"files_complete": 2, "files_partial": 1},
        {"stopped": "deadline"},
    ):
        assert not status(**change).done


def test_status_json_is_machine_readable():
    out = status(files_complete=2, files_partial=1).to_json()
    assert out["files"] == {"total": 3, "complete": 2, "partial": 1, "unscanned": 0}
    assert out["done"] is False
    assert out["mirrored"] is True
    assert out["revision"] == "r"
    assert json.loads(json.dumps(out)) == out


@given(
    st.integers(0, 50),
    st.integers(0, 50),
    st.booleans(),
    st.integers(0, 3),
    st.one_of(st.none(), st.just("deadline")),
)
def test_done_implies_all_files_labelled(total, complete, mirrored, unscanned, stopped):
    s = status(
        files_total=total,
        files_complete=complete,
        mirrored=mirrored,
        unscanned=unscanned,
        stopped=stopped,
    )
    if s.done:
        assert complete == total > 0
        assert mirrored
        assert unscanned == 0
        assert stopped is None
    assert s.to_json()["done"] is s.done


def test_is_done_requires_the_same_revision_and_true():
    ok = status().to_json()
    assert is_done(ok, "r")
    assert not is_done(ok, "other")
    assert not is_done(None, "r")
    assert not is_done({"done": "yes", "revision": "r"}, "r")
    assert not is_done([1], "r")


def test_decide_table():
    assert decide(done=True, live=True) == FINISHED
    assert decide(done=True, live=False) == FINISHED
    assert decide(done=False, live=True) == WAIT
    assert decide(done=False, live=False) == SUBMIT


@given(st.booleans(), st.booleans())
def test_never_submits_while_a_job_is_live_or_after_done(done, live):
    action = decide(done=done, live=live)
    assert (action == SUBMIT) == (not done and not live)


@given(st.integers(-5, 10**6), st.floats(1, 1e4), st.floats(1, 1e5))
def test_pause_is_bounded_and_monotone(failures, interval, cap):
    p = pause(failures, interval, cap)
    assert 0 < p <= cap
    assert p >= min(interval, cap)
    assert pause(failures + 1, interval, cap) >= p


def test_job_names_keep_publish_apart_from_planning():
    assert job_name("luf-", "publish", "wiki") == "luf-publish-wiki"
    for mode in ("plan", "replan", "repair"):
        assert job_name("luf-", mode, "wiki") == "luf-plan-wiki"
    assert job_name("luf-", "publish", "x" * 40) == "luf-publish-" + "x" * 20


# --- the loop ------------------------------------------------------------------------------


class Fakes:
    def __init__(self, statuses, lives):
        self.statuses, self.lives = list(statuses), list(lives)
        self.submitted = 0
        self.sleeps: list[float] = []
        self.logs: list[str] = []

    def io(self):
        return LoopIO(self.status, self.live, self.submit, self.sleeps.append, self.logs.append)

    def status(self):
        v = self.statuses.pop(0)
        if isinstance(v, Exception):
            raise v
        return v

    def live(self):
        v = self.lives.pop(0)
        if isinstance(v, Exception):
            raise v
        return v

    def submit(self):
        self.submitted += 1
        return str(1000 + self.submitted)


DONE = status().to_json()
OPEN = status(files_complete=1).to_json()


def test_loop_submits_when_nothing_is_live_and_stops_when_done():
    f = Fakes([None, OPEN, OPEN, DONE], [False, True, False])
    assert run_loop(f.io(), "r", interval=10, cap=100) == FINISHED
    assert f.submitted == 2  # cycles 1 and 3; cycle 2 had a live job
    assert f.sleeps == [10, 10, 10]  # no sleep after the finishing check


def test_loop_survives_slow_policy_checks_and_timeouts():
    f = Fakes([OPEN, RemoteError("bucket"), OPEN, DONE], [RemoteError("frontend slow"), False])
    f.submit = lambda: (_ for _ in ()).throw(RemoteError("usagepolicycheck timed out"))
    assert run_loop(f.io(), "r", interval=10, cap=60) == FINISHED
    assert f.sleeps == [20, 40, 60]  # one failure per cycle: backoff, capped
    assert sum("retry" in m for m in f.logs) == 3


def test_loop_resets_the_backoff_after_a_good_cycle():
    f = Fakes([RemoteError("a"), OPEN, DONE], [True])
    run_loop(f.io(), "r", interval=10, cap=100)
    assert f.sleeps == [20, 10]


def test_loop_gives_up_after_max_cycles():
    f = Fakes([OPEN, OPEN], [True, True])
    assert run_loop(f.io(), "r", interval=1, cap=2, max_cycles=2) == "gave up"
    assert f.submitted == 0


def test_a_programming_error_is_not_swallowed():
    f = Fakes([ValueError("bug")], [])
    with pytest.raises(ValueError, match="bug"):
        run_loop(f.io(), "r", interval=1, cap=2)


def test_read_status_from_the_bucket(tmp_path):
    remote = DirRemote(tmp_path / "bucket")
    assert read_status(remote, "d") is None
    src = tmp_path / "s.json"
    src.write_text(json.dumps(DONE))
    remote.put([(src, status_path("d"))])
    assert read_status(remote, "d") == DONE
    src.write_text("{not json")
    remote.put([(src, status_path("d"))])
    assert read_status(remote, "d") is None
