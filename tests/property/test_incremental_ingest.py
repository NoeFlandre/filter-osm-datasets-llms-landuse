from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.unit.test_incremental_ingest import setup, upload

CHUNKS = ["c1", "c2", "c3"]
STEP = st.one_of(
    st.tuples(st.sampled_from(["upload", "start", "end"]), st.sampled_from(CHUNKS)),
    st.tuples(st.just("pull"), st.sampled_from(["0", "180", "900", "4000"])),
    st.tuples(st.just("restart"), st.just("")),
)


@settings(max_examples=60, deadline=None)
@given(st.lists(STEP, max_size=40))
def test_no_part_is_lost_or_double_counted_under_any_schedule(steps):
    """Parts of a chunk appear only while its assignment lives; any schedule ingests them all."""
    with TemporaryDirectory() as tmp:
        transport, progress, remote, clock = setup(Path(tmp))
        live: set[str] = set()
        uploaded: dict[str, set[str]] = {c: set() for c in CHUNKS}
        n = 0
        for kind, arg in steps:
            if kind == "start":
                live.add(arg)
                transport.track({arg})
            elif kind == "upload" and arg in live:
                n += 1
                uploaded[arg].add(f"s{n}")
                upload(remote, arg, f"p{n}", [f"s{n}"])
            elif kind == "end":
                live.discard(arg)
            elif kind == "pull":
                clock.now += int(arg)
                transport.pull(progress, set(live))
            elif kind == "restart":
                transport, progress, remote, clock = setup(Path(tmp), clock)
        live.clear()
        transport.pull(progress, set(live))
        transport.pull(progress, set(live))  # idempotent
        for chunk in CHUNKS:
            assert progress.index("fp").shas(chunk) == uploaded[chunk]


STEP_429 = st.one_of(
    st.tuples(st.sampled_from(["upload", "start", "end"]), st.sampled_from(CHUNKS)),
    st.tuples(st.sampled_from(["pull", "limited"]), st.sampled_from(["0", "180", "4000"])),
)


@settings(max_examples=60, deadline=None)
@given(st.lists(STEP_429, max_size=40))
def test_rate_limited_pulls_never_lose_or_duplicate_parts(steps):
    """A pull that hits a rate limit gives up; a later pull still ingests everything once."""
    from tests.unit.test_background_ingest import limited

    with TemporaryDirectory() as tmp:
        transport, progress, remote, clock = setup(Path(tmp))
        real = remote.ls
        live: set[str] = set()
        uploaded: dict[str, set[str]] = {c: set() for c in CHUNKS}
        n = 0
        for kind, arg in steps:
            if kind == "start":
                live.add(arg)
                transport.track({arg})
            elif kind == "upload" and arg in live:
                n += 1
                uploaded[arg].add(f"s{n}")
                upload(remote, arg, f"p{n}", [f"s{n}"])
            elif kind == "end":
                live.discard(arg)
            elif kind in ("pull", "limited"):
                clock.now += int(arg)

                def fail(prefix):
                    raise limited()

                remote.ls = fail if kind == "limited" else real
                transport.pull(progress, set(live))
        remote.ls = real
        live.clear()
        clock.now += 4000
        transport.pull(progress, set())
        transport.pull(progress, set())
        for chunk in CHUNKS:
            assert progress.index("fp").shas(chunk) == uploaded[chunk]
