from datetime import timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.domain.gpu import GpuSpec, Profile, gpu_key, ineligibility
from landuse_filter.domain.scheduling import Slot, assign_chunks, rank_slots, useful_sentences


def test_eligibility():
    assert ineligibility(GpuSpec("A100", (8, 0), 40960)) is None
    assert "compute capability" in ineligibility(GpuSpec("V100", (7, 0), 32768))
    assert "MiB" in ineligibility(GpuSpec("RTX 3070", (8, 6), 8192))


def test_gpu_key():
    assert gpu_key("NVIDIA L40S") == "l40s"
    assert gpu_key("NVIDIA RTX A6000") == "rtx_a6000"


def test_profile_engine_args():
    assert Profile("x", cuda_graph_max_bs=64).engine_args()["cuda_graph_max_bs"] == 64
    assert "cuda_graph_max_bs" not in Profile("x").engine_args()


def slot(site, sps, wait=0, wall=60, gpus=1):
    return Slot(
        site, "c", "g", gpus, 1, timedelta(minutes=wait), timedelta(minutes=wall), None, sps
    )


def test_rank_prefers_useful_work_and_drops_useless():
    setup = timedelta(minutes=10)
    ranked = rank_slots([slot("a", 1.0), slot("b", 2.0), slot("c", 9.0, wall=5)], setup)
    assert [s.site for s in ranked] == ["b", "a"]
    assert (
        useful_sentences(slot("a", 1.0, wait=60), setup)
        == useful_sentences(slot("a", 1.0), setup) / 2
    )


def test_assign_skips_taken_and_respects_capacity():
    pending = [("a", 10), ("b", 10), ("c", 10), ("d", 10)]
    assert assign_chunks(pending, {"a"}, capacity=20, overflow=1.0) == ["b", "c"]
    assert assign_chunks([("big", 100)], set(), capacity=1) == ["big"]  # never starve a job


def test_memory_exactly_at_minimum_is_eligible():
    assert ineligibility(GpuSpec("T", (8, 0), 16 * 1024)) is None
    assert ineligibility(GpuSpec("T", (8, 0), 16 * 1024 - 1)) == "16383 MiB < 16384 MiB"


def test_gpu_key_turns_dashes_into_underscores():
    assert gpu_key("NVIDIA A100-SXM4-40GB") == "a100_sxm4_40gb"


def test_profile_engine_args_exact():
    assert Profile("x").engine_args() == {"max_running_requests": 64, "mem_fraction_static": 0.75}


def test_useful_sentences_exact_value():
    s = slot("a", 2.0, wait=60, wall=70, gpus=3)
    assert useful_sentences(s, timedelta(minutes=10)) == 2.0 * 3 * 3600 * 0.5


def test_rank_keeps_slots_with_less_than_one_useful_sentence():
    tiny = slot("a", 1e-4, wall=11)
    assert rank_slots([tiny], timedelta(minutes=10)) == [tiny]


def test_assign_budget_is_capacity_times_overflow():
    pending = [("a", 10), ("b", 10), ("c", 10)]
    assert assign_chunks(pending, set(), capacity=10, overflow=2.0) == ["a", "b"]


def _slot_args(**over):
    from datetime import timedelta

    base = {
        "site": "lyon",
        "cluster": "sirius",
        "gpu": "A100",
        "free": 3,
        "walltime": timedelta(minutes=30),
        "job_type": None,
        "sentences_per_second": 8.0,
        "besteffort": False,
        "queue_room": True,
        "queued_wait": timedelta(hours=1),
    }
    return {**base, **over}


def test_a_free_gpu_makes_an_immediate_slot_with_all_free_nodes():
    from datetime import timedelta

    from landuse_filter.domain.scheduling import Slot, slot_for

    assert slot_for(**_slot_args(free=3, job_type="night")) == Slot(
        "lyon", "sirius", "A100", 1, 3, timedelta(0), timedelta(minutes=30), "night", 8.0
    )


def test_no_free_gpu_makes_a_queued_slot_only_if_the_site_has_queue_room():
    from datetime import timedelta

    from landuse_filter.domain.scheduling import Slot, slot_for

    assert slot_for(**_slot_args(free=0)) == Slot(
        "lyon",
        "sirius",
        "A100",
        1,
        1,
        timedelta(hours=1),
        timedelta(minutes=30),
        None,
        8.0,
        queued=True,
    )
    assert slot_for(**_slot_args(free=0, queue_room=False)) is None


def test_besteffort_only_clusters_never_queue():
    from landuse_filter.domain.scheduling import slot_for

    assert slot_for(**_slot_args(free=0, besteffort=True)) is None
    slot = slot_for(**_slot_args(free=2, besteffort=True))
    assert (slot.free_nodes, slot.besteffort, slot.queued) == (2, True, False)


def test_no_walltime_means_no_slot():
    from landuse_filter.domain.scheduling import slot_for

    assert slot_for(**_slot_args(walltime=None)) is None


def test_exactly_one_free_gpu_is_a_free_slot_not_a_queued_one():
    from landuse_filter.domain.scheduling import slot_for

    slot = slot_for(**_slot_args(free=1))
    assert (slot.free_nodes, slot.queued, slot.wait.total_seconds()) == (1, False, 0)


def _ladder(window=600, night=True, day=60, preferred=120, fallback=30):
    from landuse_filter.domain.scheduling import walltime_ladder

    m = timedelta
    return tuple(
        int(w.total_seconds() // 60)
        for w in walltime_ladder(
            window_max=m(minutes=window),
            night=night,
            day=m(minutes=day),
            preferred=m(minutes=preferred),
            fallback=m(minutes=fallback),
        )
    )


def test_night_ladder_is_long_then_fallback():
    assert _ladder() == (120, 30)


def test_night_ladder_is_capped_by_the_window():
    assert _ladder(window=90) == (90, 30)
    assert _ladder(window=20) == (20,)
    assert _ladder(window=0) == ()


def test_night_fallback_never_exceeds_the_preferred_walltime():
    assert _ladder(preferred=20, fallback=30) == (20,)
    assert _ladder(preferred=30, fallback=30) == (30,)


def test_day_ladder_is_single_and_ignores_night_values():
    assert _ladder(window=60, night=False, day=60) == (60,)
    assert _ladder(window=45, night=False, day=60) == (45,)


def test_slot_for_carries_the_fallbacks():
    from landuse_filter.domain.scheduling import slot_for

    s = slot_for(
        site="a",
        cluster="c",
        gpu="g",
        free=1,
        walltime=timedelta(hours=2),
        job_type="night",
        sentences_per_second=1.0,
        besteffort=False,
        queue_room=False,
        queued_wait=timedelta(0),
        fallbacks=(timedelta(minutes=30),),
    )
    assert s.fallbacks == (timedelta(minutes=30),)


def _day(window=600, day=60, short=30):
    from landuse_filter.domain.scheduling import walltime_ladder

    m = timedelta
    return tuple(
        int(w.total_seconds() // 60)
        for w in walltime_ladder(
            window_max=m(minutes=window),
            night=False,
            day=m(minutes=day),
            preferred=m(minutes=120),
            fallback=m(minutes=30),
            day_short=m(minutes=short),
        )
    )


def test_day_ladder_is_long_then_short():
    assert _day() == (60, 30)


def test_day_ladder_with_equal_values_is_the_single_current_rung():
    assert _day(day=30, short=30) == (30,)


def test_day_ladder_is_capped_by_the_window_and_short_never_exceeds_long():
    assert _day(window=45) == (45, 30)
    assert _day(window=20) == (20,)
    assert _day(day=20, short=30) == (20,)
    assert _day(window=0) == ()


def test_without_long_keeps_only_the_shortest_rung():
    from landuse_filter.domain.scheduling import long_attempt_tripped, without_long

    m = timedelta
    assert without_long((m(minutes=60), m(minutes=30))) == (m(minutes=30),)
    assert without_long(()) == ()
    assert without_long((m(minutes=90), m(minutes=60), m(minutes=30))) == (m(minutes=30),)
    assert not long_attempt_tripped(2, 3)
    assert long_attempt_tripped(3, 3)
    assert (long_attempt_tripped(2), long_attempt_tripped(3)) == (False, True)


@given(
    window=st.integers(0, 1000),
    day=st.integers(1, 200),
    short=st.integers(1, 200),
)
def test_day_ladder_properties(window, day, short):
    ladder = _day(window=window, day=day, short=short)
    assert all(0 < w <= window for w in ladder)
    assert all(w <= day for w in ladder)
    assert list(ladder) == sorted(set(ladder), reverse=True)
    assert len(ladder) <= 2


def test_default_overflow_is_twenty_percent():
    pending = [("a", 6), ("b", 6), ("c", 1)]
    assert assign_chunks(pending, set(), capacity=10) == ["a", "b"]  # budget 12, not 13


def _stale(**kw):
    from landuse_filter.domain.scheduling import is_stale_besteffort

    base = {"queue": "besteffort", "state": "Waiting", "submitted_at": 0.0, "now": 1200.0}
    return is_stale_besteffort(**{**base, **kw})


def test_stale_besteffort_boundary_is_the_threshold():
    assert _stale(now=1200.0)
    assert not _stale(now=1199.9)
    assert not _stale(now=1200.0, max_wait=timedelta(minutes=21))


@pytest.mark.parametrize("queue", ["abaca", "default", "night", "exotic", ""])
def test_stale_never_for_other_queues(queue):
    assert not _stale(queue=queue, now=10**9)


@pytest.mark.parametrize("state", ["Running", "Launching", "Hold", "Unknown"])
def test_stale_never_when_not_waiting(state):
    assert not _stale(state=state, now=10**9)


def test_stale_never_without_submission_time():
    assert not _stale(submitted_at=None, now=10**9)
