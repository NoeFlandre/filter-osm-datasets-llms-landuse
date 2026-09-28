from datetime import timedelta

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
