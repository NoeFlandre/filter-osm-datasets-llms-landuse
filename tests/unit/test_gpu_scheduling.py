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
