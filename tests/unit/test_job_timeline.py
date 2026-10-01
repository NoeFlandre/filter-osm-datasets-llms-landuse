import pytest

from landuse_filter.domain.job_timeline import (
    generation_seconds,
    is_measured,
    seconds_left,
    stopped_early,
    useful_fraction,
)


def job(**kw):
    base = {
        "walltime_s": 1000,
        "started_at": "2026-10-01T10:00:00+00:00",
        "engine_ready_at": "2026-10-01T10:02:00+00:00",
        "last_result_at": "2026-10-01T10:10:00+00:00",
        "ended_at": "2026-10-01T10:10:30+00:00",
        "env_ready_seconds": 100,
        "stopped": False,
        "gpu_key": "l40s",
    }
    return {**base, **kw}


def test_generation_seconds_and_measured():
    assert generation_seconds(job()) == 480
    assert generation_seconds(job(last_result_at="2026-10-01T10:01:00+00:00")) == 0
    assert generation_seconds({"walltime_s": 5}) is None
    assert generation_seconds(job(engine_ready_at=None)) is None
    assert is_measured(job())
    assert not is_measured(job(walltime_s=0))
    assert not is_measured({"gpu": "x"})


def test_seconds_left_includes_env_setup():
    assert seconds_left(job()) == 1000 - 630 - 100
    assert seconds_left({"walltime_s": 10}) is None
    assert seconds_left(job(env_ready_seconds=None)) == 370


def test_seconds_left_without_env_seconds():
    j = job()
    del j["env_ready_seconds"]
    assert seconds_left(j) == 370


def test_stopped_early_needs_unsignalled_and_more_than_ten_percent_left():
    assert stopped_early(job())  # 270 s left of 1000
    assert not stopped_early(job(stopped=True))
    assert not stopped_early(job(env_ready_seconds=270))  # exactly 10 % left
    assert stopped_early(job(env_ready_seconds=269))
    assert not stopped_early({"walltime_s": 10})


def test_useful_fraction_overall_and_by_gpu_ignores_old_records():
    jobs = [
        job(),
        job(stopped=True, gpu_key="h100", last_result_at="2026-10-01T10:12:00+00:00"),
        {"gpu": "L40S", "sentences_per_second": 2.0},  # old record
    ]
    out = useful_fraction(jobs)
    assert out["overall"] == {
        "jobs": 2,
        "useful_fraction": pytest.approx((480 + 600) / 2000, abs=1e-3),
        "stopped_early": 0.5,
        "ran_to_checkpoint": 0.5,
    }
    assert list(out["by_gpu"]) == ["h100", "l40s"]
    assert out["by_gpu"]["l40s"]["useful_fraction"] == 0.48
    assert out["by_gpu"]["h100"]["stopped_early"] == 0.0
    assert out["by_gpu"]["h100"]["ran_to_checkpoint"] == 1.0


def test_useful_fraction_falls_back_to_gpu_name_and_handles_nothing():
    out = useful_fraction([job(gpu_key=None, gpu="L40S"), job(gpu_key=None)])
    assert list(out["by_gpu"]) == ["L40S", "unknown"]
    empty = useful_fraction([])
    assert empty == {
        "overall": {
            "jobs": 0,
            "useful_fraction": None,
            "stopped_early": None,
            "ran_to_checkpoint": None,
        },
        "by_gpu": {},
    }


def test_is_measured_boundaries():
    no_wall = job()
    del no_wall["walltime_s"]
    assert not is_measured(no_wall)
    assert is_measured(job(walltime_s=1))


def test_group_values_are_rounded_to_three_places_and_zero_generation_counts_zero():
    jobs = [
        job(),  # 480 s of 1000, early
        job(stopped=True, last_result_at="2026-10-01T10:02:00+00:00"),  # 0 s of 1000
        job(stopped=True, last_result_at="2026-10-01T10:02:00+00:00"),
    ]
    overall = useful_fraction(jobs)["overall"]
    assert overall["jobs"] == 3
    assert overall["useful_fraction"] == 0.16  # 480 / 3000, no extra second for idle jobs
    assert overall["stopped_early"] == 0.333
    assert overall["ran_to_checkpoint"] == 0.667
    third = useful_fraction([job(last_result_at="2026-10-01T10:02:00+00:00") for _ in range(3)])
    assert third["overall"]["useful_fraction"] == 0.0
    odd = useful_fraction([job(walltime_s=700), job(walltime_s=700, stopped=True)])
    assert odd["overall"]["useful_fraction"] == 0.686  # 960 / 1400
