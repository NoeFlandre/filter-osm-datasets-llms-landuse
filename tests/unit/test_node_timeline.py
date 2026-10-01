from datetime import datetime

from landuse_filter.application.node_timeline import Timeline


def test_stamps_follow_the_monotonic_clock_from_the_anchor():
    tl = Timeline()
    anchor = datetime.fromisoformat(tl.at(tl.started_mono))
    later = datetime.fromisoformat(tl.at(tl.started_mono + 90.0))
    assert (later - anchor).total_seconds() == 90
    assert datetime.fromisoformat(tl.at(tl.started_mono - 60.0)) < anchor
    assert tl.at(None) is None
    assert anchor.utcoffset().total_seconds() == 0
    assert datetime.fromisoformat(tl.now()) >= anchor


def test_env_seconds_and_record_keys():
    from landuse_filter.application.node_timeline import env_ready_seconds, timeline_record

    assert env_ready_seconds({"LUF_ENV_READY_SECONDS": "95"}) == 95
    assert env_ready_seconds({"LUF_ENV_READY_SECONDS": "x"}) is None
    assert env_ready_seconds({}) is None
    tl = Timeline()
    rec = timeline_record(
        tl,
        engine_ready=tl.started_mono + 60,
        first_result=None,
        last_result=tl.started_mono + 100,
        walltime_s=1800,
        assigned_texts=4000,
        environ={"LUF_ENV_READY_SECONDS": "30"},
    )
    assert set(rec) == {
        "started_at",
        "env_ready_seconds",
        "engine_ready_at",
        "first_result_at",
        "last_result_at",
        "ended_at",
        "walltime_s",
        "assigned_texts",
    }
    assert rec["first_result_at"] is None
    assert rec["env_ready_seconds"] == 30
    assert (rec["walltime_s"], rec["assigned_texts"]) == (1800, 4000)
