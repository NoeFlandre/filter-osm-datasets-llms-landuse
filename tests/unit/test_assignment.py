from landuse_filter.application.assignment import Assignment, CycleReport

FULL = {
    "id": "abc",
    "name": "luf-abc",
    "site": "nancy",
    "cluster": "gres",
    "gpu": "L40S",
    "chunks": ["c0"],
    "kind": "work",
    "fp": "fp1",
    "state": "submitted",
    "job_id": "42",
    "window": 128,
    "engine_kwargs": {"a": 1},
    "sampling": {"t": 0},
    "walltime_s": 3600,
    "late_after_s": 900,
    "provenance": {"code_commit": "c"},
    "bucket": "b",
    "submitted_at": "2026-09-30T10:00:00",
    "error": "boom",
}


def test_full_assignment_round_trips():
    assert Assignment.from_json(FULL).to_json() == FULL


def test_legacy_file_without_optional_keys_loads_and_round_trips():
    legacy = {k: v for k, v in FULL.items() if k not in ("late_after_s", "kind", "window")}
    a = Assignment.from_json(legacy)
    assert (a.late_after_s, a.window, a.kind) == (None, None, "work")
    out = a.to_json()
    assert set(out) & {"late_after_s", "window"} == set()  # absent stays absent
    assert out["chunks"] == ["c0"]


def test_unknown_keys_survive_a_read_modify_write():
    a = Assignment.from_json({**FULL, "future_key": {"x": 1}})
    a.state = "ended"
    assert a.to_json()["future_key"] == {"x": 1}
    assert a.to_json()["state"] == "ended"


def test_job_id_none_is_written_explicitly_and_empty_file_loads():
    assert Assignment(id="x").to_json()["job_id"] is None
    assert Assignment.from_json({}).state == ""  # unknown state is not treated as live


def test_late_tolerance_falls_back_for_old_files():
    assert Assignment.from_json({}).late_tolerance(900.0) == 900.0
    assert Assignment.from_json({"late_after_s": 0}).late_tolerance(900.0) == 0


def test_cycle_report_json():
    r = CycleReport(3, 2, ["nancy/gres:1"])
    assert r.to_json() == {"pending_chunks": 3, "live_jobs": 2, "submitted": ["nancy/gres:1"]}
    assert CycleReport(0, 0).submitted == []
