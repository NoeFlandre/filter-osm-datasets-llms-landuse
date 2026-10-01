from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.memory import ClusterMemory
from landuse_filter.domain.capacity import Cluster

NOW = datetime(2026, 9, 28, 22, tzinfo=ZoneInfo("Europe/Paris"))


def cluster(site: str, name: str) -> Cluster:
    return Cluster(site, name, "L40S", 46068, (8, 9), 2, 2, ("abaca",), exotic=False)


def test_a_back_off_ending_exactly_now_is_over(tmp_path):
    m = ClusterMemory(WorkStore(tmp_path))
    m.back_off("nancy", "gres", NOW)
    assert not m.backed_off(cluster("nancy", "gres"), NOW)
    assert m.backed_off(cluster("nancy", "gres"), NOW - timedelta(seconds=1))


def test_memory_is_per_site_and_cluster(tmp_path):
    m = ClusterMemory(WorkStore(tmp_path))
    m.back_off("nancy", "gres", NOW + timedelta(hours=1))
    m.remember_besteffort_only("nancy", "gres")
    for other in (cluster("lyon", "gres"), cluster("nancy", "grue")):
        assert not m.backed_off(other, NOW)
        assert not m.besteffort_only(other)


def test_memory_survives_a_restart(tmp_path):
    ClusterMemory(WorkStore(tmp_path)).back_off("nancy", "gres", NOW + timedelta(hours=1))
    ClusterMemory(WorkStore(tmp_path)).remember_besteffort_only("nancy", "gres")
    again = ClusterMemory(WorkStore(tmp_path))
    assert again.backed_off(cluster("nancy", "gres"), NOW)
    assert again.besteffort_only(cluster("nancy", "gres"))


def test_a_later_back_off_replaces_the_earlier_one(tmp_path):
    m = ClusterMemory(WorkStore(tmp_path))
    m.back_off("nancy", "gres", NOW + timedelta(hours=5))
    m.back_off("nancy", "gres", NOW + timedelta(minutes=5))
    assert not m.backed_off(cluster("nancy", "gres"), NOW + timedelta(minutes=10))


def _record(m, ok, now=NOW):
    return m.record_long_attempt("nancy", "gres", ok=ok, now=now, limit=3, pause=timedelta(hours=1))


def test_long_attempts_trip_after_n_consecutive_failures_and_pause_an_hour(tmp_path):
    m = ClusterMemory(WorkStore(tmp_path))
    c = cluster("nancy", "gres")
    assert [_record(m, False), _record(m, False)] == [False, False]
    assert not m.long_throttled(c, NOW)
    assert _record(m, False)
    assert m.long_throttled(c, NOW + timedelta(minutes=59))
    assert not m.long_throttled(c, NOW + timedelta(hours=1))


def test_a_success_restarts_the_failure_count(tmp_path):
    m = ClusterMemory(WorkStore(tmp_path))
    _record(m, False)
    _record(m, False)
    assert not _record(m, True)
    assert [_record(m, False), _record(m, False)] == [False, False]
    assert _record(m, False)


def test_after_a_pause_the_count_starts_again_and_is_per_cluster(tmp_path):
    m = ClusterMemory(WorkStore(tmp_path))
    for _ in range(3):
        _record(m, False)
    assert not _record(m, False, NOW + timedelta(hours=2))
    assert not m.long_throttled(cluster("lyon", "gres"), NOW)
    assert not m.long_throttled(cluster("nancy", "other"), NOW)
