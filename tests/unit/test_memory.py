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
