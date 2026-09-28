from datetime import timedelta

from landuse_filter.domain.capacity import Cluster, free_gpus, oarsub_arguments, walltime_text

GRES = Cluster("nancy", "gres", "L40S", 46068, (8, 9), 2, 7, ("abaca", "production"), exotic=False)
CHUC = Cluster("lille", "chuc", "A100", 40960, (8, 0), 4, 8, ("default",), exotic=False)


def test_free_gpus_from_slots():
    nodes = {
        "gres-1.nancy.grid5000.fr": {
            "hard": "alive",
            "free_slots": 24,
            "freeable_slots": 0,
            "busy_slots": 24,
        },
        "gres-2.nancy.grid5000.fr": {
            "hard": "alive",
            "free_slots": 0,
            "freeable_slots": 48,
            "busy_slots": 0,
        },
        "gres-3.nancy.grid5000.fr": {
            "hard": "dead",
            "free_slots": 48,
            "freeable_slots": 0,
            "busy_slots": 0,
        },
        "grat-1.nancy.grid5000.fr": {
            "hard": "alive",
            "free_slots": 64,
            "freeable_slots": 0,
            "busy_slots": 0,
        },
    }
    assert free_gpus(GRES, nodes, besteffort_counts=False) == 1
    assert free_gpus(GRES, nodes, besteffort_counts=True) == 3


def test_production_cluster_uses_abaca_without_night_type():
    args = oarsub_arguments(GRES, timedelta(hours=1), "night", "luf-1", command="run.sh")
    assert args[:2] == ["-q", "abaca"]
    assert "night" not in args
    assert "host=1/gpu=1,walltime=1:00" in args
    assert args[-1] == "run.sh"


def test_default_cluster_gets_night_type_and_exotic():
    exotic = Cluster("lyon", "sirius", "A100", 40960, (8, 0), 8, 1, ("default",), exotic=True)
    args = oarsub_arguments(exotic, timedelta(minutes=90), "night", "n", command="c")
    assert args[0:2] == ["-t", "exotic"]
    assert "night" in args
    assert "-q" not in args


def test_besteffort():
    args = oarsub_arguments(CHUC, timedelta(hours=1), None, "n", command="c", besteffort=True)
    assert args[:2] == ["-t", "besteffort"]


def test_walltime_text():
    assert walltime_text(timedelta(minutes=5)) == "0:05"
