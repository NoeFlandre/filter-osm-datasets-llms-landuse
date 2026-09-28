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
    assert args[:4] == ["-q", "default", "-t", "exotic"]  # regression: Lyon routes unqualified jobs
    assert "night" in args


def test_testing_only_cluster_is_not_submittable():
    granite = Cluster(
        "nancy",
        "granite",
        "RTX PRO 6000",
        97887,
        (12, 0),
        2,
        18,
        ("admin", "testing"),
        exotic=False,
    )
    assert not granite.submittable
    assert GRES.submittable


def test_besteffort():
    args = oarsub_arguments(CHUC, timedelta(hours=1), None, "n", command="c", besteffort=True)
    assert args[:2] == ["-t", "besteffort"]


def test_walltime_text():
    assert walltime_text(timedelta(minutes=5)) == "0:05"


def node(**state):
    return {"gres-1.nancy.grid5000.fr": {"hard": "alive", **state}}


def test_node_gpus_missing_counters_default_to_zero():
    assert free_gpus(GRES, node(), besteffort_counts=True) == 0
    assert free_gpus(GRES, node(free_slots=24, busy_slots=24), besteffort_counts=True) == 1
    assert free_gpus(GRES, node(busy_slots=48, freeable_slots=48), besteffort_counts=True) == 1
    assert free_gpus(GRES, node(free_slots=48), besteffort_counts=False) == 2
    assert free_gpus(GRES, node(free_slots=47, busy_slots=1), besteffort_counts=False) == 1


def test_node_gpus_freeable_counts_as_busy_without_besteffort():
    state = node(free_slots=24, freeable_slots=24)
    assert free_gpus(GRES, state, besteffort_counts=False) == 1
    assert free_gpus(GRES, state, besteffort_counts=True) == 2


def test_node_gpus_whole_gpus_only():
    assert free_gpus(GRES, node(free_slots=1, busy_slots=2), besteffort_counts=False) == 0
    assert (
        free_gpus(CHUC, {"chuc-1": {"free_slots": 3, "busy_slots": 5}}, besteffort_counts=False)
        == 1
    )


def test_reservation_starting_exactly_at_job_end_does_not_block():
    waiting = [{"state": "waiting", "scheduled_at": 150}]
    state = node(free_slots=48, reservations=waiting)
    assert free_gpus(GRES, state, besteffort_counts=False, now=50, walltime_s=100) == 2
    assert free_gpus(GRES, state, besteffort_counts=False, now=51, walltime_s=100) == 0
    assert free_gpus(GRES, state, besteffort_counts=False, walltime_s=151) == 0
    assert free_gpus(GRES, state, besteffort_counts=False, walltime_s=150) == 2


def test_no_walltime_ignores_reservations_by_default():
    state = node(free_slots=48, reservations=[{"state": "waiting", "scheduled_at": 0.5}])
    assert free_gpus(GRES, state, besteffort_counts=False) == 2


def test_oarsub_arguments_full_command_line():
    args = oarsub_arguments(CHUC, timedelta(minutes=125), "night", "luf-7", command="run.sh")
    assert args == [
        "-q",
        "default",
        "-t",
        "night",
        "-p",
        "cluster='chuc'",
        "-l",
        "host=1/gpu=1,walltime=2:05",
        "--checkpoint",
        "300",
        "-n",
        "luf-7",
        "-O",
        "luf/logs/%jobid%.out",
        "-E",
        "luf/logs/%jobid%.err",
        "run.sh",
    ]


def test_non_exotic_besteffort_job_drops_job_type():
    args = oarsub_arguments(CHUC, timedelta(hours=1), "night", "n", command="c", besteffort=True)
    assert args[:3] == ["-t", "besteffort", "-p"]


def test_walltime_text_truncates_partial_minutes():
    assert walltime_text(timedelta(seconds=119)) == "0:01"
    assert walltime_text(timedelta(hours=10, minutes=3)) == "10:03"


def test_single_slot_node():
    assert free_gpus(GRES, node(free_slots=1), besteffort_counts=False) == 2
