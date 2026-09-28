import pytest

from landuse_filter.adapters.settings_file import load
from landuse_filter.domain.settings import SettingsError, parse_settings


def test_repository_config_is_valid():
    ops = load()
    assert "grenoble" in ops.sites
    assert ops.bucket == "NoeFlandre/landuse-filter-work"
    assert ops.output_repo("x") == "NoeFlandre/x-landuse"


def good():
    return {
        "schema_version": 1,
        "hub": {"namespace": "n", "bucket": "n/b"},
        "grid5000": {
            "sites": ["a"],
            "walltime_minutes": 60,
            "night_walltime_minutes": 120,
            "max_jobs": 1,
            "max_jobs_per_site": 1,
            "interval_seconds": 5,
            "cuda_module": "c",
        },
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(schema_version=2), "schema_version"),
        (lambda r: r["grid5000"].update(sites=[]), "sites"),
        (lambda r: r["grid5000"].update(max_jobs=0), "positive"),
        (lambda r: r["grid5000"].update(max_jobs=True), "max_jobs"),
        (lambda r: r["hub"].pop("bucket"), "bucket"),
    ],
)
def test_invalid_settings_are_rejected(mutate, message):
    raw = good()
    mutate(raw)
    with pytest.raises(SettingsError, match=message):
        parse_settings(raw)
