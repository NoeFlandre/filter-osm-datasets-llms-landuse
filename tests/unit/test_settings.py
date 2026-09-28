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
        (lambda r: r.update(schema_version=2), "luf.toml: schema_version must be 1"),
        (lambda r: r.pop("schema_version"), "luf.toml: schema_version must be 1"),
        (lambda r: r.pop("hub"), r"luf.toml: missing \[hub\] table"),
        (lambda r: r.pop("grid5000"), r"luf.toml: missing \[grid5000\] table"),
        (lambda r: r["grid5000"].update(sites=[]), "luf.toml: 'sites' must be a non-empty list"),
        (lambda r: r["grid5000"].update(sites=[""]), "luf.toml: 'sites' must be a non-empty list"),
        (lambda r: r["grid5000"].update(sites=[3]), "luf.toml: 'sites' must be a non-empty list"),
        (lambda r: r["grid5000"].update(sites="a"), "luf.toml: 'sites' must be list"),
        (lambda r: r["grid5000"].update(max_jobs=True), "luf.toml: 'max_jobs' must be int"),
        (lambda r: r["grid5000"].update(cuda_module=1), "luf.toml: 'cuda_module' must be str"),
        (lambda r: r["hub"].pop("bucket"), "luf.toml: 'bucket' must be str"),
        (lambda r: r["hub"].pop("namespace"), "luf.toml: 'namespace' must be str"),
    ],
)
def test_invalid_settings_are_rejected(mutate, message):
    raw = good()
    mutate(raw)
    with pytest.raises(SettingsError, match=f"^{message}"):
        parse_settings(raw)


@pytest.mark.parametrize(
    "key",
    [
        "walltime_minutes",
        "night_walltime_minutes",
        "max_jobs",
        "max_jobs_per_site",
        "interval_seconds",
    ],
)
def test_every_number_must_be_at_least_one(key):
    raw = good()
    raw["grid5000"][key] = 0
    with pytest.raises(SettingsError, match=f"^luf.toml: '{key}' must be positive$"):
        parse_settings(raw)
    raw["grid5000"][key] = 1
    assert getattr(parse_settings(raw), key) == 1


def test_all_fields_are_read():
    ops = parse_settings(good())
    assert (ops.namespace, ops.bucket, ops.sites, ops.cuda_module) == ("n", "n/b", ("a",), "c")
    assert (
        ops.walltime_minutes,
        ops.night_walltime_minutes,
        ops.max_jobs,
        ops.max_jobs_per_site,
        ops.interval_seconds,
    ) == (60, 120, 1, 1, 5)
