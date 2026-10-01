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
            "night_fallback_walltime_minutes": 30,
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
        "night_fallback_walltime_minutes",
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
        ops.night_fallback_walltime_minutes,
        ops.max_jobs,
        ops.max_jobs_per_site,
        ops.interval_seconds,
    ) == (60, 120, 30, 1, 1, 5)


def test_day_walltime_settings_are_optional_and_validated():
    ops = parse_settings(good())
    assert (ops.day_walltime_minutes, ops.day_long_max_failures) == (None, 3)
    raw = good()
    raw["grid5000"].update(day_walltime_minutes=60, day_long_max_failures=5)
    ops = parse_settings(raw)
    assert (ops.day_walltime_minutes, ops.day_long_max_failures) == (60, 5)
    raw["grid5000"]["day_walltime_minutes"] = 0
    with pytest.raises(SettingsError, match="'day_walltime_minutes' must be positive"):
        parse_settings(raw)


def test_chunk_overflow_defaults_validates_and_reads_environment():
    from landuse_filter.adapters.settings_file import with_environment

    assert parse_settings(good()).chunk_overflow == 1.2
    raw = good()
    raw["grid5000"]["chunk_overflow"] = 1.6
    assert parse_settings(raw).chunk_overflow == 1.6
    raw["grid5000"]["chunk_overflow"] = 1
    assert parse_settings(raw).chunk_overflow == 1.0  # boundary is allowed
    raw["grid5000"]["chunk_overflow"] = 1.6
    env = with_environment(good(), {"LUF_CHUNK_OVERFLOW": "2"})
    assert parse_settings(env).chunk_overflow == 2.0
    for bad, message in (
        (0.9, "must be at least 1.0"),
        ("x", "must be float"),
        (True, "must be float"),
    ):
        raw["grid5000"]["chunk_overflow"] = bad
        with pytest.raises(SettingsError, match=f"luf.toml: 'chunk_overflow' {message}$"):
            parse_settings(raw)
    with pytest.raises(SettingsError, match="LUF_CHUNK_OVERFLOW"):
        with_environment(good(), {"LUF_CHUNK_OVERFLOW": "abc"})
