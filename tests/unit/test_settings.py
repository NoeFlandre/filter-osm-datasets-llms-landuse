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


def test_policy_check_defaults_validates_and_reads_environment():
    from landuse_filter.adapters.settings_file import with_environment

    assert parse_settings(good()).policy_check == "per-job"
    raw = good()
    raw["grid5000"]["policy_check"] = "per-batch"
    assert parse_settings(raw).policy_check == "per-batch"
    env = with_environment(good(), {"LUF_POLICY_CHECK": "per-batch"})
    assert parse_settings(env).policy_check == "per-batch"
    raw["grid5000"]["policy_check"] = "sometimes"
    with pytest.raises(SettingsError, match="'policy_check' must be one of per-job, per-batch"):
        parse_settings(raw)


def test_submit_workers_defaults_validates_and_reads_environment():
    from landuse_filter.adapters.settings_file import with_environment

    assert parse_settings(good()).submit_workers == 1
    raw = good()
    raw["grid5000"]["submit_workers"] = 8
    assert parse_settings(raw).submit_workers == 8
    raw["grid5000"]["submit_workers"] = 1
    assert parse_settings(raw).submit_workers == 1  # boundary
    env = with_environment(good(), {"LUF_SUBMIT_WORKERS": "4"})
    assert parse_settings(env).submit_workers == 4
    raw["grid5000"]["submit_workers"] = 0
    with pytest.raises(SettingsError, match="'submit_workers' must be positive"):
        parse_settings(raw)


def test_controller_settings_defaults_equal_the_shipped_luf_toml():
    from datetime import timedelta

    from landuse_filter.adapters.settings_file import DEFAULT, load
    from landuse_filter.application.controller import Settings

    ops = load(DEFAULT, environ={})
    s = Settings(datasets=[], sites=[])
    assert s.max_jobs_total == ops.max_jobs
    assert s.max_jobs_per_site == ops.max_jobs_per_site
    assert s.walltime == timedelta(minutes=ops.walltime_minutes)
    assert s.night_walltime == timedelta(minutes=ops.night_walltime_minutes)
    assert s.night_fallback_walltime == timedelta(minutes=ops.night_fallback_walltime_minutes)
    assert s.bucket == ops.bucket
    assert s.day_long_max_failures == ops.day_long_max_failures
    assert s.chunk_overflow == ops.chunk_overflow
    assert s.submit_workers == ops.submit_workers
    assert s.policy_check == ops.policy_check


def test_settings_from_ops_takes_its_defaults_from_the_ops_and_lets_overrides_win():
    from datetime import timedelta

    from landuse_filter.adapters.settings_file import DEFAULT, load
    from landuse_filter.application.controller import Settings

    ops = load(DEFAULT, environ={})
    s = Settings.from_ops(ops, ["benchmark"], besteffort=True)
    assert s.sites == list(ops.sites)
    assert s.datasets == ["benchmark"]
    assert s.besteffort is True
    assert s.walltime == timedelta(minutes=ops.walltime_minutes)
    assert s.day_walltime is None


OPTIONAL_FIELDS = {
    "day_walltime_minutes": ("int", None, 90),
    "day_long_max_failures": ("int", 3, 5),
    "chunk_overflow": ("float", 1.2, 1.5),
    "policy_check": ("str", "per-job", "per-batch"),
    "submit_workers": ("int", 1, 4),
}


@pytest.mark.parametrize("name", list(OPTIONAL_FIELDS))
def test_optional_field_is_described_by_the_table(name):
    from landuse_filter.domain.settings import FIELDS

    kind, default, _ = OPTIONAL_FIELDS[name]
    (f,) = [f for f in FIELDS if f.name == name]
    assert (f.table, f.kind, f.required, f.default) == ("grid5000", kind, False, default)
    assert type(f.default) is type(default)


@pytest.mark.parametrize("name", list(OPTIONAL_FIELDS))
def test_optional_field_defaults_when_missing_and_reads_the_value_when_present(name):
    _, default, value = OPTIONAL_FIELDS[name]
    assert getattr(parse_settings(good()), name) == default
    raw = good()
    raw["grid5000"][name] = value
    assert getattr(parse_settings(raw), name) == value


@pytest.mark.parametrize(
    ("name", "bad", "message"),
    [
        ("day_walltime_minutes", 0, "luf.toml: 'day_walltime_minutes' must be positive"),
        ("day_walltime_minutes", "x", "luf.toml: 'day_walltime_minutes' must be int"),
        ("day_walltime_minutes", True, "luf.toml: 'day_walltime_minutes' must be int"),
        ("day_long_max_failures", 0, "luf.toml: 'day_long_max_failures' must be positive"),
        ("day_long_max_failures", None, "luf.toml: 'day_long_max_failures' must be int"),
        ("submit_workers", -1, "luf.toml: 'submit_workers' must be positive"),
        ("submit_workers", 1.5, "luf.toml: 'submit_workers' must be int"),
        ("chunk_overflow", 0.5, "luf.toml: 'chunk_overflow' must be at least 1.0"),
        ("chunk_overflow", "x", "luf.toml: 'chunk_overflow' must be float"),
        ("chunk_overflow", True, "luf.toml: 'chunk_overflow' must be float"),
        ("policy_check", "never", "luf.toml: 'policy_check' must be one of"),
        ("policy_check", None, "luf.toml: 'policy_check' must be one of"),
    ],
)
def test_optional_field_present_but_invalid_is_rejected_not_defaulted(name, bad, message):
    raw = good()
    raw["grid5000"][name] = bad
    with pytest.raises(SettingsError, match=message):
        parse_settings(raw)


def test_optional_fields_are_coerced_from_the_environment_by_kind():
    from landuse_filter.adapters.settings_file import ENV_FIELDS, with_environment

    for name in OPTIONAL_FIELDS:
        assert ENV_FIELDS[name] == "grid5000"
    env = {
        "LUF_DAY_WALLTIME_MINUTES": "90",
        "LUF_CHUNK_OVERFLOW": "1.5",
        "LUF_POLICY_CHECK": "per-batch",
    }
    g5k = with_environment(good(), env)["grid5000"]
    assert g5k["day_walltime_minutes"] == 90
    assert g5k["chunk_overflow"] == 1.5
    assert g5k["policy_check"] == "per-batch"
    with pytest.raises(SettingsError, match="LUF_SUBMIT_WORKERS: 'x' is not an integer"):
        with_environment(good(), {"LUF_SUBMIT_WORKERS": "x"})
    with pytest.raises(SettingsError, match="LUF_CHUNK_OVERFLOW: 'x' is not a number"):
        with_environment(good(), {"LUF_CHUNK_OVERFLOW": "x"})
