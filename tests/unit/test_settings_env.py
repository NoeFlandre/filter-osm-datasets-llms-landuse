import pytest

from landuse_filter.adapters.settings_file import load
from landuse_filter.domain.settings import SettingsError

TOML = (
    'schema_version = 1\n[hub]\nnamespace = "n"\nbucket = "n/b"\n'
    '[grid5000]\nsites = ["a"]\nwalltime_minutes = 60\nnight_walltime_minutes = 120\n'
    "night_fallback_walltime_minutes = 30\n"
    'max_jobs = 1\nmax_jobs_per_site = 1\ninterval_seconds = 5\ncuda_module = "c"\n'
)


@pytest.fixture
def toml(tmp_path):
    path = tmp_path / "luf.toml"
    path.write_text(TOML)
    return path


def test_environment_overrides_file_and_unset_keeps_it(toml):
    assert load(toml, {}).bucket == "n/b"
    ops = load(toml, {"LUF_BUCKET": "x/y", "LUF_SITES": "a, b", "LUF_MAX_JOBS": "7"})
    assert (ops.bucket, ops.sites, ops.max_jobs, ops.namespace) == ("x/y", ("a", "b"), 7, "n")
    assert load(None, {"LUF_CONFIG": str(toml)}).cuda_module == "c"


def test_environment_integer_must_parse_and_validate(toml):
    with pytest.raises(SettingsError, match="LUF_MAX_JOBS: 'x' is not an integer"):
        load(toml, {"LUF_MAX_JOBS": "x"})
    with pytest.raises(SettingsError, match="must be positive"):
        load(toml, {"LUF_MAX_JOBS": "0"})
