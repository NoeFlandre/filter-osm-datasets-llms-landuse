"""The prompt and the default settings ship inside the package (#218).

An installed wheel has no repository root beside the package, so nothing may be located by
walking up from ``__file__``. The installed-layout test runs ``luf`` code from a copy of the
package alone, in a directory that is not a checkout.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from landuse_filter.adapters.settings_file import DEFAULT, load
from landuse_filter.domain.prompting import PROMPT_SHA256

PACKAGE = Path(__file__).parents[2] / "src" / "landuse_filter"


def test_installed_package_reads_its_prompt_and_settings_without_a_checkout(tmp_path):
    site = tmp_path / "site-packages"
    shutil.copytree(PACKAGE, site / "landuse_filter", ignore=shutil.ignore_patterns("__pycache__"))
    elsewhere = tmp_path / "not-a-checkout"
    elsewhere.mkdir()
    code = textwrap.dedent(
        """
        import hashlib

        import landuse_filter
        from landuse_filter.cli import OPS, PROMPT

        print(landuse_filter.__file__)
        print(hashlib.sha256(PROMPT.read_bytes()).hexdigest())
        print(OPS.sites[0])
        """
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("LUF_")}
    env["PYTHONPATH"] = str(site)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=elsewhere,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    imported, digest, first_site = result.stdout.splitlines()
    assert Path(imported).is_relative_to(site)
    assert digest == PROMPT_SHA256
    assert first_site == load(environ={}).sites[0]


def test_default_settings_load_without_luf_config_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    ops = load(environ={})

    assert ops.namespace == "NoeFlandre"
    assert ops.sites == load(DEFAULT, environ={}).sites


def test_luf_config_still_overrides_the_packaged_defaults(tmp_path):
    override = tmp_path / "mine.toml"
    override.write_text(
        DEFAULT.read_text(encoding="utf-8").replace(
            'namespace = "NoeFlandre"', 'namespace = "other"'
        ),
        encoding="utf-8",
    )

    ops = load(environ={"LUF_CONFIG": str(override)})

    assert ops.namespace == "other"
    assert ops.sites == load(environ={}).sites
