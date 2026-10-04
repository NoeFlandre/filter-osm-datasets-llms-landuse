import os
import subprocess
import sys
from pathlib import Path

from landuse_filter.adapters import g5k
from landuse_filter.cli import g5k as g5k_cli


def test_inventory_command_reads_the_helper_before_mocked_transport(monkeypatch, tmp_path):
    helper = Path(g5k_cli.__file__).parents[1] / "adapters" / "frontend" / "inventory.py"
    calls = []

    def offline_ssh(site, command, *, timeout=120, stdin=None):
        calls.append((site, command, timeout, stdin))
        return "[]"

    monkeypatch.setattr(g5k, "ssh", offline_ssh)

    g5k_cli.g5k_inventory(work=tmp_path, site="nancy")

    assert calls == [("nancy", "python3 -", 600, helper.read_bytes())]
    assert (tmp_path / "inventory.json").read_text(encoding="utf-8") == "[]"


def test_installed_inventory_command_uses_its_packaged_helper_offline(tmp_path):
    startup_dir = tmp_path / "startup"
    startup_dir.mkdir()
    (startup_dir / "sitecustomize.py").write_text(
        "from landuse_filter.adapters import g5k\n"
        "def offline_ssh(site, command, *, timeout=120, stdin=None):\n"
        "    if site != 'nancy' or command != 'python3 -' or not stdin:\n"
        "        raise RuntimeError('unexpected transport request')\n"
        "    if b'Reference API' not in stdin:\n"
        "        raise RuntimeError('inventory helper content is missing')\n"
        "    return '[]'\n"
        "g5k.ssh = offline_ssh\n",
        encoding="utf-8",
    )
    executable = Path(sys.executable).with_name("luf")
    assert executable.is_file(), "the luf console script must be installed for this test"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(startup_dir), environment.get("PYTHONPATH")])
    )

    result = subprocess.run(
        [str(executable), "g5k", "inventory", "--work", str(tmp_path / "work"), "--site", "nancy"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "work" / "inventory.json").read_text(encoding="utf-8") == "[]"
