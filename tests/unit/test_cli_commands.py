"""Typer CliRunner tests: option parsing, defaults, exit codes; use cases are stubbed."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
from typer.testing import CliRunner

from landuse_filter import config
from landuse_filter.adapters import benchmark, hub
from landuse_filter.adapters import g5k as g5k_adapter
from landuse_filter.adapters import remote as remote_mod
from landuse_filter.application import (
    bench,
    locate,
    node_main,
    remote_plan,
    remote_publish,
    repair,
    results,
    sync,
)
from landuse_filter.application import controller as ctl_mod
from landuse_filter.application import publish as publish_mod
from landuse_filter.application import status as status_mod
from landuse_filter.cli import OPS, app
from landuse_filter.cli import bench as bench_cli
from landuse_filter.cli import g5k as g5k_cli
from landuse_filter.cli import node as node_cli

runner = CliRunner()


def invoke(*args):
    return runner.invoke(app, [str(a) for a in args])


@pytest.mark.parametrize(
    "cmd",
    [
        [],
        ["g5k"],
        ["node"],
        ["bench"],
        ["g5k", "run"],
        ["g5k", "run-admission"],
        ["g5k", "cpu-job"],
        ["g5k", "clean"],
        ["g5k", "pause"],
        ["g5k", "resume"],
        ["node", "run"],
        ["node", "plan"],
        ["node", "replan"],
        ["node", "publish"],
        ["node", "repair"],
        ["bench", "admit"],
        ["bench", "compare"],
        ["plan"],
        ["publish"],
        ["status"],
    ],
)
def test_help_of_every_command_succeeds(cmd):
    result = invoke(*cmd, "--help")
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_version_and_fingerprint_json():
    assert invoke("version").exit_code == 0
    result = invoke("fingerprint", "--json")
    assert result.exit_code == 0
    assert "config_fingerprint" in json.loads(result.output)


def test_missing_required_options_are_usage_errors():
    assert invoke("g5k", "run").exit_code == 2
    assert invoke("node", "run").exit_code == 2
    assert invoke("g5k", "cpu-job", "plan").exit_code == 2
    assert invoke("node", "plan", "--dataset", "d").exit_code == 2


def test_node_plan_rejects_a_non_positive_chunk_size():
    result = invoke("node", "plan", "--dataset", "d", "--revision", "r", "--chunk-size", 0)
    assert result.exit_code == 2


# --- g5k run -------------------------------------------------------------------------


@pytest.fixture
def loop(monkeypatch):
    seen = {}
    monkeypatch.setattr(g5k_cli, "_controller", lambda work, settings: (work, settings))
    monkeypatch.setattr(ctl_mod, "run_loop", lambda ctl, **kw: seen.update(ctl=ctl, **kw))
    return seen


def test_g5k_run_defaults_come_from_the_ops_settings(loop, tmp_path):
    result = invoke("g5k", "run", "--datasets", "benchmark,other", "--work", tmp_path)
    assert result.exit_code == 0, result.output
    work, s = loop["ctl"]
    assert work == tmp_path
    assert s.datasets == ["benchmark", "other"]
    assert s.sites == list(OPS.sites)
    assert s.bucket == OPS.bucket
    assert s.max_jobs_total == OPS.max_jobs
    assert s.max_jobs_per_site == OPS.max_jobs_per_site
    assert s.walltime == timedelta(minutes=OPS.walltime_minutes)
    assert s.night_walltime == timedelta(minutes=OPS.night_walltime_minutes)
    assert s.max_queued_per_site == 1
    assert s.besteffort is False
    assert s.gpu_models == []
    assert s.window is None
    assert s.namespace is None
    assert loop["interval"] == OPS.interval_seconds
    assert loop["once"] is False


def test_g5k_run_options_are_forwarded(loop, tmp_path):
    result = invoke(
        "g5k", "run", "--datasets", "benchmark", "--work", tmp_path,
        "--sites", "nancy,lyon", "--gpu-models", "l40s,a100", "--max-jobs", 3,
        "--max-jobs-per-site", 2, "--walltime-minutes", 90, "--night-walltime-minutes", 600,
        "--besteffort", "--max-queued-per-site", 4, "--window", 64, "--namespace", "ns",
        "--bucket", "me/b", "--interval", 7, "--once",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    s = loop["ctl"][1]
    assert s.sites == ["nancy", "lyon"]
    assert s.gpu_models == ["l40s", "a100"]
    assert (s.max_jobs_total, s.max_jobs_per_site, s.max_queued_per_site) == (3, 2, 4)
    assert s.walltime == timedelta(minutes=90)
    assert s.night_walltime == timedelta(minutes=600)
    assert s.besteffort
    assert s.window == 64
    assert s.namespace == "ns"
    assert s.bucket == "me/b"
    assert loop["interval"] == 7
    assert loop["once"] is True


def test_g5k_run_rejects_a_non_integer_walltime(loop):
    assert invoke("g5k", "run", "--datasets", "b", "--walltime-minutes", "abc").exit_code == 2


def test_g5k_run_admission_builds_one_controller_per_gpu(monkeypatch, tmp_path):
    built, ran = [], {}

    class FakeController:
        def __init__(self, store, settings, log, sites):
            built.append(settings)

    monkeypatch.setattr(ctl_mod, "Controller", FakeController)
    monkeypatch.setattr(ctl_mod, "run_many", lambda ctls, cache, **kw: ran.update(ctls=ctls, **kw))
    result = invoke(
        "g5k", "run-admission", "--gpus", "l40s,a100", "--work", tmp_path, "--sites", "nancy"
    )
    assert result.exit_code == 0, result.output
    assert [s.gpu_models for s in built] == [["l40s"], ["a100"]]
    assert [s.namespace for s in built] == ["gpu-l40s", "gpu-a100"]
    assert all(s.datasets == ["benchmark"] and s.besteffort for s in built)
    assert all(s.sites == ["nancy"] and s.max_jobs_total == 5 for s in built)
    assert all(s.max_jobs_per_site == 3 and s.max_queued_per_site == 2 for s in built)
    assert all(s.bucket == OPS.bucket for s in built)
    assert len(ran["ctls"]) == 2
    assert ran["interval"] == OPS.interval_seconds


# --- g5k cpu-job ---------------------------------------------------------------------


@pytest.fixture
def remote(monkeypatch):
    calls = SimpleNamespace(submitted=[], deployed=[], policy=0)
    monkeypatch.setattr(ctl_mod, "commit", lambda ref="": "abc123")
    monkeypatch.setattr(ctl_mod, "git_archive", lambda ref: b"")
    monkeypatch.setattr(
        g5k_adapter, "deploy_code", lambda site, c, a: calls.deployed.append((site, c)) or "CODE"
    )
    monkeypatch.setattr(g5k_adapter, "ssh", lambda site, cmd, **k: "")
    monkeypatch.setattr(g5k_adapter, "policy_check", lambda site: setattr(calls, "policy", 1))
    monkeypatch.setattr(
        g5k_adapter, "submit", lambda site, args: calls.submitted.append((site, args)) or "4242"
    )
    return calls


def test_cpu_job_submits_a_node_job_script_command(remote):
    result = invoke(
        "g5k", "cpu-job", "replan", "--site", "lille", "--dataset", "ds", "--revision", "rev1"
    )
    assert result.exit_code == 0, result.output
    assert "replan job 4242 on lille" in result.output
    ((site, args),) = remote.submitted
    assert site == "lille"
    assert args[-1] == "CODE/scripts/node_job.sh CODE replan ds rev1"
    assert args[args.index("-p") + 1] == g5k_cli.CPU_JOB_PROPERTY
    assert "-q" in args
    assert remote.deployed == [("lille", "abc123")]
    assert remote.policy == 1


def test_cpu_job_rejects_an_unknown_mode(remote):
    result = invoke(
        "g5k", "cpu-job", "explode", "--site", "lille", "--dataset", "d", "--revision", "r"
    )
    assert result.exit_code == 2
    assert remote.submitted == []


# --- g5k clean / pause / resume / storage ----------------------------------------------


def _listing(site):
    if site == "lyon":
        raise g5k_adapter.RemoteError("down")
    return [("luf/code/old", 1.0)]


def test_clean_defaults_to_a_dry_run(monkeypatch, tmp_path):
    removed = []
    monkeypatch.setattr(ctl_mod, "commit", lambda ref="": "head")
    monkeypatch.setattr(g5k_adapter, "project_listing", _listing)
    monkeypatch.setattr(g5k_adapter, "remove", lambda site, plan: removed.append((site, plan)))
    result = invoke("g5k", "clean", "--sites", "nancy,lyon", "--work", tmp_path)
    assert result.exit_code == 0, result.output
    assert "nancy: 1 path(s) would be deleted" in result.output
    assert "lyon: unreachable" in result.output
    assert removed == []


def test_clean_apply_deletes(monkeypatch, tmp_path):
    removed = []
    monkeypatch.setattr(ctl_mod, "commit", lambda ref="": "head")
    monkeypatch.setattr(g5k_adapter, "project_listing", _listing)
    monkeypatch.setattr(g5k_adapter, "remove", lambda site, plan: removed.append((site, plan)))
    result = invoke("g5k", "clean", "--sites", "nancy", "--work", tmp_path, "--apply")
    assert result.exit_code == 0, result.output
    assert "nancy: 1 path(s) deleted" in result.output
    assert removed == [("nancy", ["luf/code/old"])]


def test_pause_and_resume_toggle_the_flag_file(tmp_path):
    assert invoke("g5k", "pause", "--work", tmp_path).exit_code == 0
    assert (tmp_path / "PAUSED").read_text() == "paused\n"
    assert invoke("g5k", "resume", "--work", tmp_path).exit_code == 0
    assert not (tmp_path / "PAUSED").exists()
    assert invoke("g5k", "resume", "--work", tmp_path).exit_code == 0  # idempotent


def test_pause_cancel_cancels_live_jobs(monkeypatch, tmp_path):
    fake = SimpleNamespace(cancel_all=lambda: ["nancy:1", "lyon:2"])
    monkeypatch.setattr(g5k_cli, "_controller", lambda work, settings: fake)
    result = invoke("g5k", "pause", "--work", tmp_path, "--cancel")
    assert result.exit_code == 0
    assert "nancy:1\nlyon:2" in result.output
    assert (tmp_path / "PAUSED").exists()


def test_pause_cancel_with_no_jobs(monkeypatch, tmp_path):
    fake = SimpleNamespace(cancel_all=list)
    monkeypatch.setattr(g5k_cli, "_controller", lambda w, s: fake)
    assert "no live jobs" in invoke("g5k", "pause", "--work", tmp_path, "--cancel").output


def test_storage_survives_an_unreachable_site(monkeypatch):
    def usage(site):
        if site == "lyon":
            raise g5k_adapter.RemoteError("ssh failed")
        return " 3G used \n"

    monkeypatch.setattr(g5k_adapter, "home_usage", usage)
    result = invoke("g5k", "storage", "--sites", "nancy,lyon")
    assert result.exit_code == 0
    assert "nancy: 3G used" in result.output
    assert "lyon: unreachable (ssh failed)" in result.output


# --- node ------------------------------------------------------------------------------


@pytest.mark.parametrize("code", [0, 1, 5])
def test_node_run_exits_with_the_use_case_code(monkeypatch, code):
    seen = []
    monkeypatch.setattr(node_main, "run", lambda a: seen.append(a) or code)
    result = invoke("node", "run", "--assignment", "asg-1")
    assert result.exit_code == code
    assert seen == ["asg-1"]


def test_node_publish_passes_dataset_revision_bucket(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "scratch_dir", lambda: tmp_path)
    monkeypatch.setattr(remote_mod, "BucketRemote", lambda bucket: ("remote", bucket))
    seen = {}

    def fake(rem, scratch, dataset, revision, fp, *, should_stop):
        seen.update(rem=rem, dataset=dataset, revision=revision, fp=fp)
        assert should_stop() is None
        return results_report

    results_report = SimpleNamespace(new_files=2, stopped=None)
    monkeypatch.setattr("signal.signal", lambda *_: None)
    monkeypatch.setattr(remote_publish, "run_publish", fake)
    monkeypatch.setattr("dataclasses.asdict", lambda r: {"new_files": r.new_files})
    result = invoke("node", "publish", "--dataset", "d", "--revision", "r", "--bucket", "x/y")
    assert result.exit_code == 0, result.output
    assert seen == {
        "rem": ("remote", "x/y"),
        "dataset": "d",
        "revision": "r",
        "fp": config.GENERATION_FP,
    }
    assert json.loads(result.output) == {"new_files": 2}


def test_node_plan_and_replan_use_the_default_bucket_and_chunk_size(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "scratch_dir", lambda: tmp_path)
    monkeypatch.setattr(remote_mod, "BucketRemote", lambda bucket: ("remote", bucket))
    monkeypatch.setattr(
        node_cli,
        "_plan_inputs",
        lambda dataset, revision, chunk_size: SimpleNamespace(
            chunk_size=chunk_size, rev=revision, fetch="F"
        ),
    )
    seen = {}

    def fake_plan(rem, scratch, dataset, fp, inputs, **kw):
        seen["plan"] = (rem, dataset, inputs.chunk_size, inputs.rev, kw["should_stop"]())
        return {"ok": "plan"}

    def fake_replan(rem, scratch, dataset, fp, inputs, **kw):
        seen["replan"] = (rem, dataset, inputs.chunk_size, kw["locate"])
        return {"ok": "replan"}

    monkeypatch.setattr(remote_plan, "run_plan", fake_plan)
    monkeypatch.setattr(remote_plan, "run_replan", fake_replan)
    monkeypatch.setattr(locate, "locator", lambda dataset, fetch, cell_of: ("loc", dataset, fetch))
    out = invoke("node", "plan", "--dataset", "d", "--revision", "r")
    assert out.exit_code == 0, out.output
    assert json.loads(out.output) == {"ok": "plan"}
    assert seen["plan"] == (("remote", OPS.bucket), "d", 2000, "r", False)
    out = invoke("node", "replan", "--dataset", "d", "--revision", "r", "--chunk-size", 50)
    assert out.exit_code == 0, out.output
    assert seen["replan"] == (("remote", OPS.bucket), "d", 50, ("loc", "d", "F"))


def test_node_repair_dedupes_and_resets_the_card_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "scratch_dir", lambda: tmp_path)
    monkeypatch.setattr(remote_mod, "BucketRemote", lambda bucket: ("remote", bucket))
    monkeypatch.setattr(publish_mod, "output_repo", lambda d: f"out/{d}")
    monkeypatch.setattr(
        hub, "remote_files", lambda repo: ["generations/a.parquet", "generations/b.parquet", "x"]
    )
    monkeypatch.setattr(hub, "download_all", lambda repo, rev, paths: [[Path(paths[0])]])
    calls = {}
    monkeypatch.setattr(
        repair,
        "dedupe_generations",
        lambda paths: {
            Path("generations/a.parquet"): pa.table({"x": [1]}),
            Path("generations/b.parquet"): None,
        },
    )
    monkeypatch.setattr(hub, "upload", lambda repo, files, msg: calls.update(up=(repo, files)))
    monkeypatch.setattr(hub, "delete", lambda repo, paths, msg: calls.update(rm=(repo, paths)))
    monkeypatch.setattr(
        remote_publish, "reset_card_cache", lambda rem, scratch, d: calls.update(reset=(rem, d))
    )
    result = invoke("node", "repair", "--dataset", "ds")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"rewritten": 1, "deleted": 1}
    assert calls["up"][0] == "out/ds"
    assert calls["up"][1][0][1] == "generations/a.parquet"
    assert calls["rm"] == ("out/ds", ["generations/b.parquet"])
    assert calls["reset"] == (("remote", OPS.bucket), "ds")


# --- bench ---------------------------------------------------------------------------


def _bench_world(monkeypatch, tmp_path, *, have, passed=True):
    items = {"i1": "s1", "i2": "s2"}
    plan_dir = tmp_path / "plans" / "benchmark" / config.GENERATION_FP
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "items.json").write_text(json.dumps(items))
    ref = [SimpleNamespace(item_id=i, language="en", expected="a") for i in items]
    monkeypatch.setattr(bench_cli, "_bench_root", lambda work: tmp_path)
    monkeypatch.setattr(benchmark, "read_reference", lambda root: ref)
    monkeypatch.setattr(remote_mod, "BucketRemote", lambda bucket: "remote")
    seen = {}

    def gathered(store, remote, fp):
        seen["fp"] = fp
        return {"s1": "x", "s2": "y"} if have else {"s1": "x"}

    monkeypatch.setattr(results, "gathered_decisions", gathered)
    gate = SimpleNamespace(passed=passed)
    monkeypatch.setattr(bench, "compare", lambda r, c, n: seen.update(n=n, cand=c) or gate)
    monkeypatch.setattr("dataclasses.asdict", lambda g: {"passed": g.passed})
    monkeypatch.setattr(bench, "macro_scores", lambda preds: {"macro": len(preds)})
    monkeypatch.setattr(
        benchmark, "ReferencePrediction", lambda *a: SimpleNamespace(item_id=a[0]), raising=False
    )
    monkeypatch.setattr(sync, "upload_gate", lambda rem, store, gpu: seen.update(uploaded=gpu))
    return seen


def test_bench_admit_passes_and_uploads_the_gate(monkeypatch, tmp_path):
    seen = _bench_world(monkeypatch, tmp_path, have=True)
    result = invoke("bench", "admit", "--gpu", "l40s", "--work", tmp_path, "--resamples", 5)
    assert result.exit_code == 0, result.output
    assert seen["uploaded"] == "l40s"
    assert seen["n"] == 5
    assert seen["fp"].endswith("-gpu-l40s")
    written = json.loads((tmp_path / "gates/admission/l40s.json").read_text())
    assert written["status"] == "admitted"
    assert "status: admitted" in result.output


def test_bench_admit_exit_3_when_the_gate_fails(monkeypatch, tmp_path):
    _bench_world(monkeypatch, tmp_path, have=True, passed=False)
    result = invoke("bench", "admit", "--gpu", "l40s", "--work", tmp_path, "--json")
    assert result.exit_code == 3
    assert json.loads(result.output)["status"] == "rejected"


def test_bench_admit_exit_4_when_incomplete(monkeypatch, tmp_path):
    seen = _bench_world(monkeypatch, tmp_path, have=False)
    result = invoke("bench", "admit", "--gpu", "l40s", "--work", tmp_path)
    assert result.exit_code == 4
    assert "incomplete: 1/2" in result.output
    assert "uploaded" not in seen


def test_bench_compare_namespace_and_label(monkeypatch, tmp_path):
    seen = _bench_world(monkeypatch, tmp_path, have=True)
    result = invoke(
        "bench", "compare", "--work", tmp_path, "--namespace", "cand", "--label", "mine",
        "--resamples", 3, "--json",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert seen["fp"].endswith("-cand")
    assert seen["n"] == 3
    assert (tmp_path / "gates/mine.json").exists()
    assert json.loads(result.output)["gate"] == {"passed": True}


def test_bench_compare_exit_codes(monkeypatch, tmp_path):
    _bench_world(monkeypatch, tmp_path, have=False)
    assert invoke("bench", "compare", "--work", tmp_path).exit_code == 4
    _bench_world(monkeypatch, tmp_path, have=True, passed=False)
    assert invoke("bench", "compare", "--work", tmp_path).exit_code == 3


# --- status / plan / publish -------------------------------------------------------------


def test_status_defaults_to_the_benchmark_dataset(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        status_mod, "summarize", lambda store, ds: seen.update(ds=ds) or {"benchmark": {"n": 1}}
    )
    result = invoke("status", "--work", tmp_path)
    assert result.exit_code == 0
    assert seen["ds"] == ["benchmark"]
    assert "benchmark: {'n': 1}" in result.output
    invoke("status", "--work", tmp_path, "--datasets", "a,b")
    assert seen["ds"] == ["a", "b"]


def test_plan_rejects_an_unknown_dataset(tmp_path):
    result = invoke("plan", "--dataset", "nope", "--revision", "r", "--work", tmp_path)
    assert result.exit_code == 2


def test_publish_exit_4_when_nothing_new(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        publish_mod,
        "publish",
        lambda store, ds, rev, dry_run: (
            seen.update(a=(ds, rev, dry_run)) or SimpleNamespace(new_files=0)
        ),
    )
    monkeypatch.setattr("dataclasses.asdict", lambda r: {"new_files": r.new_files})
    result = invoke("publish", "--dataset", "d", "--revision", "r", "--work", tmp_path, "--dry-run")
    assert result.exit_code == 4
    assert seen["a"] == ("d", "r", True)


def test_node_publish_card_only_runs_the_card_job(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "scratch_dir", lambda: tmp_path)
    monkeypatch.setattr(remote_mod, "BucketRemote", lambda bucket: ("remote", bucket))
    seen = {}

    def fake(rem, scratch, dataset, revision):
        seen.update(rem=rem, dataset=dataset, revision=revision)
        return SimpleNamespace(new_files=0)

    monkeypatch.setattr(remote_publish, "run_card_only", fake)
    monkeypatch.setattr("dataclasses.asdict", lambda r: {"new_files": r.new_files})
    result = invoke("node", "publish", "--card-only", "--dataset", "d", "--revision", "r")
    assert result.exit_code == 0, result.output
    assert seen["dataset"] == "d"


def test_stop_watch_reports_signals_and_the_deadline(monkeypatch):
    import os
    import signal
    import time

    from landuse_filter.cli.node import _stop_watch

    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR2, signal.SIGINT)}
    try:
        monkeypatch.delenv("LUF_JOB_DEADLINE_EPOCH", raising=False)
        watch = _stop_watch()
        assert watch() is None
        os.kill(os.getpid(), signal.SIGUSR2)
        assert watch() == "signal SIGUSR2"
        monkeypatch.setenv("LUF_JOB_DEADLINE_EPOCH", str(time.time() + 60))
        assert _stop_watch()() == "deadline"  # within the 6-minute margin
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


def test_publish_jobs_have_their_own_oar_name(remote):
    invoke("g5k", "cpu-job", "publish", "--site", "lille", "--dataset", "ds", "--revision", "r")
    invoke("g5k", "cpu-job", "plan", "--site", "lille", "--dataset", "ds", "--revision", "r")
    names = [args[args.index("-n") + 1] for _, args in remote.submitted]
    assert names == ["luf-publish-ds", "luf-plan-ds"]


def test_publish_loop_submits_until_the_status_says_done(remote, monkeypatch):
    state = {"polls": 0}
    done = {"done": True, "revision": "r1"}

    def read(_remote, dataset):
        state["polls"] += 1
        return done if state["polls"] >= 3 else None

    monkeypatch.setattr("landuse_filter.application.publish_loop.read_status", read)
    monkeypatch.setattr(remote_mod, "BucketRemote", lambda b: object())
    monkeypatch.setattr(
        g5k_adapter, "our_jobs", lambda site: [SimpleNamespace(name="luf-plan-ds")]
    )  # a planning job is live, not a publish job
    monkeypatch.setattr("time.sleep", lambda s: None)
    result = invoke("g5k", "publish-loop", "--site", "lille", "--dataset", "ds", "--revision", "r1")
    assert result.exit_code == 0, result.output
    assert len(remote.submitted) == 2  # polls 1 and 2: nothing live under its own name
    assert remote.submitted[0][1][-1].endswith("publish ds r1")
    assert result.output.strip().endswith("finished")
