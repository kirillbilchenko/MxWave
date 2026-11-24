"""Component diagnostics retain strict arithmetic and restore production after any failure."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_exact_gate_rejects_rounding_even_when_relative_error_is_small(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    bench = importlib.import_module("benchmark_bf16_kernels")
    expected = torch.ones(4096, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0] = 1.0078125
    assert bench.arithmetic(actual, expected, exact=False)["passed"]
    assert not bench.arithmetic(actual, expected, exact=True)["passed"]
    actual[0] = float("nan")
    assert not bench.arithmetic(actual, expected, exact=False)["passed"]


def test_counter_privilege_is_confined_to_separate_profile_container(monkeypatch, tmp_path) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    runner = importlib.import_module("run_bf16_kernels")
    plain = runner.command("fixture", tmp_path, profile=False)
    profiled = runner.command("fixture", tmp_path, profile=True)
    assert "SYS_ADMIN" not in plain and "SYS_ADMIN" in profiled
    assert "--privileged" not in profiled and "/ncu/ncu" in profiled
    assert "--clock-control" in profiled and "none" in profiled
    assert "--network" in profiled and profiled[profiled.index("--network") + 1] == "none"
    assert "--profile" not in plain and "--profile" in profiled
    assert "mxwave-glm53-full-fixture-raw-r1" in plain
    assert "mxwave-glm53-full-fixture-aligned-r1" in profiled
    cache = runner.command("fixture", tmp_path, profile=False, cache=True)
    assert "/work/diagnose_bf16_cache.py" in cache and "/ncu/ncu" not in cache
    assert "SYS_ADMIN" not in cache and "180" in cache
    with pytest.raises(ValueError, match="must not run under a profiler"):
        runner.command("fixture", tmp_path, profile=True, cache=True)


def test_failed_fixture_restores_after_independent_watchdog_ack(monkeypatch, tmp_path) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    runner = importlib.import_module("run_bf16_kernels")
    c = runner.control
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "OLD", tmp_path / "old")
    (tmp_path / "diagnostics-v1").mkdir()
    (tmp_path / "bf16-kernels-fixture").mkdir()
    guard = c.OLD / "triple-v2/src/integrations/marlin_scales/quiet_window.py"
    guard.parent.mkdir(parents=True)
    guard.write_text("def activity():\n return {'busy':False,'counters':{'tokens':1}}\n")
    services = [{"Id": identifier, "Image": c.IMAGE, "Config": {}, "HostConfig": {},
                 "Mounts": [], "State": {"Running": True}} for identifier in c.ORIGINAL_IDS]
    monkeypatch.setattr(c, "inspect", lambda rank, name: services[rank]
                        if name in c.ORIGINAL_IDS else {"State": {"Running": False, "ExitCode": 1}})
    monkeypatch.setattr(c, "preflight", lambda *args: {})
    monkeypatch.setattr(runner, "memory_available", lambda rank: 64 * 1024**3)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr("sys.argv", ["run_bf16_kernels.py", "--revision", "fixture", "--pause-glm"])
    calls = []

    def run(rank, argv, timeout=90):
        calls.append(argv)
        if argv[:2] == ["docker", "stop"]:
            state = json.loads((tmp_path / "bf16-kernels-controller-fixture.json").read_text())
            assert state["service_paused"] and state["watchdog_pid"] == 999
        return "{}"

    def popen(*args, **kwargs):
        assert not any(argv[:2] == ["docker", "stop"] for argv in calls)
        assert kwargs["start_new_session"] and kwargs["stdin"] == runner.subprocess.DEVNULL
        (tmp_path / "watchdog-fixture.json").write_text('{"ready":true}')
        return SimpleNamespace(pid=999)

    restored = []

    def restore(revision, path):
        state = json.loads(path.read_text())
        assert state["phase"] == "restoring" and state["service_paused"]
        restored.append(revision)
        return {"status": "complete", "health": 200}

    monkeypatch.setattr(c, "run", run)
    monkeypatch.setattr(c, "restore", restore)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="timing component exited 1"):
        runner.main()
    state = json.loads((tmp_path / "bf16-kernels-controller-fixture.json").read_text())
    assert restored == ["fixture"] and state["phase"] == "failed_restored"
    assert not state["service_paused"]


def test_review_rejects_missing_orders_and_unattributed_counters(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    analyzer = importlib.import_module("analyze_bf16_kernels")
    timing = {"status": "passed", "profile": False, "device": {}, "limits": [], "cases": []}
    rows = [["ID", "Kernel Name", "Grid Size", "Block Size"], ["", "", "", ""]]
    for name in analyzer.CASES:
        case = {"name": name, "gpu_weight_bits_exact": True, "logical_shape": [64, 64],
                "native_shape": [64, 64], "raw_bytes": 8192, "packed_bytes": 6000,
                "source_weight_sha256": "source", "artifact_sha256": "artifact", "measurements": []}
        for m in (1, 4, 8):
            for s in (1, 2, 4, 8):
                arm = {"dense_median_us": 10, "fused_median_us": 9, "median_paired_ratio": .9,
                    "samples": [{"order": order, "cuda_ratio": .9} for order in (
                        ["dense", "fused"], ["fused", "dense"])]}
                case["measurements"].append({"tokens": m, "split_k": s,
                    "candidate_gates_passed": s == 1, "arithmetic": {"relative_l2": 0},
                    "eager": copy.deepcopy(arm), "graph": copy.deepcopy(arm)})
        timing["cases"].append(case)
        for arm in ("dense", "fused_s1", "fused_s4"):
            kernel = "cutlass::Kernel2" if arm == "dense" else "BF16TripleBitmap_MM_Kernel"
            rows.append([str(len(rows) - 2), f"bf16_profile/{name}/M4/{arm}/{kernel}", "(1,1,1)",
                         "(128,1,1)"])
    for _ in range(11):
        rows.append([str(len(rows) - 2), "bf16_profile/head-r0/M4/fused_s1/helper", "(1,1,1)",
                     "(128,1,1)"])
    assert analyzer.review(timing, rows)["profile_total_kernels"] == 20
    altered = copy.deepcopy(timing)
    altered["cases"][0]["measurements"][0]["eager"]["samples"].pop()
    with pytest.raises(ValueError, match="both timing orders"):
        analyzer.review(altered, rows)
    with pytest.raises(ValueError, match="NVTX label"):
        analyzer.review(timing, rows[:2] + [["0", "unattributed", "(1,1,1)", "(128,1,1)"]])
    with pytest.raises(ValueError, match="Missing counter groups"):
        analyzer.review(timing, rows[:-1])


def test_cache_eviction_precedes_timing_interval(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    probe = importlib.import_module("diagnose_bf16_cache")
    events = []
    trace = []

    class Event:
        def __init__(self, **kwargs):
            self.name = "start" if not events else "end"
            events.append(self)

        def record(self):
            trace.append(self.name)

        def synchronize(self):
            trace.append("synchronize")

        def elapsed_time(self, other):
            assert other is events[1]
            return .125

    monkeypatch.setattr(probe.torch.cuda, "Event", Event)
    assert probe.timed_call(lambda: trace.append("kernel"), lambda: trace.append("evict")) == 125
    assert trace == ["evict", "start", "kernel", "end", "synchronize"]
