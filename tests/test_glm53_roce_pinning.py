"""A transport lead needs both ranks, both treatment orders and exact observed execution."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def fixture() -> ModuleType:
    """Import CPU-only review logic without initializing torch or a GPU."""
    path = Path(__file__).parents[1] / "integrations/glm53_mxfp4/benchmark_roce_pinning.py"
    spec = importlib.util.spec_from_file_location("roce_pinning_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(fixture: ModuleType) -> list[dict]:
    """Construct complete orders with a positive lead on each rank and exact output bytes."""
    return [{"rank": rank, "status": "complete", "source_sha256": "same-code",
        "runtime": {"error_seq": 0}, "original_proxy_affinity": [0, 1, 2, 3],
        "records": [{"block": block, "case": case["name"], "mode": mode,
            "bits_exact": True, "output_sha256": "same-bits",
            "observed_proxy_affinity": [0, 1, 2, 3] if mode == "off" else [3],
            "gpu_median_us": 100 if mode == "off" else 90,
            "gpu_p95_us": 120 if mode == "off" else 100,
            "wall_median_us": 130 if mode == "off" else 110,
            "wall_p95_us": 150 if mode == "off" else 130}
            for block, mode in enumerate(fixture.ORDER) for case in fixture.CASES]}
        for rank in (0, 1)]


def test_gate_requires_consistency_in_each_rank_and_order(fixture: ModuleType) -> None:
    reports = evidence(fixture)
    assert fixture.review_reports(reports)["full_model_followup_gate_passed"]
    for row in reports[1]["records"]:
        if row["block"] >= 4 and row["mode"] == "auto":
            row["gpu_p95_us"] = 125
    result = fixture.review_reports(reports)
    assert not result["component_lead_gate_passed"]
    assert not result["full_model_followup_gate_passed"]


def test_fast_collectives_alone_do_not_justify_model_reload(fixture: ModuleType) -> None:
    reports = evidence(fixture)
    for report in reports:
        for row in report["records"]:
            if row["case"] == "model_spaced_graph":
                row["wall_median_us"] = 100
    result = fixture.review_reports(reports)
    assert result["component_lead_gate_passed"]
    assert not result["full_model_followup_gate_passed"]


@pytest.mark.parametrize("fault", ["rank", "missing", "code", "bytes", "affinity", "runtime"])
def test_gate_rejects_incomplete_or_unobserved_comparison(fixture: ModuleType, fault: str) -> None:
    reports = copy.deepcopy(evidence(fixture))
    if fault == "rank":
        reports.pop()
    elif fault == "missing":
        reports[0]["records"].pop()
    elif fault == "code":
        reports[0]["source_sha256"] = "changed"
    elif fault == "bytes":
        reports[0]["records"][0]["output_sha256"] = "changed"
    elif fault == "affinity":
        reports[0]["records"][0]["observed_proxy_affinity"] = [3]
    else:
        reports[0]["runtime"]["error_seq"] = 1
    with pytest.raises(ValueError):
        fixture.review_reports(reports)


def test_percentile_has_fixed_tail_definition(fixture: ModuleType) -> None:
    assert fixture.percentile([float(v) for v in range(1, 21)], .95) == 19
    with pytest.raises(ValueError):
        fixture.percentile([], .95)


def test_authorized_pause_arms_watchdog_then_restores_after_fixture_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime failure must stop owned ranks and invoke restoration even while unwinding."""
    source = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(source))
    spec = importlib.util.spec_from_file_location("roce_paused_runner", source /
                                                "run_roce_pinning_idle.py")
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    c = runner.control
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "OLD", tmp_path / "old")
    guard = c.OLD / "triple-v2/src/integrations/marlin_scales/quiet_window.py"
    guard.parent.mkdir(parents=True)
    guard.write_text("def activity():\n return {'busy':False,'counters':{'tokens':1}}\n")
    (tmp_path / "diagnostics-v1").mkdir()
    (tmp_path / "transport-fixture").mkdir()
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runner, "memory_available", lambda rank: 64 * 1024**3)
    calls = []
    services = [{"Id": identifier, "Image": c.IMAGE, "Config": {}, "HostConfig": {},
                 "Mounts": [], "State": {"Running": True}} for identifier in c.ORIGINAL_IDS]
    monkeypatch.setattr(c, "inspect", lambda rank, identifier: services[rank])

    def run(rank, command, timeout=90):
        calls.append((rank, command))
        if command[0] == "sha256sum":
            return "same-code source.py"
        if command[:2] == ["docker", "inspect"]:
            return json.dumps({"Running": False, "ExitCode": 1})
        if command[:2] == ["docker", "ps"]:
            return "owned-id"
        return ""

    def popen(*args, **kwargs):
        assert not any(cmd[:2] == ["docker", "stop"] for _, cmd in calls)
        (tmp_path / "watchdog-fixture.json").write_text('{"ready":true}')
        return SimpleNamespace(pid=999)

    restored = []

    def restore(revision, path):
        state = json.loads(path.read_text())
        assert state["service_paused"] and state["phase"] == "restoring"
        stopped = [cmd[-1] for _, cmd in calls if cmd[:2] == ["docker", "stop"]]
        assert set(stopped[:2]) == set(c.ORIGINAL_IDS)
        assert len(stopped) == 4
        assert all(name.startswith("mxwave-glm53-full-fixture-raw-") for name in stopped[2:])
        restored.append(revision)
        return {"status": "complete", "health": 200}

    monkeypatch.setattr(c, "run", run)
    monkeypatch.setattr(c, "restore", restore)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr("sys.argv", ["runner", "--revision", "fixture", "--pause-glm"])
    with pytest.raises(RuntimeError, match="fixture rank failed"):
        runner.main()
    state = json.loads((tmp_path / "transport-controller-fixture.json").read_text())
    assert restored == ["fixture"]
    assert state["phase"] == "failed_restored" and not state["service_paused"]
