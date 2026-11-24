"""The palette pipeline owns one pause for a whole batch and preserves failed evidence."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch):
    """Import the orchestration without touching CUDA or either Spark."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    return importlib.import_module("run_scale_pipeline")


def test_batch_rejects_unowned_or_ambiguous_jobs(runner) -> None:
    valid = {"name": "latest", "root": "bf16-kernels-latest", "kind": "counters"}
    assert runner.jobs({"jobs": [valid]}) == [valid]
    for rows in ([], [valid, valid], [{**valid, "root": "../other"}],
                 [{**valid, "kind": "fullmodel"}], [None],
                 [valid, {**valid, "name": "other"}]):
        with pytest.raises((ValueError, TypeError)):
            runner.jobs({"jobs": rows})


def test_component_pair_never_restores_service_between_arms(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner.control, "ROOT", tmp_path)
    calls = []
    inspections = iter([{"Id": "arm0", "State": {"Running": False, "ExitCode": 0}},
                        {"Id": "arm1", "State": {"Running": False, "ExitCode": 0}}])
    monkeypatch.setattr(runner.control, "run", lambda rank, argv, **kw: calls.append(argv) or "logs")
    monkeypatch.setattr(runner.control, "inspect", lambda *a: next(inspections))
    monkeypatch.setattr(runner.control, "remote_json", lambda *a: {"status": "passed"})
    monkeypatch.setattr(runner.control, "restore", lambda *a: pytest.fail("Restored between arms"))
    result = runner.run_job("batch", {"name": "stage", "root": "bf16-kernels-stage",
                                      "kind": "timing"}, lambda phase: None)
    assert result["status"] == "collected" and len(result["stages"]) == 2
    assert [c for c in calls if c[:2] == ["docker", "rm"]] == [
        ["docker", "rm", "arm0"], ["docker", "rm", "arm1"]]
    assert not any(c[:2] in (["docker", "start"], ["docker", "stop"]) for c in calls)
    assert all((tmp_path / "bf16-kernels-stage" / (s + "-container.json")).exists()
               for s in ("sparse", "sparse-confirm"))


def test_failed_component_retains_its_container_and_does_not_launch_confirmation(
    runner, monkeypatch, tmp_path,
) -> None:
    monkeypatch.setattr(runner.control, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(runner.control, "run", lambda rank, argv, **kw: calls.append(argv) or "logs")
    monkeypatch.setattr(runner.control, "inspect", lambda *a: {
        "Id": "failed", "State": {"Running": False, "ExitCode": 1}})
    with pytest.raises(RuntimeError, match="exited 1"):
        runner.run_job("batch", {"name": "stage", "root": "bf16-kernels-stage",
                                  "kind": "timing"}, lambda phase: None)
    assert not any(c[:2] == ["docker", "rm"] for c in calls)
    assert sum(c[:3] == ["docker", "run", "-d"] for c in calls) == 1
    assert (tmp_path / "bf16-kernels-stage/sparse-container.json").is_file()


def test_preflight_requires_container_visible_libraries_before_a_pause(runner, monkeypatch) -> None:
    calls = []
    expected = {"libmxwave_moe_aligned.so": "control", "libmxwave_moe_sparse.so": "candidate"}
    replies = iter([json.dumps({"libraries_sha256": expected}), json.dumps(expected)])
    monkeypatch.setattr(runner.control, "run", lambda rank, argv, **kw:
                        calls.append(argv) or next(replies))
    result = runner.validate(Path("/prepared/bf16-kernels-check"))
    assert result["container_visible_libraries"] == expected
    assert calls[1][:3] == ["docker", "run", "--rm"]
    assert "NVIDIA_VISIBLE_DEVICES=void" in calls[1]
    assert not any(c[:2] in (["docker", "stop"], ["docker", "start"]) for c in calls)
    replies = iter([json.dumps({"libraries_sha256": expected}), json.dumps({})])
    with pytest.raises(ValueError, match="Container-visible"):
        runner.validate(Path("/prepared/bf16-kernels-check"))
