"""The candidate-only profiler must preserve the qualified path and always restore GLM."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the controller without importing torch, vLLM or accessing either GPU."""
    source = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(source))
    return importlib.import_module("run_nsys_model")


def test_diagnostic_cannot_silently_change_qualified_execution(runner: ModuleType) -> None:
    reference = {"source_metadata": {"config": "fixed"}, "libraries": {"native": "same"},
        "vocabulary": 154880, "code": {"kda_runtime.py": "same", "full_model_control.py": "old"}}
    current = copy.deepcopy(reference)
    current["code"]["full_model_control.py"] = "inherited-proxy-settings"
    current["code"]["nsys_worker.py"] = "diagnostic"
    runner.validate_qualified_source(current, reference)
    for name in ("kda_runtime.py", "unqualified_adapter.py"):
        changed = copy.deepcopy(current)
        changed["code"][name] = "changed"
        with pytest.raises(ValueError, match="Qualified execution changed"):
            runner.validate_qualified_source(changed, reference)


def test_launch_has_only_one_cupti_client_and_preserves_candidate_settings(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.c, "run", lambda *a, **kw: "")
    original = {"Config": {"Env": ["GLM_FAST_LOAD=1", "GLM_ROCE_PROXY_CPUS=auto"]}, "Mounts": []}
    argv, meta = runner.launch(0, "fixture", original, 154880, "manifest")
    command = meta["command"]
    profiler = json.loads(command[command.index("--profiler-config") + 1])
    assert profiler == {"profiler": "cuda", "ignore_frontend": True,
                        "detailed_trace_annotation": True}
    assert "--trace=cuda,nvtx,osrt" in argv and "--pytorch=none" in argv
    assert "--capture-range-end=stop" in argv and "--wait=all" in argv
    assert all("stop-shutdown" not in token for token in argv)
    assert argv[argv.index("--entrypoint") + 1] == str(runner.NSYS)
    assert command[command.index("--kv-cache-memory-bytes") + 1] == str(4 * 1024**3)
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.85"
    assert json.loads(command[command.index("--compilation-config") + 1])[
        "cudagraph_capture_sizes"] == [1]
    assert command[command.index("--worker-extension-cls") + 1] == "nsys_worker.NsysWorker"
    env = meta["environment"]
    assert env["MXWAVE_MOE_MODE"] == "aligned" and env["MXWAVE_MOE_DISPATCH"] == "dispatcher"
    assert env["MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS"] == "8"
    assert env["MXWAVE_GLM53_VOCAB_MODE"] == "packed"
    assert env["GLM_ROCE_PROXY_CPUS"] == "auto" and env["GLM_FAST_LOAD"] == "1"


def test_startup_failure_restores_originals_after_watchdog_is_armed(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = runner.c
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "OLD", tmp_path / "old")
    (tmp_path / "diagnostics-v1").mkdir()
    manifest = tmp_path / "vocab-v2/prepared.json"
    manifest.parent.mkdir()
    manifest.write_text("{}")
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    proto = tmp_path / "run-combineddispatchv1/results/kda/protocol.json"
    proto.parent.mkdir(parents=True)
    proto.write_text(json.dumps({"timing": [{"prompt_tokens": 256, "batch": 4,
                                          "prompts": [[1] * 256] * 4}]}))
    guard = c.OLD / "triple-v2/src/integrations/marlin_scales/quiet_window.py"
    guard.parent.mkdir(parents=True)
    guard.write_text("def activity():\n return {'busy':False,'counters':{'tokens':1}}\n")
    services = [{"Id": identifier, "Image": c.IMAGE, "Config": {"Env": []}, "HostConfig": {},
                 "Mounts": [], "State": {"Running": True}} for identifier in c.ORIGINAL_IDS]
    monkeypatch.setattr(c, "inspect", lambda rank, name: services[rank] if name in c.ORIGINAL_IDS
                        else {"State": {"Running": False}})
    bound = {"code": {}, "source_metadata": {}, "libraries": {}, "vocabulary": 154880}
    monkeypatch.setattr(c, "preflight", lambda *args: bound)
    monkeypatch.setattr(c, "remote_json", lambda *args: {"phase": "complete", "service_paused": False,
        "preflight": bound, "vocabulary_fixture": {"manifest_sha256": sha}})
    monkeypatch.setattr(runner, "cpu_cli_check", lambda *args: "validated")
    monkeypatch.setattr(c, "health", lambda port: False)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    calls = []

    def run(rank, argv, timeout=90):
        calls.append((rank, argv))
        if argv[:2] == ["docker", "stop"]:
            assert json.loads((tmp_path / "nsys-controller-fixture.json").read_text())[
                "service_paused"]
        if argv[0] == "sha256sum":
            return sha + " file"
        return "tool-or-container"

    def popen(*args, **kwargs):
        assert not any(argv[:2] == ["docker", "stop"] for _, argv in calls)
        assert kwargs["start_new_session"] and kwargs["stdin"] == runner.subprocess.DEVNULL
        (tmp_path / "watchdog-fixture.json").write_text('{"ready":true}')
        return SimpleNamespace(pid=999)

    restored = []

    def restore(revision, path):
        value = json.loads(path.read_text())
        assert value["phase"] == "restoring" and value["service_paused"]
        assert [argv[-1] for _, argv in calls if argv[:2] == ["docker", "stop"]] == list(
            c.ORIGINAL_IDS)
        restored.append(revision)
        return {"status": "complete", "health": 200}

    monkeypatch.setattr(c, "run", run)
    monkeypatch.setattr(c, "restore", restore)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="exited during startup"):
        runner.control("fixture")
    state = json.loads((tmp_path / "nsys-controller-fixture.json").read_text())
    assert restored == ["fixture"]
    assert state["phase"] == "failed_restored" and not state["service_paused"]
