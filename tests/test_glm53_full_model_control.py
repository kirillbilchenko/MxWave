"""Service-restoration and scope checks for the opt-in GLM full-model controller."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the operational controller with all state confined to a temporary folder."""
    path = Path(__file__).parents[1] / "integrations/glm53_mxfp4/full_model_control.py"
    spec = importlib.util.spec_from_file_location("full_model_control_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def original(identifier: str, running: bool = False) -> dict[str, Any]:
    """Create an original-container identity with a configuration digest input."""
    return {"Id": identifier, "Image": "pinned-image", "Config": {"Env": ["PRIVATE=secret"]},
            "HostConfig": {"IpcMode": "host"}, "Mounts": [], "State": {"Running": running}}


def test_restore_starts_same_worker_then_head_and_checks_health(
    control: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = [original(identifier) for identifier in control.ORIGINAL_IDS]
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"originals": [control.identity(x) for x in services]}))
    calls = []

    def run(rank: int, command: list[str], timeout: float = 90) -> str:
        calls.append((rank, command))
        if command[:2] == ["docker", "start"]:
            assert command[2] == control.ORIGINAL_IDS[rank]
            services[rank]["State"]["Running"] = True
        return ""

    monkeypatch.setattr(control, "run", run)
    monkeypatch.setattr(control, "inspect", lambda rank, container: services[rank])
    monkeypatch.setattr(control, "health", lambda port: port == 8000)
    result = control.restore("fixture", path)
    assert result["status"] == "complete" and result["health"] == 200
    assert [(rank, cmd[2]) for rank, cmd in calls if cmd[:2] == ["docker", "start"]] == [
        (1, control.ORIGINAL_IDS[1]), (0, control.ORIGINAL_IDS[0])]
    assert not any(cmd[:2] in (["docker", "rm"], ["docker", "run"]) for _, cmd in calls)


def test_restore_refuses_configuration_drift(
    control: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = [original(identifier) for identifier in control.ORIGINAL_IDS]
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"originals": [control.identity(x) for x in services]}))
    services[1]["Config"]["Env"].append("CHANGED=yes")
    calls = []
    monkeypatch.setattr(control, "stop_owned", lambda revision: None)
    monkeypatch.setattr(control, "inspect", lambda rank, container: services[rank])
    monkeypatch.setattr(control, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(RuntimeError, match="configuration changed"):
        control.restore("fixture", path)
    assert not calls
    assert json.loads((tmp_path / "restoration-fixture.json").read_bytes())["status"] == "failed"


def test_container_identity_ignores_docker_mount_listing_order(control: ModuleType) -> None:
    service = original("id")
    service["Mounts"] = [{"Destination": "/model", "Source": "/source"},
                         {"Destination": "/overlay", "Source": "/runtime"}]
    before = control.identity(service)
    service["Mounts"].reverse()
    assert control.identity(service) == before
    service["Mounts"][0]["Source"] = "/changed"
    assert control.identity(service) != before


def test_experiment_stops_only_exact_owned_names(
    control: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def run(rank: int, command: list[str], timeout: float = 90) -> str:
        calls.append((rank, command))
        return "own-id" if command[:2] == ["docker", "ps"] else ""

    monkeypatch.setattr(control, "run", run)
    control.stop_owned("fixture")
    stops = [(rank, command[-1]) for rank, command in calls if command[1] == "stop"]
    assert len(stops) == 12
    assert all(name.startswith(("mxwave-glm53-full-fixture-", "mxwave-glm53-bench-fixture-"))
               for _, name in stops)
    assert all(identifier not in str(calls) for identifier in control.ORIGINAL_IDS)
    assert all(command[-1].startswith("name=^/") and command[-1].endswith("$")
               for _, command in calls if command[1] == "ps")


def test_engine_copies_no_secrets_and_excludes_old_checkpoint_specific_hooks(
    control: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "run", lambda *args, **kwargs: "")
    service = original("id")
    service["Config"]["Env"] += ["HF_TOKEN=hidden", "LLM_API_KEY=hidden", "GLM_FAST_LOAD=1",
        "GLM_FAST_LOAD_THREADS=16", "GLM_FAST_LOAD_TOKEN=hidden",
        "GLM_ROCE_PROXY_CPUS=auto", "GLM_ROCE_PROXY_ISOLATE=0", "GLM_ROCE_MAX_SIZE=4MiB",
        "QMIX_FP8_BLOCK=1", "GLM_CERT_HEAD=1", "GLM_TARGET_VOCAB_ARGMAX=1",
        "NCCL_NET=IB", "GLM_KPOOL_FIX=1"]
    service["Mounts"] = [
        {"Source": "/home/kirya/glm53-runtime-test", "Destination": "/overlay", "RW": False},
        {"Source": "/private/secret", "Destination": "/run/secrets/key", "RW": False},
        {"Source": "/previous/checkpoint", "Destination": "/model", "RW": False}]
    argv, settings = control.engine(0, "aligned", "fixture", service, 154880)
    assert "hidden" not in str((argv, settings))
    assert not any(source == "/private/secret" or target.startswith("/run/secrets/")
                   for source, target, _ in settings["mounts"])
    assert "previous/checkpoint" not in str(settings)
    assert settings["environment"]["NCCL_NET"] == "IB"
    assert settings["environment"]["GLM_KPOOL_FIX"] == "1"
    assert settings["environment"]["GLM_FAST_LOAD"] == "1"
    assert settings["environment"]["GLM_FAST_LOAD_THREADS"] == "16"
    assert settings["environment"]["GLM_ROCE_PROXY_CPUS"] == "auto"
    assert settings["environment"]["GLM_ROCE_PROXY_ISOLATE"] == "0"
    assert settings["environment"]["GLM_ROCE_MAX_SIZE"] == "4MiB"
    assert "GLM_FAST_LOAD_TOKEN" not in settings["environment"]
    assert "QMIX_FP8_BLOCK" not in settings["environment"]
    assert "GLM_TARGET_VOCAB_ARGMAX" not in settings["environment"]
    assert "--speculative-config" not in settings["command"]
    assert settings["command"][settings["command"].index("--port") + 1] == "18784"
    command = settings["command"]
    assert command[command.index("--reasoning-parser") + 1] == "glm45"
    assert json.loads(command[command.index("--default-chat-template-kwargs") + 1]) == {
        "reasoning_effort": "low"}


def test_transport_reuse_requires_both_ranks_and_matching_cpu_placement(control: ModuleType) -> None:
    old = [{"environment": {"GLM_ROCE_ALLREDUCE": "1", "B12X_ROCE_HCA": "rail0,rail1",
                            "OMP_NUM_THREADS": "4", "MXWAVE_MOE_MODE": "stock"}}
           for _ in range(2)]
    candidate = json.loads(json.dumps(old))
    candidate[0]["environment"]["MXWAVE_MOE_MODE"] = "aligned"
    control.validate_transport_control(old, candidate)
    for key, value in (("GLM_ROCE_PROXY_CPUS", "auto"), ("B12X_ROCE_HCA", "rail0"),
                       ("OMP_NUM_THREADS", "8"), ("NCCL_NET", "Socket")):
        changed = json.loads(json.dumps(candidate))
        changed[1]["environment"][key] = value
        with pytest.raises(ValueError, match="matched control"):
            control.validate_transport_control(old, changed)
    with pytest.raises(ValueError, match="both ranks"):
        control.validate_transport_control(old[:1], candidate)
    with pytest.raises(ValueError, match="Missing saved transport"):
        control.validate_transport_control([{}, {}], candidate)


def test_failed_preparation_never_stops_production(
    control: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control.signal, "signal", lambda *args: None)
    monkeypatch.setattr(control, "remote_json", lambda *args: {
        "phase": "failed_preserved", "status": "failed"})
    calls = []
    monkeypatch.setattr(control, "run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(control, "restore", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="preparation gate failed"):
        control.control("fixture")
    assert not calls
    state = json.loads((tmp_path / "full-controller-fixture.json").read_bytes())
    assert state["phase"] == "preparation_failed" and not state["service_paused"]


def test_attention_mapping_is_opt_in_and_does_not_modify_original(
    control: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "run", lambda *args, **kwargs: "")
    service = original("id")
    _, default = control.engine(0, "stock", "fixture", service, 154880)
    argv, repaired = control.engine(0, "stock", "fixture", service, 154880,
                                    attention_mapping=True)
    assert "MXWAVE_GLM53_ATTENTION_MAPPING" not in default["environment"]
    assert repaired["environment"]["MXWAVE_GLM53_ATTENTION_MAPPING"] == "1"
    assert "MXWAVE_GLM53_ATTENTION_MAPPING=1" in argv
    assert service["Config"]["Env"] == ["PRIVATE=secret"]


def test_independent_watchdog_restores_if_controller_dies(
    control: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"phase": "benchmarking_raw", "service_paused": True}))

    def dead(pid: int, signum: int) -> None:
        raise ProcessLookupError

    calls = []
    monkeypatch.setattr(control.os, "kill", dead)
    monkeypatch.setattr(control, "restore", lambda *args: calls.append(args))
    control.watch("fixture", 98765, path)
    assert calls == [("fixture", path)]
    assert json.loads((tmp_path / "watchdog-fixture.json").read_bytes())["ready"]


def test_kda_arm_requires_repaired_loading_and_keeps_expert_kernel_stock(
    control: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "run", lambda *args, **kwargs: "")
    service = original("id")
    with pytest.raises(ValueError, match="audited attention mapping"):
        control.engine(0, "kda", "fixture", service, 154880)
    _, candidate = control.engine(0, "kda", "fixture", service, 154880,
                                  attention_mapping=True)
    environment = candidate["environment"]
    assert environment["MXWAVE_MOE_MODE"] == "stock"
    assert environment["MXWAVE_GLM53_KDA_MODE"] == "packed"
    assert environment["MXWAVE_GLM53_KDA_SOURCE"] == "/model"
    assert candidate["command"].count("--kv-cache-memory-bytes") == 1
    assert service["Config"]["Env"] == ["PRIVATE=secret"]
