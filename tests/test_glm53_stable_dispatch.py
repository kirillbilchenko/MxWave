"""The deterministic dispatch experiment must be explicit and fail outside its scope."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import the CPU contract without a Triton installation or GPU."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_cpu_metadata_cannot_enter_a_gpu_dispatch_consumer() -> None:
    module = importlib.import_module("stable_dispatch")
    with pytest.raises(ValueError, match="CUDA device"):
        module.canonicalize(torch.zeros(32, dtype=torch.int32),
                            torch.zeros(4, dtype=torch.int32),
                            torch.tensor([8], dtype=torch.int32),
                            num_tokens=2, num_experts=288, block=8)


@pytest.mark.parametrize("tokens,experts,block", [(2049, 288, 8), (-1, 288, 8),
                                                (2, 0, 8), (2, 288, 7)])
def test_unsupported_geometry_is_rejected_before_loading_gpu_code(tokens, experts, block) -> None:
    module = importlib.import_module("stable_dispatch")
    with pytest.raises(ValueError, match="geometry"):
        module.canonicalize(*(torch.empty(0, dtype=torch.int32) for _ in range(3)),
                            num_tokens=tokens, num_experts=experts, block=block)


def test_quality_plugin_installs_dispatch_only_when_explicitly_enabled(monkeypatch) -> None:
    worker = importlib.import_module("quality_worker")
    stable = importlib.import_module("stable_dispatch")
    calls = []
    monkeypatch.setattr(worker.model_plugin, "install", lambda: calls.append("model"))
    monkeypatch.setattr(stable, "install", lambda: calls.append("dispatch"))
    monkeypatch.delenv("MXWAVE_STABLE_MOE_DISPATCH", raising=False)
    worker.install()
    assert calls == ["model"]
    monkeypatch.setenv("MXWAVE_STABLE_MOE_DISPATCH", "1")
    worker.install()
    assert calls == ["model", "model", "dispatch"]


def test_both_quality_arms_use_the_same_opt_in_dispatch_and_workspace(monkeypatch) -> None:
    driver = importlib.import_module("run_stable_quality")
    def original(*args, **kwargs):
        command = ["serve", "--worker-extension-cls", "quality_worker.QualityWorker"]
        return (["docker", "run", "--entrypoint", "python", *command],
                {"command": command, "environment": {}})
    monkeypatch.setattr(driver, "_launch", original)
    for label in ("control", "candidate"):
        argv, meta = driver.launch(0, "test", label, {}, 154880, "manifest")
        assert meta["environment"]["MXWAVE_STABLE_MOE_DISPATCH"] == "1"
        assert meta["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert argv[argv.index("--worker-extension-cls") + 1] == "stable_dispatch.StableWorker"
