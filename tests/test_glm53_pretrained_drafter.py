"""Check conversion boundaries and prevent a pretrained trial from interrupting capture."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from integrations.glm53_mxfp4.prepare_pretrained_drafter import _QUEUE
from integrations.glm53_mxfp4.quantize_pretrained_drafter_fp8 import quantize_blocks


def test_block_scales_and_zero_matrix() -> None:
    weight = torch.zeros(256, 256, dtype=torch.bfloat16)
    weight[:128, :128] = 448
    weight[128:, :128] = -224
    weight[:128, 128:] = 112
    quantized, scales = quantize_blocks(weight)
    assert torch.equal(scales, torch.tensor([[1.0, 0.25], [0.5, 1.0]]))
    restored = (quantized.float().reshape(2, 128, 2, 128)
                * scales[:, None, :, None]).reshape(256, 256)
    assert torch.equal(restored, weight.float())


@pytest.mark.parametrize("kind", ["nonfinite", "shape", "dtype"])
def test_reject_invalid_weights(kind: str) -> None:
    value = torch.zeros(128, 128, dtype=torch.bfloat16)
    if kind == "nonfinite":
        value[0, 0] = float("nan")
    elif kind == "shape":
        value = value[:127]
    else:
        value = value.float()
    with pytest.raises(ValueError):
        quantize_blocks(value)


class StillWaiting(Exception):
    """End a test after verifying that the queue waits instead of launching."""


def queue_fixture(tmp_path: Path, phase: str) -> tuple[dict[str, Any], Path, Path]:
    """Create source-pinned queue inputs with a separately owned capture controller."""
    parent = tmp_path / "drafter-full-capture-v1"
    parent.mkdir()
    (parent / "plan.json").write_text('{}\n')
    (parent / "controller.json").write_text(json.dumps({
        "phase": phase, "service_paused": phase == "measuring_ct",
        "restoration": {"health_200": True, "originals_running": True},
    }))
    root = tmp_path / "trial"
    root.mkdir()
    (root / "plan.json").write_text(json.dumps({"wait_capture_sha256":
        hashlib.sha256((parent / "plan.json").read_bytes()).hexdigest()}))
    (root / "source-manifest.json").write_text(json.dumps({"plan.json":
        hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()}))
    namespace: dict[str, Any] = {"__file__": str(root / "wait_for_capture.py"),
                                 "__name__": "queue_fixture"}
    # Execute our own generated queue, with all serving subprocesses substituted below.
    exec(compile(_QUEUE, "wait_for_capture.py", "exec"), namespace)  # noqa: S102
    return namespace, root, parent


@pytest.mark.parametrize("phase", ["measuring_ct", "complete_restored"])
def test_no_launch_without_verified_capture(tmp_path: Path, phase: str) -> None:
    namespace, root, _ = queue_fixture(tmp_path, phase)
    launched = []

    def wait(_: float) -> None:
        raise StillWaiting

    namespace["subprocess"] = SimpleNamespace(run=lambda *a, **k: launched.append(a))
    namespace["time"] = SimpleNamespace(time=lambda: 1.0, sleep=wait)
    with pytest.raises(StillWaiting):
        namespace["main"]()
    assert not launched
    assert json.loads((root / "queue.json").read_bytes())["phase"].startswith("waiting")


def test_failed_capture_never_launches_trial(tmp_path: Path) -> None:
    namespace, _, _ = queue_fixture(tmp_path, "failed_restored")
    launched = []
    namespace["subprocess"] = SimpleNamespace(run=lambda *a, **k: launched.append(a))
    with pytest.raises(AssertionError, match="Capture failed"):
        namespace["main"]()
    assert not launched


def test_verified_restoration_allows_bounded_controller(tmp_path: Path) -> None:
    namespace, root, parent = queue_fixture(tmp_path, "complete_restored")
    (parent / "postprocess-complete.json").write_text('{"status":"verified"}')
    (parent / "run-prepared-status.json").write_text('{"step":"finalize","returncode":0}')
    launched = []

    def run(argv: list[str], **_: Any) -> SimpleNamespace:
        launched.append(Path(argv[-1]).name)
        (root / "controller.json").write_text(
            '{"phase":"complete_restored","service_paused":false}',
        )
        (root / "run-prepared-status.json").write_text(
            '{"step":"finalize","returncode":0,"control_returncode":0}',
        )
        return SimpleNamespace(returncode=0)

    namespace["subprocess"] = SimpleNamespace(run=run, STDOUT=-2)
    namespace["main"]()
    assert launched == ["prepare.py", "run-prepared.py"]
    assert json.loads((root / "queue.json").read_bytes())["training_started"] is False
