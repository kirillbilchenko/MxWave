"""Activation evidence must survive buffer reuse, respect bounds and retain failed gates."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise CPU helpers without connecting to the cluster."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_snapshots_are_exact_independent_of_later_workspace_reuse(tmp_path) -> None:
    worker = importlib.import_module("repeat_worker")
    analyzer = importlib.import_module("analyze_repeat_model")
    words = torch.tensor([0, 32768, 16256, 32640, 65408, 32705], dtype=torch.uint16)
    value = words.view(torch.bfloat16).reshape(2, 3).transpose(0, 1)
    snapshot = worker.Snapshots()
    snapshot.record("mlp/input", {"x": value})
    expected = value.contiguous().view(torch.uint8).numpy().tobytes()
    value.zero_()
    root = tmp_path / "capture"
    report = snapshot.dump(root)
    item = report["events"][0]["tensors"][0]
    raw, decoded = analyzer.read_tensor(root, item)
    assert raw == expected
    assert item["path"] == "/x" and item["shape"] == [3, 2]
    assert decoded.shape == (6,)
    with pytest.raises(FileExistsError):
        snapshot.dump(root)


def test_snapshot_overflow_is_explicit_and_does_not_crash_forward(tmp_path) -> None:
    worker = importlib.import_module("repeat_worker")
    snapshots = worker.Snapshots(maximum=8)
    snapshots.record("good", torch.zeros(2))
    snapshots.record("overflow", torch.zeros(1))
    snapshots.record("ignored", torch.zeros(100))
    assert snapshots.nbytes == 8
    report = snapshots.dump(tmp_path / "bounded")
    assert report["status"] == "failed"
    assert [r["name"] for r in report["events"]] == ["good"]
    analyzer = importlib.import_module("analyze_repeat_model")
    with pytest.raises(ValueError, match="incomplete"):
        analyzer.compare_snapshots(tmp_path / "bounded", tmp_path / "bounded")


def test_first_changed_boundary_keeps_exact_input_and_quantifies_output(tmp_path) -> None:
    worker = importlib.import_module("repeat_worker")
    analyzer = importlib.import_module("analyze_repeat_model")
    for label, output in (("left", [1.0, 2.0]), ("right", [1.0, 2.25])):
        snap = worker.Snapshots()
        snap.record("mlp/input", torch.tensor([0.5, 0.25], dtype=torch.bfloat16))
        snap.record("mlp/output", torch.tensor(output, dtype=torch.bfloat16))
        snap.dump(tmp_path / label)
    result = analyzer.compare_snapshots(tmp_path / "left", tmp_path / "right")
    assert result["events"][0]["bits_exact"]
    first = result["first_changed_boundary"]
    assert first["name"] == "mlp/output"
    assert first["changed"][0]["max_abs"] == 0.25
    assert first["changed"][0]["different_values"] == 1
    manifest = tmp_path / "right/snapshots.json"
    data = json.loads(manifest.read_bytes())
    bad = tmp_path / "right" / data["events"][1]["tensors"][0]["file"]
    bad.write_bytes(b"\0" * 4)
    with pytest.raises(ValueError, match="bytes changed"):
        analyzer.compare_snapshots(tmp_path / "left", tmp_path / "right")


def test_equal_numeric_values_do_not_hide_signed_zero_byte_changes() -> None:
    worker = importlib.import_module("repeat_worker")
    result = worker.difference(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert result["bits_exact"] is False
    assert result["different_values"] == 0 and result["max_abs"] == 0.0


def test_snapshot_paths_cannot_escape_manifest_directory(tmp_path) -> None:
    analyzer = importlib.import_module("analyze_repeat_model")
    with pytest.raises(ValueError, match="local"):
        analyzer.read_tensor(tmp_path, {"file": "../file.bin"})
    item = {"file": "tensor.bin", "nbytes": 4, "dtype": "torch.float32", "shape": [2]}
    raw = b"\0" * 4
    (tmp_path / "tensor.bin").write_bytes(raw)
    item["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="shape"):
        analyzer.read_tensor(tmp_path, item)


def test_repeat_launch_preserves_stock_math_and_sets_workspace_before_process_start(
    monkeypatch,
) -> None:
    runner = importlib.import_module("run_repeat_model")
    command = ["serve", "--worker-extension-cls", "quality_worker.QualityWorker"]
    meta = {"environment": {"MXWAVE_MOE_MODE": "stock"}, "command": command}
    argv = ["docker", "run", "--entrypoint", "python", *command]
    monkeypatch.setattr(runner, "launch", lambda *a: (argv, meta))
    actual, result = runner.repeat_launch(0, "test", {}, 154880, "manifest")
    assert result["environment"]["MXWAVE_MOE_MODE"] == "stock"
    assert result["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert actual.index("CUBLAS_WORKSPACE_CONFIG=:4096:8") < actual.index("--entrypoint")
    assert actual[actual.index("--worker-extension-cls") + 1] == "repeat_worker.RepeatWorker"
