"""Check portable asset integrity, explicit launch policy and inert plugin discovery."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mxwave_glm53 import bundle, cli


@pytest.fixture
def asset_source(tmp_path, monkeypatch):
    """Use small artifacts to exercise the complete export/verify transaction."""
    source = tmp_path / "source"
    paths = {
        "native/libmxwave_moe_raw.so": "native/libmxwave_moe_raw.so",
        "native/libmxwave_moe_sparse.so": "quality-best-v1-native/libmxwave_moe_sparse.so",
        "native/libmxwave_kda_zipserv.so": "native/libmxwave_kda_zipserv.so",
        "rounding-reference.json": "ct-draft-length-v1/rounding-reference.json",
    }
    pins = {}
    for target, origin in paths.items():
        path = source / origin
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(origin)
        pins[target] = bundle.digest(path)
    cases = []
    for kind in ("head", "embedding"):
        for rank in (0, 1):
            path = source / "vocab-v2" / f"{kind}-{rank}.bin"
            path.parent.mkdir(exist_ok=True)
            path.write_text(f"{kind} {rank}")
            cases.append({"kind": kind, "rank": rank, "artifact": path.name,
                          "artifact_sha256": bundle.digest(path)})
    prepared = source / "vocab-v2/prepared.json"
    prepared.write_text(json.dumps({"cases": cases}))
    pins["vocab/prepared.json"] = bundle.digest(prepared)
    monkeypatch.setattr(bundle, "PINNED_ASSETS", pins)
    return source


def test_bundle_is_relocatable_and_rejects_tampering(asset_source, tmp_path):
    first = tmp_path / "first"
    manifest = bundle.prepare_bundle(asset_source, first)
    moved = tmp_path / "another-location"
    first.rename(moved)
    assert bundle.verify_bundle(moved) == manifest
    (moved / "vocab/head-1.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        bundle.verify_bundle(moved)


def test_failed_export_is_atomic_and_never_overwrites(asset_source, tmp_path):
    target = tmp_path / "bundle"
    (asset_source / "vocab-v2/head-0.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        bundle.prepare_bundle(asset_source, target)
    assert not target.exists()
    target.mkdir()
    with pytest.raises(FileExistsError):
        bundle.prepare_bundle(asset_source, target)


@pytest.mark.parametrize("name", ["../secret", "/etc/passwd", "escape"])
def test_bundle_rejects_escape(tmp_path, name):
    root = tmp_path / "bundle"
    root.mkdir()
    outside = tmp_path / "secret"
    outside.write_bytes(b"outside")
    (root / "escape").symlink_to(outside)
    with pytest.raises(ValueError):
        bundle.local_file(root, name)


def test_manifest_cannot_remove_required_payload(asset_source, tmp_path):
    target = tmp_path / "bundle"
    manifest = bundle.prepare_bundle(asset_source, target)
    del manifest["files"]["vocab/head-0.bin"]
    (target / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inventory differs"):
        bundle.verify_bundle(target)


def test_plugin_is_inert_without_opt_in():
    program = (
        "import os,sys; os.environ.pop('MXWAVE_GLM53_ENABLE',None);"
        "from mxwave_glm53.plugin import register; register(); register();"
        "assert 'vllm' not in sys.modules; assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", program], check=True)


def test_launch_pins_profile_and_has_no_experiment_paths(asset_source, tmp_path, monkeypatch):
    target = tmp_path / "runtime"
    bundle.prepare_bundle(asset_source, target)
    model, draft = tmp_path / "model", tmp_path / "draft"
    model.mkdir()
    draft.mkdir()
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    monkeypatch.setattr(cli, "verify_files", lambda root, files: None)
    args = argparse.Namespace(
        bundle=target, model=model, drafter=draft, reports=tmp_path / "reports",
        node_rank=1, master_addr="10.0.0.1", master_port=29719,
        port=8046, host="127.0.0.1", max_model_len=32768, kv_cache_gib=10,
    )
    command, env = cli.launch_command(args)
    assert "--headless" in command, "The TP follower must not start another engine/API server"
    args.node_rank = 0
    leader_command, _ = cli.launch_command(args)
    assert "--headless" not in leader_command
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--kv-cache-memory-bytes") + 1] == str(10 * 1024**3)
    spec = json.loads(command[command.index("--speculative-config") + 1])
    assert spec["num_speculative_tokens"] == 5
    assert spec["kv_cache_dtype"] == "fp8_e4m3"
    assert "attention_backend" not in spec
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.85"
    assert command[command.index("--block-size") + 1] == "2304"
    assert spec["num_speculative_tokens_per_batch_size"] == [[1, 1, 5], [2, 2, 5], [3, 16, 3]]
    assert env["VLLM_PLUGINS"] == "mxwave_glm53"
    assert "PYTHONPATH" not in env
    assert not any("/audit/" in x or "/trial/" in x or "/overlay/" in x for x in command)
    assert not args.reports.exists(), "Building a launch must not start serving or create reports"
    args.draft_attention_backend = "TRITON_ATTN"
    triton_command, triton_env = cli.launch_command(args)
    triton_spec = json.loads(triton_command[triton_command.index("--speculative-config") + 1])
    assert triton_spec == {**spec, "attention_backend": "TRITON_ATTN"}
    normalized = list(triton_command)
    normalized[normalized.index("--speculative-config") + 1] = json.dumps(spec)
    assert normalized == leader_command, "Backend choice must not lower any capacity settings"
    assert triton_env == env
    args.draft_attention_backend = "unsupported"
    with pytest.raises(ValueError, match="Invalid serving"):
        cli.launch_command(args)


def test_doctor_requires_pinned_vllm(monkeypatch):
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: "0.31.0")
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.platform, "machine", lambda: "aarch64")
    result = cli.doctor()
    assert result["status"] == "failed"
    assert "vllm==0.32.0" in result["errors"][0]


def test_packaged_mxfp8_recipe_matches_original():
    import torch

    from integrations.glm53_mxfp4.kda_mxfp8_roundtrip import roundtrip as original
    from mxwave_glm53._runtime.kda_mxfp8_runtime import encode_weight

    weight = torch.randn((17, 64), generator=torch.Generator().manual_seed(42)).bfloat16()
    decoded, codes, scales = original(weight)
    actual_codes, actual_scales, actual_decoded, _ = encode_weight(weight, chunk_rows=3)
    assert torch.equal(codes, actual_codes)
    assert torch.equal(scales, actual_scales)
    assert torch.equal(decoded, actual_decoded.float())


def test_marlin_032_dispatch_boundary_in_isolated_process():
    fixture = Path(__file__).with_name("glm53_package_dispatch_fixture.py")
    # The historical integration registers the same torch op namespace in other tests.
    subprocess.run([
        sys.executable, "-c",
        "import runpy,sys; runpy.run_path(sys.argv[1], run_name='__main__')", str(fixture),
    ], check=True)
