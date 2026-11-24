"""CPU-only tests for the opt-in MoE scale cache and startup timing."""

from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
# Filled by the module fixture below: the flat integration modules, imported once.
moe_runtime: Any = None
pmw: Any = None
scale_cache: Any = None
scale_tiles: Any = None
REAL_SOURCE_PATHS: Any = None


@pytest.fixture(scope="module", autouse=True)
def flat_modules():
    """Import the flat CT modules without leaking sys.path, the worker stub or its patch.

    model_plugin registers a torch operator and cannot be imported twice, so existing
    sys.modules entries are reused and left in place; only what this module adds on top
    (palette_model_worker, the CombinedWorker stub, model_plugin.pack_scales) is undone.
    """
    import importlib

    path = sys.path[:]
    sys.path[:0] = [str(ROOT / "integrations/glm53_mxfp4"), str(ROOT / "integrations")]
    added_stub = "combined_worker" not in sys.modules
    if added_stub:
        stub = types.ModuleType("combined_worker")
        stub.CombinedWorker = type("CombinedWorker", (), {})
        sys.modules["combined_worker"] = stub
    global moe_runtime, pmw, scale_cache, scale_tiles, REAL_SOURCE_PATHS
    moe_runtime = importlib.import_module("moe_runtime")
    scale_cache = importlib.import_module("scale_cache")
    scale_tiles = importlib.import_module("scale_tiles")
    plugin = importlib.import_module("model_plugin")
    original_pack = plugin.pack_scales
    sys.modules.pop("palette_model_worker", None)
    pmw = importlib.import_module("palette_model_worker")
    REAL_SOURCE_PATHS = scale_cache._source_paths
    yield
    plugin.pack_scales = original_pack
    sys.modules.pop("palette_model_worker", None)
    if added_stub:
        sys.modules.pop("combined_worker", None)
    sys.path[:] = path


SHAPE = (4, 8, 64)  # 2048 bytes = 128 whole vectors


def make_raw(affine: bool = True, seed: int = 0, outliers: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(np.prod(SHAPE))
    raw = (120 + rng.choice(3, n, p=[0.6, 0.3, 0.1])).astype(np.uint8)
    if not affine:
        raw[: n // 3] = 200  # three-value palette that is not consecutive
    raw[rng.choice(n, outliers, replace=False)] = rng.integers(0, 256, outliers, dtype=np.uint8)
    return raw


def entry_from(raw: np.ndarray, shape: tuple[int, ...] = SHAPE) -> Any:
    """Entry built the way the original code builds its staging fields: from the real bytes."""
    from moe_scale_policy import sparse_stage_descriptor

    packed = scale_tiles.pack_sparse(raw)
    sha = hashlib.sha256(raw).hexdigest()
    policy, base = sparse_stage_descriptor(raw, packed.descriptor)
    return scale_cache.Entry(packed.selectors, packed.exceptions, packed.descriptor, shape, sha,
                             int(raw.min()), int(raw.max()), policy, base)


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for var in ("MXWAVE_STARTUP_TIMING", "MXWAVE_SCALE_CACHE_DIR", "MXWAVE_PALETTE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    scale_cache.cache_key.cache_clear()
    scale_cache._STATE.__init__()
    monkeypatch.setattr(scale_cache, "SWITCH_DIR", tmp_path / "switch")
    pmw._source_scales.clear()
    pmw._packed_sources.clear()
    # Deterministic, tmp-only key inputs.
    src = tmp_path / "srcs"
    src.mkdir()
    (src / "a.py").write_text("a")
    ckpt = tmp_path / "model"
    ckpt.mkdir()
    (ckpt / "model.safetensors.index.json").write_text(
        '{"weight_map": {"w": "shard-1.safetensors"}}')
    (ckpt / "shard-1.safetensors").write_bytes(b"12345")
    (ckpt / "config.json").write_text("{}")
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_SOURCE", str(ckpt))
    monkeypatch.setattr(scale_cache, "_source_paths", lambda: {"a": str(src / "a.py")})
    monkeypatch.setattr(scale_cache, "_tp", lambda: (0, 2))
    yield
    scale_cache.cache_key.cache_clear()


@pytest.mark.parametrize("affine", [True, False])
def test_roundtrip_real_pack_sparse(tmp_path, affine):
    raw = make_raw(affine)
    entry = entry_from(raw)
    assert bool(entry.descriptor & (1 << 24)) is affine
    scale_cache.store_entry(tmp_path, 7, entry)
    got = scale_cache.load_entry(tmp_path, 7, SHAPE)
    assert got is not None
    assert np.array_equal(got.selectors, entry.selectors) and got.selectors.dtype == np.uint32
    assert np.array_equal(got.exceptions, entry.exceptions) and got.exceptions.dtype == np.uint8
    assert got.descriptor == entry.descriptor and got.sha256 == entry.sha256
    assert got.shape == SHAPE
    assert (got.raw_min, got.raw_max, got.policy, got.base) == (
        entry.raw_min, entry.raw_max, entry.policy, entry.base)


def test_key_changes(tmp_path, monkeypatch):
    base = scale_cache.cache_key()
    ckpt = tmp_path / "model"

    def changed(mutate) -> str:
        scale_cache.cache_key.cache_clear()
        mutate()
        return scale_cache.cache_key()

    src = tmp_path / "srcs" / "a.py"
    assert changed(lambda: src.write_text("b")) != base
    assert changed(lambda: src.write_text("a")) == base
    assert changed(lambda: (ckpt / "model.safetensors.index.json").write_text("{1}")) != base
    assert changed(lambda: (ckpt / "model.safetensors.index.json").write_text(
        '{"weight_map": {"w": "shard-1.safetensors"}}')) == base
    assert changed(lambda: (ckpt / "config.json").write_text("{2}")) != base
    assert changed(lambda: (ckpt / "shard-1.safetensors").write_bytes(b"123456")) != base
    assert changed(lambda: monkeypatch.setattr(scale_cache, "_tp", lambda: (1, 2))) != base
    monkeypatch.setattr(scale_cache, "_tp", lambda: (0, 2))
    lib = tmp_path / "lib.so"
    lib.write_text("x")
    monkeypatch.setenv("MXWAVE_MOE_ALIGNED_LIBRARY", str(lib))
    with_lib = changed(lambda: None)
    assert with_lib != base
    assert changed(lambda: lib.write_text("y")) != with_lib
    assert changed(lambda: monkeypatch.setattr(scale_cache, "_version", lambda n: "9.9")) != with_lib


def test_key_is_computed_once(monkeypatch):
    calls = []
    monkeypatch.setattr(scale_cache, "key_payload", lambda: calls.append(1) or {
        "checkpoint": {"c": "x"}, "shards": {"s": 1}, "sources": {}, "aligned_library": "x",
        "vllm": "1"})
    scale_cache.cache_key.cache_clear()
    scale_cache.cache_key()
    scale_cache.cache_key()
    assert len(calls) == 1


def test_corrupt_missing_mismatch_fall_back(tmp_path):
    raw = make_raw()
    scale_cache.store_entry(tmp_path, 0, entry_from(raw))
    path = scale_cache._entry_path(tmp_path, 0)
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is not None
    assert scale_cache.load_entry(tmp_path, 1, SHAPE) is None  # missing
    assert scale_cache.load_entry(tmp_path, 0, (4, 8, 128)) is None  # shape mismatch
    assert scale_cache.load_entry(tmp_path, 3, SHAPE) is None  # index mismatch via name
    path.write_bytes(path.read_bytes()[:100])
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is None  # truncated
    path.write_bytes(b"not a zip")
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is None
    # Wrong dtype: selectors saved as int64.
    bad = entry_from(raw)
    bad.selectors = bad.selectors.astype(np.int64)
    scale_cache.store_entry(tmp_path, 0, bad)
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is None


def fake_ops(log: list[str]):
    real_upload = moe_runtime.sparse_upload_ranged

    def pack(raw):
        log.append("pack")
        return scale_tiles.pack_sparse(raw)

    def verify(packed, raw):
        log.append("verify")
        moe_runtime.sparse_cpu_verify(packed, raw)

    def upload(sel, exc, desc, shape, device, raw_min, raw_max):
        log.append("upload")
        assert device == torch.device("cpu")
        return real_upload(sel, exc, desc, shape, device, raw_min, raw_max)
    return {"pack": pack, "verify": verify, "upload": upload, "on_sha": lambda sha: None}


def scales_tensor(raw: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(raw.copy()).reshape(SHAPE)  # uint8 stands in for E8M0


class Viewable:
    """CPU stand-in for an E8M0 tensor: .view(torch.uint8) is the identity."""

    def __init__(self, t: torch.Tensor) -> None:
        self.t, self.shape, self.device = t, t.shape, t.device

    def view(self, dtype):
        return self.t


def test_pack_group_miss_then_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path / "c"))
    raw = make_raw()
    scales = Viewable(scales_tensor(raw))
    log: list[str] = []
    packed, sha = scale_cache.pack_group(scales, 0, **fake_ops(log))
    assert not scale_cache._entry_path(tmp_path / "c", 0).exists()  # writes wait for commit
    scale_cache.commit(packed)
    assert log == ["pack", "verify", "upload"]
    assert scale_cache._STATE.last == "miss"
    assert list((tmp_path / "c" / scale_cache.cache_key()).glob(".*")) == []
    log.clear()
    packed2, sha2 = scale_cache.pack_group(scales, 0, **fake_ops(log))
    assert log == ["upload"] and sha2 == sha and scale_cache._STATE.last == "hit"
    assert torch.equal(packed.records, packed2.records)
    assert torch.equal(packed.exceptions, packed2.exceptions)
    assert packed2.descriptor[0].item() == packed2.records.data_ptr()  # fresh pointers
    assert packed2.descriptor[2].item() == packed.descriptor[2].item()
    assert packed2.stage_metadata_verified and packed2.aligned_nibble_base == packed.aligned_nibble_base
    assert scale_cache.hit_index(packed2) == 0 and scale_cache.hit_index(packed) is None
    assert scale_cache._STATE.hit_groups == 1 and scale_cache._STATE.groups == 2


def test_pack_group_corrupt_entry_is_rewritten(tmp_path, monkeypatch):
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path))
    scales = Viewable(scales_tensor(make_raw()))
    scale_cache.commit(scale_cache.pack_group(scales, 0, **fake_ops([]))[0])
    scale_cache._entry_path(tmp_path, 0).write_bytes(b"junk")
    log: list[str] = []
    scale_cache.commit(scale_cache.pack_group(scales, 0, **fake_ops(log))[0])
    assert log == ["pack", "verify", "upload"]
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is not None


def test_invalidate_deletes_entry_and_retry_does_not_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path))
    scales = Viewable(scales_tensor(make_raw()))
    scale_cache.commit(scale_cache.pack_group(scales, 0, **fake_ops([]))[0])
    packed, _ = scale_cache.pack_group(scales, 0, **fake_ops([]))
    assert scale_cache.hit_index(packed) == 0
    scale_cache.invalidate(packed)
    assert not scale_cache._entry_path(tmp_path, 0).exists()
    assert scale_cache.pop_stale_id() == id(packed)
    assert scale_cache._STATE.bypass
    with scale_cache.retry_uncached(0):
        log: list[str] = []
        redone, _ = scale_cache.pack_group(scales, 0, **fake_ops(log))
    assert log == ["pack", "verify", "upload"]
    assert not scale_cache._entry_path(tmp_path, 0).exists()
    scale_cache.commit(redone)  # a proven retry repairs the cache
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is not None
    assert scale_cache._STATE.hit_groups == 0 and scale_cache._STATE.groups == 2


def test_atomic_write_leaves_no_tmp(tmp_path):
    scale_cache.store_entry(tmp_path, 0, entry_from(make_raw()))
    names = sorted(p.name for p in (tmp_path / scale_cache.cache_key()).iterdir())
    assert names == ["000.npz"]


def test_write_oserror_only_warns(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    entry = entry_from(make_raw())
    scale_cache.store_entry(tmp_path, 0, entry)
    scale_cache.store_entry(tmp_path, 1, entry)
    out = capsys.readouterr().out
    assert out.count("write failed") == 1
    assert list((tmp_path / scale_cache.cache_key()).iterdir()) == []  # tmp cleaned up


def palette_env(monkeypatch, calls):
    native = SimpleNamespace(encoding="sparse_palette", affine_only=True)
    monkeypatch.setattr(pmw.model_plugin, "native", lambda kind: native)
    sentinel = SimpleNamespace(affine_palette=True)

    def original(scales, *args, encoding, **kwargs):
        calls.append((scales, args, encoding, kwargs))
        return sentinel
    monkeypatch.setattr(pmw, "_original_pack", original)
    return sentinel


def test_default_path_is_original(tmp_path, monkeypatch):
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("MXWAVE_PALETTE_MODEL", "1")
    calls: list[Any] = []
    sentinel = palette_env(monkeypatch, calls)
    monkeypatch.setattr(pmw.scale_cache, "pack_group",
                        lambda *a, **k: pytest.fail("cache path used with env unset"))
    scales = torch.zeros(1, 1, 16, dtype=torch.uint8)
    assert pmw.model_plugin.pack_scales(scales, "codec", encoding="aligned_nibbles") is sentinel
    assert calls == [(scales, ("codec",), "sparse_palette", {})]
    assert pmw._source_scales[0]["shape"] == [1, 1, 16]
    assert pmw._packed_sources[id(sentinel)] == pmw._source_scales[0]["sha256"]
    assert list(work.iterdir()) == []
    assert pmw.model_plugin.pack_scales is pmw._palette_pack


def test_palette_off_never_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("MXWAVE_STARTUP_TIMING", "1")
    calls: list[Any] = []
    palette_env(monkeypatch, calls)
    monkeypatch.setattr(pmw.scale_cache, "pack_group",
                        lambda *a, **k: pytest.fail("cache used without palette model"))
    pmw._palette_pack(torch.zeros(1, 1, 16, dtype=torch.uint8), "codec",
                      encoding="aligned_nibbles")
    pmw._palette_pack(torch.zeros(1, 1, 16, dtype=torch.uint8), "codec", encoding="nibbles")
    assert [c[2] for c in calls] == ["aligned_nibbles", "nibbles"]
    assert not (tmp_path / "c").exists()


def test_instrumented_pack_registers_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("MXWAVE_PALETTE_MODEL", "1")
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path))
    palette_env(monkeypatch, [])
    monkeypatch.setattr(pmw.moe_runtime, "check_scales", lambda s: None)
    ops = fake_ops([])
    monkeypatch.setattr(pmw.moe_runtime, "sparse_cpu_pack", ops["pack"])
    monkeypatch.setattr(pmw.moe_runtime, "sparse_upload_ranged", ops["upload"])
    raw = make_raw()
    scales = Viewable(scales_tensor(raw))
    scales.view = lambda dtype: scales.t  # type: ignore[method-assign]
    packed = pmw._palette_pack(scales, "codec", encoding="aligned_nibbles")
    assert pmw._source_scales == [{"shape": list(SHAPE), "sha256": entry_from(raw).sha256}]
    assert pmw._packed_sources[id(packed)] == pmw._source_scales[0]["sha256"]
    assert packed.encoding == "sparse_palette" and packed.affine_palette
    packed2 = pmw._palette_pack(scales, "codec", encoding="aligned_nibbles")  # group index 1: miss
    assert len(pmw._source_scales) == 2 and packed2 is not packed


def test_summary_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path))
    scale_cache._STATE.groups, scale_cache._STATE.hit_groups = 84, 84
    scale_cache.log_summary()
    line = capsys.readouterr().out.splitlines()[-1]
    assert line.startswith("MxWave scale cache: hit key=") and "groups=84/84" in line
    scale_cache._STATE.hit_groups = 10
    scale_cache.log_summary()
    assert "cache: mixed" in capsys.readouterr().out


def test_sparse_helpers_equal_original_behavior():
    raw = make_raw()
    packed = moe_runtime.sparse_cpu_pack(raw)
    moe_runtime.sparse_cpu_verify(packed, raw)
    with pytest.raises(ValueError, match="round-trip"):
        moe_runtime.sparse_cpu_verify(packed, raw ^ 1)
    from moe_scale_policy import sparse_stage_descriptor

    policy, base = sparse_stage_descriptor(raw, packed.descriptor)
    out = moe_runtime.sparse_upload(packed.selectors, packed.exceptions, packed.descriptor,
                                    SHAPE, torch.device("cpu"), policy, base)
    assert out.encoding == "sparse_palette" and out.shape == SHAPE
    assert out.descriptor.tolist()[2] == policy and out.aligned_nibble_base == base
    assert out.stage_metadata_verified and out.affine_palette


def run_prepare_layer(monkeypatch, tmp_path, corrupt_decodes: int, layers: int = 1,
                      raws: list[np.ndarray] | None = None):
    """Drive the real prepare_layer on CPU with a fake native decoder."""
    import model_plugin as mp

    monkeypatch.setenv("MXWAVE_PALETTE_MODEL", "1")
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("MXWAVE_MOE_RECORD_CODEC", "codec")
    monkeypatch.setattr(pmw.moe_runtime, "check_scales", lambda s: None)
    decodes = {"n": 0}

    def decode(packed):
        decodes["n"] += 1
        out = torch.from_numpy(scale_tiles.unpack_sparse(SimpleNamespace(
            count=int(np.prod(packed.shape)), palette=tuple(
                int(packed.descriptor[2]) >> s & 255 for s in (0, 8, 16)),
            selectors=packed.records.numpy().view(np.uint32),
            exceptions=packed.exceptions.numpy()))).reshape(packed.shape)
        if decodes["n"] <= corrupt_decodes:
            out = out ^ 1
        return out

    native = SimpleNamespace(encoding="sparse_palette", affine_only=True, decode=decode)
    monkeypatch.setattr(mp, "native", lambda kind: native)
    monkeypatch.setattr(mp, "_replace_config_scales", lambda *a: 1)
    monkeypatch.setattr(mp, "_STORAGE", {})
    monkeypatch.setattr(mp, "_LAYERS", [])
    method = SimpleNamespace(use_cutlass_mxfp4=False, moe=SimpleNamespace(layer_name="L"))
    kept = []  # the real model keeps every layer (and its replacement parameters) alive
    for n in range(layers):
        layer = torch.nn.Module()
        kept.append(layer)
        for k, name in enumerate(("w13_weight_scale", "w2_weight_scale")):
            data = raws[2 * n + k] if raws else make_raw(seed=len(name) + 7 * n)
            raw = torch.from_numpy(data.copy()).reshape(SHAPE).view(torch.float8_e8m0fnu)
            setattr(layer, name, torch.nn.Parameter(raw, requires_grad=False))
        mp.prepare_layer(method, layer, "aligned")
    return layer, decodes


def test_prepare_layer_cache_hit_and_gpu_check_fallback(tmp_path, monkeypatch, capsys):
    run_prepare_layer(monkeypatch, tmp_path, 0)  # cold miss writes 2 entries
    root = tmp_path / "c" / scale_cache.cache_key()
    assert sorted(p.name for p in root.iterdir()) == ["000.npz", "001.npz"]
    pmw._source_scales.clear()
    scale_cache._STATE.__init__()
    _, decodes = run_prepare_layer(monkeypatch, tmp_path, 0)  # warm hit
    assert decodes["n"] == 2 and scale_cache._STATE.hit_groups == 2
    assert len(pmw._source_scales) == 2
    # Hit whose GPU check fails once: entry invalidated, redone uncached, then passes.
    pmw._source_scales.clear()
    scale_cache._STATE.__init__()
    _, decodes = run_prepare_layer(monkeypatch, tmp_path, 1)
    assert decodes["n"] == 3  # 1 bad hit, 1 good retry, 1 uncached group after the bypass
    assert scale_cache._STATE.hit_groups == 0 and scale_cache._STATE.bypass
    assert len(pmw._source_scales) == 2
    assert scale_cache.load_entry(tmp_path / "c", 0, SHAPE) is not None  # self-repaired
    # Same corruption on a cold (non-cache) path keeps the original RuntimeError.
    import shutil
    shutil.rmtree(tmp_path / "c")
    pmw._source_scales.clear()
    scale_cache._STATE.__init__()
    with pytest.raises(RuntimeError, match="differ after GPU decode"):
        run_prepare_layer(monkeypatch, tmp_path, 1)
    # Bad hit and bad retry also raises.
    pmw._source_scales.clear()
    scale_cache._STATE.__init__()
    run_prepare_layer(monkeypatch, tmp_path, 0)
    pmw._source_scales.clear()
    scale_cache._STATE.__init__()
    with pytest.raises(RuntimeError, match="differ after GPU decode"):
        run_prepare_layer(monkeypatch, tmp_path, 2)
    assert "invalidated" in capsys.readouterr().out


def test_timing_lines(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MXWAVE_STARTUP_TIMING", "1")
    run_prepare_layer(monkeypatch, tmp_path, 0)
    out = capsys.readouterr().out.splitlines()
    lines = [x for x in out if x.startswith("MxWave timing: layer=1")]
    assert [x.split()[3] for x in lines] == ["name=w13", "name=w2"]
    for field in ("repack=", "d2h=", "sha=", "pack=", "roundtrip=", "upload=", "gpu_check=",
                  "cache=miss"):
        assert field in lines[0]


def tamper(path, **overrides):
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays.update(overrides)
    np.savez(path, **arrays)


def set_descriptor(path, value: int):
    with np.load(path, allow_pickle=False) as data:
        meta = data["meta"].copy()
    meta[2] = value
    tamper(path, meta=meta)


@pytest.mark.parametrize("mutate", ["bit24", "bit25", "high_bit", "dup_palette", "bad_escape",
                                    "code3", "exception_count"])
def test_entry_semantic_validation(tmp_path, mutate):
    entry = entry_from(make_raw())
    assert entry.exceptions.shape[0] > 0 and entry.descriptor >> 24 == 3
    scale_cache.store_entry(tmp_path, 0, entry)
    path = scale_cache._entry_path(tmp_path, 0)
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is not None
    d = entry.descriptor
    if mutate == "bit24":
        set_descriptor(path, d ^ (1 << 24))
    elif mutate == "bit25":
        set_descriptor(path, d ^ (1 << 25))
    elif mutate == "high_bit":
        set_descriptor(path, d | (1 << 40))
    elif mutate == "dup_palette":
        set_descriptor(path, (d & ~0xff00) | ((d & 255) << 8))
    elif mutate == "bad_escape":
        sel = entry.selectors.copy()
        sel[np.flatnonzero(sel >> 30 == 3)[0]] = 0xc0000000 | len(entry.exceptions) + 5
        tamper(path, selectors=sel)
    elif mutate == "code3":
        sel = entry.selectors.copy()
        sel[np.flatnonzero(sel >> 30 != 3)[0]] |= 0x3  # code 3 in a non-escaped word
        tamper(path, selectors=sel)
    else:
        tamper(path, exceptions=entry.exceptions[:-1])
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is None


def test_descriptor_matches_scale_tiles_flags():
    for palette, flags in (((8, 9, 10), 3), ((9, 10, 11), 1), ((8, 9, 11), 0)):
        assert scale_tiles.sparse_descriptor(palette) >> 24 == flags
        assert scale_tiles.SparseScales(16, palette, None, None).descriptor == \
            scale_tiles.sparse_descriptor(palette)


def test_retry_valueerror_records_fallback(tmp_path, monkeypatch):
    run_prepare_layer(monkeypatch, tmp_path, 0)  # warm the cache
    pmw._source_scales.clear()
    scale_cache._STATE.__init__()
    real = moe_runtime.sparse_cpu_pack

    def pack(raw):
        if scale_cache.retry_index() is not None:
            raise ValueError("retry cannot pack")
        return real(raw)
    monkeypatch.setattr(pmw.moe_runtime, "sparse_cpu_pack", pack)
    layer, _ = run_prepare_layer(monkeypatch, tmp_path, 1)  # first hit fails GPU proof
    scales = layer.mxwave_scale_qualification["scales"]
    assert scales[0]["fallback"] == "retry cannot pack" and "fallback" not in scales[1]
    assert len(pmw._source_scales) == 2 and True


def test_fallback_group_keeps_baseline_indices(tmp_path, monkeypatch):
    raws = [make_raw(affine=False), make_raw(seed=5)]
    layer, _ = run_prepare_layer(monkeypatch, tmp_path, 0, raws=raws)
    scales = layer.mxwave_scale_qualification["scales"]
    assert "Nonconsecutive" in scales[0]["fallback"] and "fallback" not in scales[1]
    assert [e["sha256"] for e in pmw._source_scales] == [
        hashlib.sha256(r).hexdigest() for r in raws]
    root = tmp_path / "c" / scale_cache.cache_key()
    assert sorted(p.name for p in root.iterdir()) == ["001.npz"]  # index 1, as in the baseline
    assert pmw._source_scales[1]["sha256"] == scale_cache.load_entry(
        tmp_path / "c", 1, SHAPE).sha256


def test_cache_exceptions_degrade_to_uncached(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path / "c"))
    scales = Viewable(scales_tensor(make_raw()))

    def boom(*a, **k):
        raise KeyError("boom")
    # Key computation failure.
    monkeypatch.setattr(scale_cache, "key_payload", boom)
    scale_cache.cache_key.cache_clear()
    log: list[str] = []
    packed, _ = scale_cache.pack_group(scales, 0, **fake_ops(log))
    scale_cache.commit(packed)
    assert log == ["pack", "verify", "upload"] and scale_cache._STATE.disabled
    assert not (tmp_path / "c").exists()
    assert capsys.readouterr().out.count("continuing without caching") == 1
    # Load failure (not just OSError).
    monkeypatch.undo()
    scale_cache._STATE.__init__()
    scale_cache.cache_key.cache_clear()
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path / "c2"))
    monkeypatch.setattr(scale_cache, "load_entry", boom)
    log.clear()
    packed, _ = scale_cache.pack_group(scales, 0, **fake_ops(log))
    assert log == ["pack", "verify", "upload"]
    # Store failure with a non-OSError.
    monkeypatch.undo()
    scale_cache._STATE.__init__()
    scale_cache.cache_key.cache_clear()
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path / "c3"))
    monkeypatch.setattr(np, "savez", boom)
    packed, _ = scale_cache.pack_group(scales, 0, **fake_ops([]))
    scale_cache.commit(packed)  # must not raise
    assert scale_cache._STATE.disabled


def test_stale_tmp_cleanup_and_other_key_report(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(root))
    key_dir = root / scale_cache.cache_key()
    key_dir.mkdir(parents=True)
    (key_dir / ".000.npz.tmp123").write_bytes(b"x")
    other = root / ("ab" * 32)
    (root / "ENABLE").write_text("")  # marker files are never key dirs
    other.mkdir()
    (other / "000.npz").write_bytes(b"y" * 10)
    scale_cache.pack_group(Viewable(scales_tensor(make_raw())), 0, **fake_ops([]))
    assert list(key_dir.iterdir()) == [] and (other / "000.npz").exists()
    assert "1 other key dirs" in capsys.readouterr().out


def test_key_gaps_logged_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scale_cache, "_source_paths", lambda: {"x": str(tmp_path / "nope")})
    scale_cache.cache_key.cache_clear()
    scale_cache.cache_key()
    scale_cache.cache_key()
    out = capsys.readouterr().out
    assert out.count("key inputs unresolved") == 1 and "sources.x" in out


def test_inactive_cache_with_palette_off(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path / "c"))
    palette_env(monkeypatch, [])
    for _ in range(2):
        pmw._palette_pack(torch.zeros(1, 1, 16, dtype=torch.uint8), "codec",
                          encoding="aligned_nibbles")
    scale_cache.log_summary()
    out = capsys.readouterr().out
    assert out.count("inactive") == 1 and "groups=" not in out


def drive_report(monkeypatch, tmp_path, corrupt: int):
    """Run 42 layers with real objects, then the real palette report invariants."""
    import model_plugin as mp

    dist = types.ModuleType("vllm.distributed")
    dist.get_tensor_model_parallel_rank = lambda: 0
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.distributed", dist)
    pmw._source_scales.clear()
    pmw._packed_sources.clear()
    scale_cache._STATE.__init__()
    run_prepare_layer(monkeypatch, tmp_path, corrupt, layers=42)
    monkeypatch.setattr(mp, "report", lambda: {
        "layers": mp._LAYERS, "layers_seen": len(mp._LAYERS)})
    worker = pmw.PaletteModelWorker()
    worker.mxwave_scale_report = mp.report
    return worker.mxwave_palette_report()


@pytest.mark.parametrize("phase", ["miss", "hit", "redo"])
def test_palette_report_after_miss_hit_redo(tmp_path, monkeypatch, phase):
    if phase != "miss":
        drive_report(monkeypatch, tmp_path, 0)
    report = drive_report(monkeypatch, tmp_path, 1 if phase == "redo" else 0)
    assert report["status"] == "passed" and report["scale_groups"] == 84
    assert len(pmw._source_scales) == 84 and len(pmw._packed_sources) == 84
    assert {e["sha256"] for e in report["source_scale_hashes"]} == set(pmw._packed_sources.values())
    import model_plugin as mp
    live = {id(s["packed"]) for s in mp._STORAGE.values() if s["kind"] == 1}
    assert set(pmw._packed_sources) == live  # no stale ids, none missing
    assert scale_cache._STATE.hit_groups == (84 if phase == "hit" else 0)


def _marker(tmp_path, name):
    switch = tmp_path / "switch"
    switch.mkdir(exist_ok=True)
    (switch / name).write_text("")
    return switch


def test_markers_enable_timing_and_cache(tmp_path, capsys):
    switch = _marker(tmp_path, "TIMING")
    _marker(tmp_path, "ENABLE")
    assert scale_cache.timing_enabled() and scale_cache.cache_dir() == switch
    assert scale_cache.active()
    out = capsys.readouterr().out
    assert f"enabled by {switch / 'TIMING'}" in out and f"enabled by {switch / 'ENABLE'}" in out
    scale_cache.timing_enabled(), scale_cache.cache_dir()
    assert capsys.readouterr().out == ""  # logged once


def test_env_zero_overrides_markers(tmp_path, monkeypatch):
    _marker(tmp_path, "TIMING")
    _marker(tmp_path, "ENABLE")
    monkeypatch.setenv("MXWAVE_STARTUP_TIMING", "0")
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", "off")
    assert not scale_cache.timing_enabled() and scale_cache.cache_dir() is None
    assert not scale_cache.active()


@pytest.mark.parametrize("value", ["", "0"])
def test_env_cache_disabled_values(tmp_path, monkeypatch, value):
    _marker(tmp_path, "ENABLE")
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", value)
    assert scale_cache.cache_dir() is None


def test_env_path_overrides_marker(tmp_path, monkeypatch):
    _marker(tmp_path, "ENABLE")
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path / "mine"))
    monkeypatch.setenv("MXWAVE_STARTUP_TIMING", "1")
    assert scale_cache.cache_dir() == tmp_path / "mine" and scale_cache.timing_enabled()


def test_no_markers_is_off_and_writes_nothing(tmp_path, capsys):
    assert not scale_cache.timing_enabled() and scale_cache.cache_dir() is None
    assert not scale_cache.active()
    assert not (tmp_path / "switch").exists() and capsys.readouterr().out == ""


def test_marker_check_error_is_off(tmp_path, monkeypatch):
    _marker(tmp_path, "TIMING")
    _marker(tmp_path, "ENABLE")

    def boom(path):
        raise PermissionError(path)

    monkeypatch.setattr(os.path, "exists", boom)
    assert not scale_cache.timing_enabled() and scale_cache.cache_dir() is None


def test_switches_resolved_once_per_process(tmp_path, monkeypatch):
    assert not scale_cache.timing_enabled() and scale_cache.cache_dir() is None
    _marker(tmp_path, "TIMING")
    _marker(tmp_path, "ENABLE")
    monkeypatch.setenv("MXWAVE_STARTUP_TIMING", "1")
    assert not scale_cache.timing_enabled() and scale_cache.cache_dir() is None
    scale_cache._STATE.__init__()  # reset
    assert scale_cache.timing_enabled() and scale_cache.cache_dir() is not None


def test_key_dir_report_ignores_marker_files(tmp_path, capsys):
    switch = _marker(tmp_path, "ENABLE")
    (switch / "TIMING").write_text("")
    (switch / "notes").mkdir()
    (switch / ("cd" * 32)).mkdir()
    (switch / ("cd" * 32) / "000.npz").write_bytes(b"z")
    scale_cache.pack_group(Viewable(scales_tensor(make_raw())), 0, **fake_ops([]))
    assert "1 other key dirs" in capsys.readouterr().out
    assert (switch / "ENABLE").exists() and (switch / "TIMING").exists()


# --- Fields the staged consumer needs (PackedScales.stage_metadata_verified, aligned_nibble_base,
# --- and the policy word at descriptor[2]) ---------------------------------------------------


def narrow_raw() -> np.ndarray:
    """No outliers: every byte in one 16-bucket, so a nibble base is available."""
    return make_raw(outliers=0)


def near_constant_raw() -> np.ndarray:
    """One value only: two of three palette entries are padding that no source byte uses."""
    return np.full(int(np.prod(SHAPE)), 121, dtype=np.uint8)


def exception_only_raw() -> np.ndarray:
    """Every 16-byte vector holds more than three distinct values: all words are escapes."""
    rng = np.random.default_rng(3)
    return np.tile(rng.permutation(256).astype(np.uint8)[:16], int(np.prod(SHAPE)) // 16)


def straddling_raw() -> np.ndarray:
    """A short numeric span across a 16-bucket boundary: fallback policy, no nibble base."""
    raw = narrow_raw()
    raw[5] = 111
    return raw


RAWS = {"narrow": narrow_raw, "outliers": make_raw, "non_affine": lambda: make_raw(affine=False),
        "constant": near_constant_raw, "all_exceptions": exception_only_raw,
        "straddling": straddling_raw}


@pytest.mark.parametrize("kind", sorted(RAWS))
def test_sparse_range_is_exact_for_valid_buffers(kind):
    raw = RAWS[kind]()
    packed = scale_tiles.pack_sparse(raw)
    assert scale_cache.sparse_range(packed.palette, packed.selectors, packed.exceptions) == (
        int(raw.min()), int(raw.max()))


def test_sparse_range_spans_chunk_boundary():
    n = (1 << 20) * 16 + 64  # one more selector chunk than a million words
    raw = np.full(n, 120, dtype=np.uint8)
    raw[-3] = 122
    packed = scale_tiles.pack_sparse(raw)
    assert scale_cache.sparse_range(packed.palette, packed.selectors, packed.exceptions) == (
        120, 122)
    raw[-1] = 5  # a final exception vector
    packed = scale_tiles.pack_sparse(raw)
    assert scale_cache.sparse_range(packed.palette, packed.selectors, packed.exceptions) == (
        5, 122)


@pytest.mark.parametrize("kind", sorted(RAWS))
def test_cache_hit_restores_every_consumer_field(tmp_path, monkeypatch, kind):
    """Miss then hit must equal the cache-off moe_runtime.pack_scales result, field by field."""
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setattr(moe_runtime, "check_scales", lambda s: None)
    raw = RAWS[kind]()
    scales = scales_tensor(raw)
    reference = moe_runtime.pack_scales(scales, Path("codec"), encoding="sparse_palette")
    assert reference.stage_metadata_verified
    miss, _ = scale_cache.pack_group(Viewable(scales), 0, **fake_ops([]))
    scale_cache.commit(miss)
    hit, _ = scale_cache.pack_group(Viewable(scales), 0, **fake_ops([]))
    assert scale_cache.hit_index(hit) == 0
    for got in (miss, hit):
        assert got.shape == reference.shape and got.encoding == reference.encoding
        assert got.affine_palette == reference.affine_palette
        assert got.stage_metadata_verified is True
        assert got.aligned_nibble_base == reference.aligned_nibble_base
        assert got.descriptor[2].item() == reference.descriptor[2].item()
        assert torch.equal(got.records, reference.records)
        assert torch.equal(got.exceptions, reference.exceptions)
        assert got.retained_bytes == reference.retained_bytes
    assert (kind in ("narrow", "constant")) == (reference.aligned_nibble_base is not None)


def test_pack_scales_sparse_branch_is_unchanged_by_the_refactor(monkeypatch):
    """Reference semantics straight from the policy and scale_tiles, independent of the helpers."""
    from moe_scale_policy import STAGE_BASE_SHIFT, STAGE_NIBBLES, STAGE_VERIFIED

    monkeypatch.setattr(moe_runtime, "check_scales", lambda s: None)
    raw = narrow_raw()
    out = moe_runtime.pack_scales(scales_tensor(raw), Path("codec"), encoding="sparse_palette")
    packed = scale_tiles.pack_sparse(raw)
    base = int(raw.min()) & ~15
    expected = packed.descriptor | STAGE_VERIFIED | STAGE_NIBBLES | base << STAGE_BASE_SHIFT
    assert out.descriptor[2].item() == expected and out.aligned_nibble_base == base
    assert out.stage_metadata_verified and out.encoding == "sparse_palette"
    assert out.affine_palette == bool(packed.descriptor & (1 << 24))
    assert out.descriptor[0].item() == out.records.data_ptr()
    assert out.descriptor[1].item() == out.exceptions.data_ptr()


def test_cached_fallback_policy_has_no_base(tmp_path, monkeypatch):
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path / "c"))
    scales = Viewable(scales_tensor(straddling_raw()))
    miss, _ = scale_cache.pack_group(scales, 0, **fake_ops([]))
    scale_cache.commit(miss)
    hit, _ = scale_cache.pack_group(scales, 0, **fake_ops([]))
    assert miss.aligned_nibble_base is None and hit.aligned_nibble_base is None
    assert hit.stage_metadata_verified
    from moe_scale_policy import STAGE_NIBBLES, STAGE_VERIFIED
    word = hit.descriptor[2].item()
    assert word & STAGE_VERIFIED and not word & STAGE_NIBBLES
    assert word == miss.descriptor[2].item()


def meta_with(path, **fields):
    names = ("version", "index", "descriptor", "n_sel", "n_exc", "raw_min", "raw_max", "policy",
             "base")
    with np.load(path, allow_pickle=False) as data:
        meta = data["meta"].copy()
    for key, value in fields.items():
        meta[names.index(key)] = value
    tamper(path, meta=meta)


@pytest.mark.parametrize("raw_fn", [narrow_raw, make_raw, near_constant_raw])
@pytest.mark.parametrize("field,delta", [
    ("raw_min", -1), ("raw_min", 1), ("raw_max", -1), ("raw_max", 1), ("raw_max", 300),
    ("policy", 1), ("policy", 1 << 40), ("policy", -(1 << 41)), ("base", 16), ("base", -1),
    ("base", -2), ("base", 3), ("version", -1)])
def test_stored_staging_fields_are_validated(tmp_path, raw_fn, field, delta):
    entry = entry_from(raw_fn())
    scale_cache.store_entry(tmp_path, 0, entry)
    path = scale_cache._entry_path(tmp_path, 0)
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is not None
    current = {"raw_min": entry.raw_min, "raw_max": entry.raw_max, "policy": entry.policy,
               "base": -1 if entry.base is None else entry.base,
               "version": scale_cache.FORMAT_VERSION}[field]
    value = delta if field == "base" and delta in (-1, -2) else current + delta
    if value == current:
        pytest.skip("mutation equals the stored value")
    if field == "base" and delta == 16:
        value = (entry.base + 16) % 256 if entry.base is not None else 16
    meta_with(path, **{field: value})
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is None


def test_base_flip_in_both_directions_is_rejected(tmp_path):
    """A nibble-capable entry must not lose its base, nor a fallback entry gain one."""
    for raw_fn, bogus in ((narrow_raw, -1), (straddling_raw, 112)):
        entry = entry_from(raw_fn())
        assert (entry.base is None) == (bogus != -1)
        scale_cache.store_entry(tmp_path, 0, entry)
        meta_with(scale_cache._entry_path(tmp_path, 0), base=bogus)
        assert scale_cache.load_entry(tmp_path, 0, SHAPE) is None


def test_old_format_entry_is_never_reused(tmp_path):
    assert scale_cache.FORMAT_VERSION >= 2
    entry = entry_from(narrow_raw())
    scale_cache.store_entry(tmp_path, 0, entry)
    path = scale_cache._entry_path(tmp_path, 0)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    legacy = arrays["meta"][:5].copy()  # the version-1 meta layout, without staging fields
    legacy[0] = 1
    tamper(path, meta=legacy)
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is None
    meta_with(path, version=1)  # version-1 number on a current-layout entry is also refused
    assert scale_cache.load_entry(tmp_path, 0, SHAPE) is None


def test_rejected_staging_entry_is_a_miss_that_repacks(tmp_path, monkeypatch):
    monkeypatch.setenv("MXWAVE_SCALE_CACHE_DIR", str(tmp_path))
    scales = Viewable(scales_tensor(narrow_raw()))
    scale_cache.commit(scale_cache.pack_group(scales, 0, **fake_ops([]))[0])
    meta_with(scale_cache._entry_path(tmp_path, 0), policy=7)
    log: list[str] = []
    packed, _ = scale_cache.pack_group(scales, 0, **fake_ops(log))
    assert log == ["pack", "verify", "upload"] and scale_cache.hit_index(packed) is None
    assert packed.aligned_nibble_base == 112


def test_key_covers_every_defining_source_file():
    scale_cache._STATE.__init__()
    paths = REAL_SOURCE_PATHS()
    for name in ("moe_runtime", "palette_model_worker", "model_plugin", "scale_tiles",
                 "scale_cache", "moe_scale_policy"):
        assert paths[name] and Path(paths[name]).name == f"{name}.py", name
    assert Path(paths["scale_tiles"]).parent == ROOT / "integrations"
    assert Path(paths["moe_scale_policy"]).parent == ROOT / "integrations/glm53_mxfp4"


def test_key_changes_with_scale_policy_file(tmp_path, monkeypatch):
    policy = tmp_path / "moe_scale_policy.py"
    policy.write_text("a")
    monkeypatch.setattr(scale_cache, "_source_paths", lambda: {"moe_scale_policy": str(policy)})
    scale_cache.cache_key.cache_clear()
    before = scale_cache.cache_key()
    policy.write_text("b")
    scale_cache.cache_key.cache_clear()
    assert scale_cache.cache_key() != before


def test_check_sparse_structure_direct():
    packed = scale_tiles.pack_sparse(make_raw())
    sel, n = packed.selectors, len(packed.exceptions)
    assert scale_tiles.check_sparse_structure(sel, n)
    assert not scale_tiles.check_sparse_structure(sel, n + 1)
    swapped = sel.copy()
    escaped = np.flatnonzero(swapped >> 30 == 3)
    swapped[escaped[0]], swapped[escaped[1]] = swapped[escaped[1]], swapped[escaped[0]]
    assert not scale_tiles.check_sparse_structure(swapped, n)
