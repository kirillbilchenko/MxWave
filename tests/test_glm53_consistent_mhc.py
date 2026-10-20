"""Guard mHC dispatch, argument semantics, tested geometry and native source bindings."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

import consistent_mhc as adapter


def test_fixed_split_scope_does_not_expand_other_geometries() -> None:
    calls = []

    def native(*args: Any) -> int:
        calls.append(args)
        return 7

    for grid in (1, 4, 32):
        assert adapter.select_fixed_split(native, 64, 16384, grid) == 48
    assert not calls
    for geometry in ((64, 16384, 33), (64, 16384, 0), (32, 16384, 1), (64, None, 1)):
        assert adapter.select_fixed_split(native, *geometry) == 7
    assert len(calls) == 4


def test_native_composition_preserves_norm_default_and_return_order() -> None:
    arguments = {
        "self": object(),
        "x": object(),
        "residual": object(),
        "post_layer_mix": object(),
        "comb_res_mix": object(),
        "fn": object(),
        "norm_weight": object(),
        "norm_eps": 0.0,
        "n_splits": 1,
        "tile_n": 3,
    }
    current = object()
    outputs = tuple(object() for _ in range(3))

    def post(**options: Any) -> object:
        assert options == {
            k: arguments[k] for k in ("x", "residual", "post_layer_mix", "comb_res_mix")
        }
        return current

    def pre(**options: Any) -> tuple[Any, ...]:
        assert options == {
            "residual": current,
            "fn": arguments["fn"],
            "norm_weight": arguments["norm_weight"],
            "norm_eps": 0.0,
            "n_splits": 48,
        }
        return outputs

    assert adapter.compose_post_pre(post, pre, arguments) == (current, *outputs)
    assert arguments["n_splits"] == 1 and arguments["tile_n"] == 3


def fake_native(monkeypatch: Any, *, splits: int = 48) -> tuple[Any, Any]:
    class Fused:
        def forward_cuda(
            self,
            x: Any,
            residual: Any,
            post_layer_mix: Any,
            comb_res_mix: Any,
            fn: Any,
            hc_scale: Any,
            hc_base: Any,
            rms_eps: float,
            hc_pre_eps: float,
            hc_sinkhorn_eps: float,
            hc_post_mult_value: float,
            sinkhorn_repeat: int,
            n_splits: int = 1,
            tile_n: int = 1,
            norm_weight: Any = None,
            norm_eps: float = 0.0,
        ) -> str:
            return "original"

    source = str(Path(__file__))
    kernels = SimpleNamespace(__file__=source, compute_num_split=lambda *args: splits)
    native = SimpleNamespace(
        __file__=source,
        mhc_post_tilelang=lambda **kwargs: "current",
        mhc_pre_tilelang=lambda **kwargs: (kwargs["norm_eps"], kwargs["n_splits"], "input"),
    )
    modules = dict(
        zip(
            adapter.NATIVE_PINS,
            (native, kernels, SimpleNamespace(__file__=source, MHCFusedPostPreOp=Fused)),
            strict=True,
        )
    )
    monkeypatch.setattr(adapter.importlib, "import_module", modules.__getitem__)
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    monkeypatch.setattr(adapter, "NATIVE_PINS", dict.fromkeys(modules, digest))
    return kernels, Fused


def test_class_dispatch_installation_is_idempotent_and_preserves_defaults(monkeypatch: Any) -> None:
    kernels, cls = fake_native(monkeypatch)
    snapshot = adapter.install_consistent_mhc()
    assert adapter.install_consistent_mhc() is snapshot
    assert kernels.compute_num_split(64, 16384, 8) == 48
    args = [object(), SimpleNamespace(shape=(8, 4, 4096)), *[object() for _ in range(10)]]
    assert cls().forward_cuda(*args) == ("current", 0.0, 48, "input")
    assert cls().forward_cuda(*args, norm_eps=1e-6) == ("current", 1e-6, 48, "input")
    args[1] = SimpleNamespace(shape=(2049, 4, 4096))
    assert cls().forward_cuda(*args) == "original"
    proof = snapshot()
    assert proof["native_small_row_splits_checked"]
    assert proof["composed_host_calls_by_rows"] == {"8": 2}
    assert proof["delegated_host_calls_by_shape"] == {"(2049, 4, 4096)": 1}
    proof["composed_host_calls_by_rows"].clear()
    assert snapshot()["composed_host_calls_by_rows"] == {"8": 2}


def test_wrong_device_split_and_changed_native_source_are_rejected(monkeypatch: Any) -> None:
    kernels, _ = fake_native(monkeypatch, splits=64)
    adapter.install_consistent_mhc()
    with pytest.raises(ValueError, match="48-split"):
        kernels.compute_num_split(64, 16384, 1)
    fake_native(monkeypatch)
    monkeypatch.setattr(adapter, "NATIVE_PINS", dict.fromkeys(adapter.NATIVE_PINS, "0" * 64))
    with pytest.raises(ValueError, match="Pinned mHC source"):
        adapter.install_consistent_mhc()
