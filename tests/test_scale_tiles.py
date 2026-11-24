"""Scale tiles must preserve all U8 patterns, chunk boundaries and exception ranks."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the experimental model-independent codec without loading a GPU runtime."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations"))


@pytest.mark.parametrize("mode", ["mask128", "palette64"])
def test_exact_patterns_lengths_and_chunk_boundary(mode: str) -> None:
    codec = importlib.import_module("scale_tiles")
    rng = np.random.default_rng(737)
    for size in (1, 15, 16, 63, 64, 65, 127, 128, 129, 1024**2 + 17):
        raw = rng.integers(0, 256, size=size, dtype=np.uint8)
        packed = codec.pack(raw, mode)
        assert np.array_equal(codec.unpack(packed), raw)
        assert packed.retained_bytes == sum(a.nbytes for a in
            (packed.selectors, packed.offsets, packed.exceptions)) + 32
    # Every 16-bit exception mask in its own 16-byte vector, alternating raw
    # values beyond any three-symbol palette, plus data fixing the primary value.
    masks = np.arange(65536, dtype=np.uint32)
    bits = ((masks[:, None] >> np.arange(16)) & 1).astype(np.uint8)
    raw = np.concatenate((np.zeros(1024**2, dtype=np.uint8), (bits * 255).reshape(-1)))
    assert np.array_equal(codec.unpack(codec.pack(raw, mode)), raw)


@pytest.mark.parametrize("mode", ["mask128", "palette64"])
def test_fail_closed_on_corrupted_offsets_and_padding(mode: str) -> None:
    codec = importlib.import_module("scale_tiles")
    raw = np.zeros(128, dtype=np.uint8)
    raw[1] = 255
    if mode == "palette64":
        raw[2:5] = [1, 2, 3]
    packed = codec.pack(raw, mode)
    changed = packed.offsets.copy()
    changed[0] = 1
    with pytest.raises(ValueError, match="Invalid"):
        codec.unpack(replace(packed, offsets=changed))
    changed = packed.offsets.copy()
    changed[-1] += 4
    with pytest.raises(ValueError, match="Invalid"):
        codec.unpack(replace(packed, offsets=changed))
    if packed.exceptions.size:
        changed = packed.exceptions.copy()
        changed[-1] = 9
        with pytest.raises(ValueError, match="padding"):
            codec.unpack(replace(packed, exceptions=changed))


def test_distribution_selection_and_unsupported_input() -> None:
    codec = importlib.import_module("scale_tiles")
    for raw in (np.zeros(16384, dtype=np.uint8),
                np.resize(np.array([0, 1, 255], dtype=np.uint8), 16384)):
        chosen = codec.choose(raw)
        assert np.array_equal(codec.unpack(chosen), raw)
        assert chosen.allocated_bytes() < raw.nbytes // 2
    with pytest.raises(ValueError, match="U8"):
        codec.pack(np.ones(16, dtype=np.int16), "mask128")
    with pytest.raises(ValueError, match="contiguous"):
        codec.pack(np.arange(32, dtype=np.uint8)[::2], "mask128")


def test_sparse_exact_all_bytes_and_cross_chunk_keys() -> None:
    codec = importlib.import_module("scale_tiles")
    cases = [np.arange(256, dtype=np.uint8).repeat(16),
             np.resize(np.array([0, 127, 255], dtype=np.uint8), 16384)]
    raw = np.full(1024**2 + 32, 120, dtype=np.uint8)
    raw[16:32] = 119
    raw[32:48] = 121
    raw[[0, 1024**2 - 1, 1024**2, 1024**2 + 31]] = [0, 255, 7, 254]
    cases.append(raw)
    for raw in cases:
        packed = codec.pack_sparse(raw)
        assert np.array_equal(codec.unpack_sparse(packed), raw)
    packed = codec.pack_sparse(cases[-1])
    assert packed.palette == (119, 120, 121)
    assert packed.descriptor & (1 << 24) and not packed.descriptor & (1 << 25)
    assert packed.retained_bytes < raw.nbytes // 2
    selectors = packed.selectors.copy()
    selectors[selectors >> 30 == 3] = 0xffffffff
    with pytest.raises(ValueError, match="exception index"):
        codec.unpack_sparse(replace(packed, selectors=selectors))


def test_two_bit_affine_fragments_match_independent_stock_bf16_order() -> None:
    # Exhaust all 3**8 common-symbol vectors and compare with the independently
    # constructed BF16 exponent bits, including subnormals and maximum exponent.
    words = np.arange(65536, dtype=np.uint32)
    words = words[(words & (words >> 1) & 0x5555) == 0]
    codes = ((words[:, None] >> (np.arange(8, dtype=np.uint32) * 2)) & 3)
    order = np.array([0, 1, 4, 5, 2, 3, 6, 7])
    original = codes[:, order]
    spread = (words & 0xff) | ((words & 0xff00) << 8)
    for base in (0, 1, 119, 120, 253):
        fragments = np.stack([((spread << shift) & 0x01800180) + base * 0x00800080
                              for shift in (7, 5, 3, 1)], axis=1).astype(np.uint32)
        expected = ((original + base) << 7).astype(np.uint16)[:, [0, 2, 1, 3, 4, 6, 5, 7]]
        assert np.array_equal(fragments.view(np.uint16), expected)
