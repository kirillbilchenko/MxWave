"""Qualify independently initialized entropy streams and their conservative storage accounting."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import the CPU screen without any GPU runtime."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_rans_exact_for_all_symbols_and_skewed_distributions() -> None:
    codec = importlib.import_module("screen_mxfp4_entropy")
    for data in (bytes(range(256)) * 8, b"\x00" * 8192,
                 b"\xff" * 65536 + bytes(range(256)), bytes(range(16)) * 257):
        table = codec.frequencies([data.count(i) for i in range(256)])
        assert sum(table) == 4096 and all(table[b] > 0 for b in data)
        payload = codec.encode(data, table)
        assert codec.decode(payload, len(data), table) == data
        with pytest.raises(ValueError, match="terminal|Truncated"):
            codec.decode(payload + b"x", len(data), table)
        with pytest.raises(ValueError, match="Invalid"):
            codec.decode(payload[:3], len(data), table)


def test_rans_uniform_nibbles_fall_back_and_constant_codes_compress() -> None:
    codec = importlib.import_module("screen_mxfp4_entropy")
    uniform = codec.screen(bytes(range(256)) * 8, nibble=True)
    assert uniform["entropy_bits_per_symbol"] == 4
    for result in uniform["stream_layouts"]:
        assert result["exact_roundtrip"] and result["raw_fallback_tiles"] == 1
        assert result["retained_bytes"] == 2048 + 32 + 5 + 4
        assert result["saving_fraction"] < 0
    for result in codec.screen(b"\x22" * 2048, nibble=True)["stream_layouts"]:
        assert result["saving_fraction"] > .8 and result["exact_roundtrip"]
    with pytest.raises(ValueError, match="complete"):
        codec.screen(b"\x00", nibble=True)
