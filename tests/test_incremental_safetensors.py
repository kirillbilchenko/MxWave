"""Tests for bounded incremental safetensors emission."""

from __future__ import annotations

from io import BytesIO

import pytest
import torch

from mxwave.incremental_safetensors import IncrementalSafeTensorsWriter


def test_u8_writer_rejects_non_u8_destination() -> None:
    stream = BytesIO()
    writer = IncrementalSafeTensorsWriter(stream, {"value": ((2,), "I8")})

    with pytest.raises(TypeError, match="declared as I8, not U8"):
        writer.write_u8_chunk("value", torch.tensor([1, 2], dtype=torch.uint8))


def test_finish_rejects_incomplete_payload() -> None:
    stream = BytesIO()
    writer = IncrementalSafeTensorsWriter(stream, {"value": ((3,), "U8")})
    writer.write_u8_chunk("value", torch.tensor([1, 2], dtype=torch.uint8))

    with pytest.raises(ValueError, match="Incomplete safetensors payloads"):
        writer.finish()
