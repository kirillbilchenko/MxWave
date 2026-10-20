"""Verify numerical evidence independently of vLLM or a CUDA device."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import math
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def values(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import only the bounded CPU evidence helper."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    return importlib.import_module("decode_state_values")


def test_metrics_measure_bits_and_finite_error_separately(values: ModuleType) -> None:
    """A signed zero changes bits while a changed finite element contributes squared error."""
    reference = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.bfloat16)
    current = torch.tensor([-0.0, 1.0, 4.0, 3.0], dtype=torch.bfloat16)
    metric = values.value_metrics(current, reference)
    assert metric["changed_elements_by_bits"] == 2 and not metric["bytes_exact"]
    assert metric["max_absolute_error_finite"] == 2 and metric["rms_error_finite"] == 1
    assert metric["relative_l2_finite"] == pytest.approx(math.sqrt(4 / 14))
    assert metric["sqnr_db_finite"] == pytest.approx(10 * math.log10(14 / 4))


def test_metrics_keep_nan_payload_equality_and_count_nonfinite(values: ModuleType) -> None:
    """Nonfinite values are excluded explicitly; unchanged NaNs remain byte-identical."""
    reference = torch.tensor([0x7FC1, 0x7F80, 0x3F80], dtype=torch.uint16).view(torch.bfloat16)
    metric = values.value_metrics(reference.clone(), reference)
    assert metric["bytes_exact"] and metric["nonfinite_pairs"] == 2
    assert metric["finite_pairs"] == 1 and metric["rms_error_finite"] == 0


def test_raw_tensor_preserves_strides_and_bf16_bits(values: ModuleType, tmp_path: Path) -> None:
    """The manifest reconstructs the exact logical tensor without pickle or dtype conversion."""
    bits = torch.tensor([0x7FC1, 0x8000, 0x3F80, 0x7FC2], dtype=torch.uint16).reshape(2, 2)
    tensor = bits.view(torch.bfloat16).t()
    result = values.save_tensor(tmp_path, "hidden.bin.gz", tensor)
    raw = gzip.decompress((tmp_path / result["path"]).read_bytes())
    assert raw == tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    assert hashlib.sha256(raw).hexdigest() == result["sha256"]
    assert result["shape"] == [2, 2] and result["dtype"] == "torch.bfloat16"
    with pytest.raises(FileExistsError):
        values.save_tensor(tmp_path, "hidden.bin.gz", tensor)
    with pytest.raises(ValueError, match="basename"):
        values.save_tensor(tmp_path, "../hidden.bin.gz", tensor)


def test_metrics_cover_the_comparison_chunk_boundary(values: ModuleType) -> None:
    """Errors on either side of a chunk boundary contribute to one aggregate metric."""
    size = values.METRIC_CHUNK_ELEMENTS + 1
    reference = torch.ones(size, dtype=torch.float32)
    current = reference.clone()
    current[-2:] = 2
    metric = values.value_metrics(current, reference)
    assert metric["changed_elements_by_bits"] == 2
    assert metric["rms_error_finite"] == pytest.approx(math.sqrt(2 / size))
