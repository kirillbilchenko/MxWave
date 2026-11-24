"""The qualified square-output exception must not expand input or unmeasured row routing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

from kda_row_routing import use_fused


def test_output_exception_preserves_input_and_all_unmeasured_rows() -> None:
    for tokens in range(1, 2049):
        for enabled in (False, True):
            assert use_fused(tokens, 12608, 4096, 12576, limit=8,
                             output_sixteen=enabled) == (tokens <= 8)
            assert use_fused(tokens, 4096, 4096, 4096, limit=8,
                             output_sixteen=enabled) == (
                                 tokens <= 8 or enabled and tokens == 16)


def test_unqualified_geometries_do_not_select_sixteen() -> None:
    for geometry in ((4096, 2048, 4096), (4160, 4096, 4096), (4096, 4096, 4000)):
        assert not use_fused(16, *geometry, limit=8, output_sixteen=True)
    assert not use_fused(8, 4096, 4096, 4096, limit=1, output_sixteen=True)
    assert use_fused(1, 4096, 4096, 4096, limit=1, output_sixteen=False)
    with pytest.raises(ValueError, match="audited"):
        use_fused(16, 4096, 4096, 4096, limit=16, output_sixteen=True)
    with pytest.raises(ValueError, match="positive"):
        use_fused(0, 4096, 4096, 4096, limit=8, output_sixteen=True)
