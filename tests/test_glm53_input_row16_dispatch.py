"""An exact16 input exception must preserve every other measured route."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

from input_row16_dispatch import INPUT_GEOMETRY, select_route
from kda_row_routing import use_fused


def test_only_exact16_input_rows_are_added() -> None:
    geometries = [
        INPUT_GEOMETRY,
        (4096, 4096, 4096),
        (12608, 4096, 12608),
        (12576, 4096, 12576),
        (12608, 4095, 12576),
    ]
    for geometry in geometries:
        for rows in range(1, 34):
            for output in (False, True):
                previous = use_fused(rows, *geometry, limit=8, output_sixteen=output)
                result = select_route(use_fused, rows, *geometry, limit=8, output_sixteen=output)
                if result != previous:
                    assert rows == 16 and geometry == INPUT_GEOMETRY
                if rows == 16 and geometry == INPUT_GEOMETRY:
                    assert result


def test_invalid_scope_keeps_the_native_guards() -> None:
    for count, limit in ((0, 8), (-1, 8), (16, 16)):
        with pytest.raises(ValueError):
            select_route(use_fused, count, *INPUT_GEOMETRY, limit=limit, output_sixteen=True)
    assert not select_route(use_fused, 16, *INPUT_GEOMETRY, limit=1, output_sixteen=False)
