"""The qualified decoder must refuse every unsafe exponent base."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from zipserv_contract import require_qualified_exponent_base


@pytest.mark.parametrize("base", [249, 250, 251, 252, 253, 254, 255, -1, 256, True])
def test_qualified_loader_refuses_every_unsafe_exponent_base(base: int) -> None:
    with pytest.raises(ValueError, match="unsafe for the qualified decoder"):
        require_qualified_exponent_base(base, where="fixture")


@pytest.mark.parametrize("base", [0, 1, 115, 248])
def test_qualified_loader_accepts_safe_boundary_bases(base: int) -> None:
    require_qualified_exponent_base(base, where="fixture")
